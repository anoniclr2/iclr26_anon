"""Check for mini or extended"""


#all imports
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
import sys
sys.path.append('../../src')                                    #needed for calibrator
sys.path.append('finetuning_text_classifiers')
#sys.path.append(os.join(os.getcwd(), '/finetuning_text_classifiers'))    #for metadataset
#from my code base
from calibrator import PrecomputedCalibrator
from metadataset.ftc.metadataset import FTCMetadataset
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed


# START HELPER functions ------------------------------------------------

def create_new_split(val_probs, val_labels, test_probs, test_labels, seed):
    """
    Creates a new validation/test split by randomly shuffling the union of the default splits.
    
    Args:
        shapes: [num_models, num_samples, num_classes]
    Outputs:
        shapes: [num_models, num_samples, num_classes]
    """
    # Combine along the sample dimension (assumed to be 0)
    combined_probs = torch.cat([val_probs, test_probs], dim=1)
    combined_labels = torch.cat([val_labels, test_labels], dim=0)
    total = combined_labels.shape[0]
    val_count = val_labels.shape[0]
    
    # Create a permutation using numpy's random generator with the given seed.
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(total)
    
    # Compute the new split sizes
    new_val_indices = permuted_indices[:val_count]
    new_test_indices = permuted_indices[val_count:]
    
    new_val_member_probs = combined_probs[:,new_val_indices,:]
    new_val_labels = combined_labels[new_val_indices]
    new_test_member_probs = combined_probs[:,new_test_indices,:]
    new_test_labels = combined_labels[new_test_indices]
    
    return new_val_member_probs, new_val_labels, new_test_member_probs, new_test_labels

def retrieve_data(metadataset, dataset_name, splits):
    """
       hp_candidates, indices, predictions, targets 
    """
    results = {}
    for split in splits:
        metadataset.set_state(dataset_name=dataset_name,
                        split=split)
        hp_candidates, indices = metadataset._get_hp_candidates_and_indices()
        predictions = metadataset.get_predictions(indices)
        targets = metadataset.get_targets()
        results[split] = (hp_candidates, indices, predictions, targets)
    return results

# used to evaluate non calibrated ensembles
def compute_ensemble_nll(member_probs, labels, eps=1e-12):
    ensemble_probs = member_probs.mean(dim=0)  # [num_samples, num_classes]
    nll = -torch.gather(torch.log(ensemble_probs + eps), 1, labels.unsqueeze(1)).squeeze(1)
    return nll.mean().item()




# END HELPER functions --------------------------------------------------




def run_dataset_experiments(results, num_seeds=2):
    print('task started')
    default_val_member_probs = results['valid'][2]
    default_val_labels = results['valid'][3]
    default_test_member_probs = results['test'][2]
    default_test_labels = results['test'][3]

    for seed in range(num_seeds):
        if seed == 0:
            val_member_probs = default_val_member_probs
            val_labels = default_val_labels
            test_member_probs = default_test_member_probs
            test_labels = default_test_labels
        else:
            # Create a new split based on the default splits.
            val_member_probs, val_labels, test_member_probs, test_labels = create_new_split(
                default_val_member_probs, default_val_labels,
                default_test_member_probs, default_test_labels,
                seed
            )
        print('ready with data')
        for method in ["convex_comb", "pure_logits"]:
            calibrator = PrecomputedCalibrator(adjusting_alpha_method=method, 
                                               clamping_alphas=False, 
                                               logits_based_adjustments=True)
            # --- Ensemble Selection Methods ---
            print('start to ensemble')
            greedy_indices, _ = calibrator.greedy_ensemble(
                member_probs=val_member_probs, 
                labels=val_labels, m=50, no_resample=False
            )
            print('first ensembel done')
            greedy_init_indices = calibrator.greedy_ensemble_with_initial(
                member_probs=val_member_probs, 
                labels=val_labels, m=50, no_resample=False, 
                tolerance=3, eps=1e-12
            )
            
            if method == 'convex_comb':
                c2_vals = np.linspace(0, 3, 100)
            else:
                c2_vals = np.linspace(0, 10, 100)
            temps = np.linspace(0.5, 2, 50)
            epi_scalar_vals = np.array([1])
            greedy_init_temp_indices, _ = calibrator.greedy_ensemble_with_initial_and_temp(
                member_probs=val_member_probs, 
                labels=val_labels, m=50, init_N=5, no_resample=False, 
                tolerance=3, eps=1e-12, c1_vals=temps, c2_vals=c2_vals, 
                epi_scalar_vals=epi_scalar_vals
            )
            print('done ensembling')
            # Create a dictionary for the three ensemble selection methods.
            ensemble_methods = {"greedy": greedy_indices,
                                "greedy_init": greedy_init_indices,
                                "greedy_init_temp": greedy_init_temp_indices}
            for ens_method, indices in ensemble_methods.items():
                val_probs_ens = val_member_probs[indices]
                test_probs_ens = test_member_probs[indices]

                # Grid search calibration using the validation ensemble probabilities.
                _, best_params = calibrator.grid_search_c1_c2_precomputed( val_probs_ens, val_labels, 
                                                                        temps, c2_vals, epi_scalar_vals)
                c1_prim = best_params['c1']
                c2_prim = best_params['c2']
                epi_scalar_prim = best_params['epi_scalar']

                # Apply calibration on the test ensemble
                calibrator_results = calibrator.predict(
                    test_probs_ens, c1_prim, c2_prim, epi_scalar_prim, test_labels
                )
                calibrated_nll = calibrator_results['nll'].mean()
                non_calibrated_nll = compute_ensemble_nll(test_probs_ens, test_labels)

                local_results.append({
                    'dataset': dataset,
                    'seed': seed,
                    'method': method,
                    'ensemble_type': ens_method,
                    'ensemble_size': len(indices),
                    'non_calibrated_nll': non_calibrated_nll,
                    'calibrated_nll': calibrated_nll,
                    'c1': c1_prim,
                    'c2': c2_prim,
                    'epi_scalar': epi_scalar_prim
                })
                print('loop done')
    return local_results

def main():
    print('started')
    data_dir = "../data"
    # NOTE "extended" "mini"
    data_version = "mini"               
    splits = ["valid", "test"]
    
    # Create a master instance to obtain dataset names.
    metadataset = FTCMetadataset(data_dir=str(data_dir), 
                                        metric_name="error",
                                        data_version=data_version)
    dataset_names = metadataset.get_dataset_names()
    datasets = {}

    for name in dataset_names:
        print("Processing dataset:", name)
        results = retrieve_data(metadataset, name, splits)
        datasets[name] = results
    
    output_file = 'llm_experimental_results.csv'
    header = ['dataset', 'seed', 'method', 'ensemble_type', 'ensemble_size', 
              'non_calibrated_nll', 'calibrated_nll', 'c1', 'c2', 'epi_scalar']
    
    # Write header if the file doesn't exist.
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
    
    all_results = []
    # Use ProcessPoolExecutor to parallelize by dataset.
    with ProcessPoolExecutor(max_workers=len(dataset_names)) as executor:
        # Submit one task per dataset.
        futures = {executor.submit(run_dataset_experiments, datasets[name]): name 
                   for name in datasets}
        for future in as_completed(futures):
            try:
                dataset_results = future.result()  # A list of dictionaries for this dataset
                all_results.extend(dataset_results)
                # Append results to file as they come in.
                with open(output_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    for row in dataset_results:
                        writer.writerow(row)
            except Exception as exc:
                print(f"Dataset {futures[future]} generated an exception: {exc}")

    print("Parallel execution completed.")

if __name__ == "__main__":
    main()
