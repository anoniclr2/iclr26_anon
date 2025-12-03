"""Check mini vs extended"""

#imports
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
import csv
sys.path.append('../../src')                                    #needed for calibrator
sys.path.append('finetuning_text_classifiers')
#sys.path.append(os.join(os.getcwd(), '/finetuning_text_classifiers'))    #for metadataset
#from my code base
from calibrator import PrecomputedCalibrator
from metadataset.ftc.metadataset import FTCMetadataset

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

def temperature_scaled_nll(member_probs, labels, temperature, eps=1e-12):
    logits = torch.log(member_probs + eps)
    scaled_logits = logits / temperature
    scaled_probs = F.softmax(scaled_logits, dim=-1)
    nll = -torch.gather(torch.log(scaled_probs + eps), 2, labels.unsqueeze(0).unsqueeze(2)).squeeze(2)
    return nll.mean().item()

# used to evaluate non calibrated ensembles
def compute_ensemble_nll(member_probs, labels, eps=1e-12):
    ensemble_probs = member_probs.mean(dim=0)  # [num_samples, num_classes]
    nll = -torch.gather(torch.log(ensemble_probs + eps), 1, labels.unsqueeze(1)).squeeze(1)
    return nll.mean().item()

# END HELPER functions --------------------------------------------------

def main():
    #get the data
    print('running main')
    output_file = 'llm_experimental_results_mini_v6.csv'
    num_seeds = 3
    data_dir = "../data"
    data_version = "mini"                                   #10% of total with 20% val split
    #data_version = "extended"                              # NOTE not yet downloaded
    print("Choosing data version: ", data_version)
    metadataset = FTCMetadataset(data_dir=str(data_dir), 
                                 metric_name="error",
                                 data_version=data_version)
    dataset_names = metadataset.get_dataset_names()
    splits = ["valid", "test"]      # based of github
    print('data loaded', dataset_names, flush=True)
    #parameters for experiements
    num_seeds = 3                           # ish number of seeds
    #datasets = dataset_names[:1]            # number of datasets
    datasets = dataset_names[5:]

    output_file = 'llm_experimental_results_mini_v6.csv'
    header = ['dataset', 'seed', 'method', 'ensemble_type', 'ensemble_size', 'nll_test', 'c1', 'c2', 'epi_scalar']
    
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    # Open the output file in append mode.
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        #Loop over the datasets
        for name in datasets:
            print("Processing dataset:", name)
            results = retrieve_data(metadataset, name, splits)
            default_val_member_probs = results['valid'][2]
            default_val_labels = results['valid'][3]
            default_test_member_probs = results['test'][2]
            default_test_labels = results['test'][3]

            for seed in range(num_seeds):
                print(f"Processing seed: {seed} out of {num_seeds}")
                if seed == 0:
                    # Use the original split
                    val_member_probs = default_val_member_probs
                    val_labels = default_val_labels
                    test_member_probs = default_test_member_probs
                    test_labels = default_test_labels
                else:
                    val_member_probs, val_labels, test_member_probs, test_labels = create_new_split(val_member_probs, val_labels, 
                                                                                            test_member_probs, test_labels,
                                                                                             seed)
                for method in ["convex_comb", "pure_logits", "convex_comb_no_exp", "convex_comb_global"]:
                    print("Processing method:", method)
                    calibrator = PrecomputedCalibrator(adjusting_alpha_method=method, 
                                           clamping_alphas=False, 
                                           logits_based_adjustments=True)
                    # --- Ensemble Selection Methods ---
                    #top_n_indices, _ = calibrator.top_n_ensemble(member_probs=val_member_probs,
                    #                                            labels=val_labels, N=10, eps=1e-12)

                    greedy_5_indices, _ = calibrator.greedy_ensemble(member_probs=val_member_probs, 
                                                                labels=val_labels, m=5, no_resample=False)

                    greedy_50_indices, _ = calibrator.greedy_ensemble(member_probs=val_member_probs, 
                                                                labels=val_labels, m=50, no_resample=False)
                    # NOTE if we can afford it we increase the grid
                    if method == 'convex_comb':
                        c2_vals = np.linspace(0, 4, 50)
                    elif method == 'pure_logits':
                        c2_vals = np.linspace(0, 10, 50)
                    elif method == 'convex_comb_no_exp':
                        c2_vals = np.linspace(0, 8, 50)
                    elif method == 'convex_comb_global':
                        c2_vals = np.linspace(0, 1, 50)
                    temps = np.linspace(0.3, 3, 50)
                    epi_scalar_vals = np.array([1])

                    # put subset first to debugg
                    #greedy_indices_calib_subset, _ = calibrator.greedy_ensemble_calibrated_subset(
                    #    member_probs=val_member_probs, labels=val_labels, m=50, subset_size = 10, seed= 42, no_resample=False, 
                    #    tolerance=3, eps=1e-12,
                    #    c1_vals=temps, c2_vals=c2_vals, epi_scalar_vals=epi_scalar_vals)

                    greedy_indices_calib_once, _ = calibrator.greedy_ensemble_calibrated_once(
                        member_probs=val_member_probs, labels=val_labels, m=50, no_resample=False, 
                        tolerance=3, eps=1e-12,
                        c1_vals=temps, c2_vals=c2_vals, epi_scalar_vals=epi_scalar_vals)
                    
                    #greedy_indices_calib, _ = calibrator.greedy_ensemble_recalibrated(
                    #    member_probs=val_member_probs, labels=val_labels, m=50, no_resample=False, 
                    #    tolerance=3, eps=1e-12,
                    #    c1_vals=temps, c2_vals=c2_vals, epi_scalar_vals=epi_scalar_vals)
                    


                    # Create a dictionary for the three ensemble selection methods.
                    ensemble_methods = { "greedy_5_baseline": greedy_5_indices,
                                         "greedy_50_baseline": greedy_50_indices,
                                         "greedy_50_temp_baseline": greedy_50_indices,
                                         #these are always post calibrated
                                         "greedy_5_post_calib": greedy_5_indices,
                                         "greedy_50_post_calib": greedy_50_indices,     
                                         "greedy_50_calib_once": greedy_indices_calib_once,
                                        }


                    for ens_method, indices in ensemble_methods.items():
                        # Update validation and test ensemble probabilities based on the selected indices
                        val_probs_ens = val_member_probs[indices]
                        test_probs_ens = test_member_probs[indices]

                        if ens_method in ["greedy_5_baseline", "greedy_50_baseline", "greedy_50_temp_baseline"]:
                            c1_prim = None
                            c2_prim = None
                            epi_scalar_prim = None

                            if ens_method == "greedy_50_temp_baseline":
                                _, best_temp = calibrator.temperature_scale(val_probs_ens, val_labels, temps)
                                #update c1_prim
                                c1_prim = best_temp
                                nll = temperature_scaled_nll(test_probs_ens, test_labels, best_temp)
                            else:
                                # Evaluate baseline (non-calibrated) ensemble NLL on the test set
                                nll = compute_ensemble_nll(test_probs_ens, test_labels)
                        else:
                            # Evaluate non-baseline ensembles    
                            _, best_params = calibrator.grid_search_c1_c2_precomputed_coarse_to_fine(val_probs_ens, val_labels, 
                                                                                    temps, c2_vals, epi_scalar_vals)
                            c1_prim = best_params['c1']
                            c2_prim = best_params['c2']
                            epi_scalar_prim = best_params['epi_scalar']

                            calibrator_results = calibrator.predict(test_probs_ens, c1_prim, c2_prim, epi_scalar_prim, 
                                                                    test_labels)
                            # Extract calibrated NLL (ensure a scalar by taking the mean over samples)
                            nll = calibrator_results['nll'].mean()

                        # Store experimental results for this ensemble type
                        experimental_results = {'dataset': name,
                                             'seed': seed,
                                             'method': method,
                                             'ensemble_type': ens_method,
                                             'ensemble_size': len(indices),
                                             'nll_test': nll,
                                             'c1': c1_prim,
                                             'c2': c2_prim,
                                             'epi_scalar': epi_scalar_prim
                                             }
                        writer.writerow(experimental_results)
                        f.flush()  # Flush the file so results are immediately written to disk.
                        print("Wrote result for dataset:", name, 
                              "seed:", seed, "method:", method, 
                              "ensemble:", ens_method, flush=True)

    print("Code executed successfully.", flush=True)


if __name__ == "__main__":
    main()
    print("Main done")