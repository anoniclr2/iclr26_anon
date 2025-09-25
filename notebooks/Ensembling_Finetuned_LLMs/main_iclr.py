"""Check mini vs extended"""

#imports
import numpy as np
import time
import argparse

import torch
import torch.nn.functional as F
import re
import os
import sys
import csv
sys.path.append('../../src')                                    #needed for calibrator
sys.path.append('finetuning_text_classifiers')
#sys.path.append(os.join(os.getcwd(), '/finetuning_text_classifiers'))    #for metadataset
#from my code base
from calibrator import PrecomputedCalibrator
from llm_helper_fn import create_new_split, retrieve_data
# for the data
from metadataset.ftc.metadataset import FTCMetadataset





def main(data_version="extended"):
    """Args: data_version: "mini" or "extended" """
    #get the data
    date = '2025-09-23' #NOTE change this when you change the code
    print('running main')
    num_seeds = 5
    data_dir = "../data"

    print("Choosing data version: ", data_version)
    metadataset = FTCMetadataset(data_dir=str(data_dir), 
                                 metric_name="error",
                                 data_version=data_version)
    dataset_names = metadataset.get_dataset_names()
    splits = ["valid", "test"]      # based of github
    print('data loaded', dataset_names, flush=True)

    #['imdb', 'mteb/tweet_sentiment_extraction', 'ag_news', 'dbpedia_14', 'stanfordnlp/sst2', 'SetFit/mnli']
    #datasets = dataset_names[:1]            # only imdb for debugging
    datasets = dataset_names
    print("Datasets to process:", datasets, flush=True)

    # Ensure an output directory for the arrays
    arr_dir = "llm_experiments_data/arrays"
    os.makedirs(arr_dir, exist_ok=True)

    output_file = f'llm_experiments_data/llm_experimental_results_{data_version}_iclr_{date}.csv'
    header = ['dataset', 'seed', 'method', 'ensemble_type', 'ensemble_size', 'ensemble_unique_size',
              'nll_test', 'c1', 'c2', 'epi_scalar', 'ensemble_time', 'calibration_time', 'path']

    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    # Open the output file in append mode.
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        #Loop over the datasets
        for dataset in datasets:
            print("Processing dataset:", dataset)
            results = retrieve_data(metadataset, dataset, splits)
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
                # NOTE only JUCAL
                for method in ["pure_logits"]:
                    print("Processing method:", method)
                    calibrator = PrecomputedCalibrator(adjusting_alpha_method=method, 
                                           clamping_alphas=False, 
                                           logits_based_adjustments=True)
                    # --- Ensemble Selection ---
                    #top_n_indices, _ = calibrator.top_n_ensemble(member_probs=val_member_probs,
                    #                                            labels=val_labels, N=10, eps=1e-12)

                    #greedy_5_indices, _ = calibrator.greedy_ensemble(member_probs=val_member_probs, 
                    #                                            labels=val_labels, m=5, no_resample=False)
                    # Time ensemble selection methods
                    greedy_5_time = time.time()
                    greedy_5_indices, _ = calibrator.greedy_ensemble(member_probs=val_member_probs,
                                                                labels=val_labels, m=5, no_resample=False)
                    greedy_5_time = time.time() - greedy_5_time
                    
                    greedy_50_time = time.time()
                    greedy_50_indices, _ = calibrator.greedy_ensemble(member_probs=val_member_probs, 
                                                                labels=val_labels, m=50, no_resample=False)
                    greedy_50_time = time.time() - greedy_50_time

                    if method == 'convex_comb':
                        c2_vals = np.linspace(0, 3, 50)
                    elif method == 'pure_logits':
                        c2_vals = np.linspace(0, 10, 50)
                    elif method == 'convex_comb_no_exp':
                        c2_vals = np.linspace(0, 8, 50)
                    elif method == 'convex_comb_global':
                        c2_vals = np.linspace(0, 1, 50)
                    # NOTE universal for all methods
                    temps = np.linspace(0.3, 3, 50)
                    epi_scalar_vals = np.array([1])

                    # NOTE these methods are currently not of interest
                    #greedy_indices_calib_subset, _ = calibrator.greedy_ensemble_calibrated_subset(
                    #    member_probs=val_member_probs, labels=val_labels, m=50, subset_size = 10, seed= 42, no_resample=False, 
                    #    tolerance=3, eps=1e-12,
                    #    c1_vals=temps, c2_vals=c2_vals, epi_scalar_vals=epi_scalar_vals)

                    #greedy_indices_calib_once, _ = calibrator.greedy_ensemble_calibrated_once(
                    #    member_probs=val_member_probs, labels=val_labels, m=50, no_resample=False, 
                    #    tolerance=3, eps=1e-12,
                    #    c1_vals=temps, c2_vals=c2_vals, epi_scalar_vals=epi_scalar_vals)

                    #greedy_indices_calib, _ = calibrator.greedy_ensemble_recalibrated(
                    #    member_probs=val_member_probs, labels=val_labels, m=50, no_resample=False, 
                    #    tolerance=3, eps=1e-12,
                    #    c1_vals=temps, c2_vals=c2_vals, epi_scalar_vals=epi_scalar_vals)
                    


                    # NOTE need to be carful with baseline and temp_baseline naming!!!
                    ensemble_methods = { "greedy_5_baseline": greedy_5_indices,
                                         "greedy_50_baseline": greedy_50_indices,
                                         "greedy_50_temp_pool_then_calibrate": greedy_50_indices,
                                         "greedy_5_temp_pool_then_calibrate": greedy_5_indices,
                                         # Calibrate-then-pool methods
                                         "greedy_50_temp_calibrate_then_pool": greedy_50_indices,
                                         "greedy_5_temp_calibrate_then_pool": greedy_5_indices,
                                         #Will apply JUCAL on these
                                         "greedy_5_post_calib": greedy_5_indices,
                                         "greedy_50_post_calib": greedy_50_indices,     
                                         #"greedy_50_calib_once": greedy_indices_calib_once,
                                         #"greedy_50_calib_every_step": greedy_indices_calib,
                                        }


                    for ens_method, indices in ensemble_methods.items():
                        # Update validation and test ensemble probabilities based on the selected indices
                        val_probs_ens = val_member_probs[indices]
                        test_probs_ens = test_member_probs[indices]

                        # Determine which ensemble time to use based on method
                        if "greedy_5" in ens_method:
                            current_ensemble_time = greedy_5_time
                        else:
                            current_ensemble_time = greedy_50_time

                        # Process based on calibration method
                        if ens_method.endswith("baseline"):
                            # Pure baseline - no calibration at all
                            c1_prim = None
                            c2_prim = None
                            epi_scalar_prim = None
                            calib_time = 0.0
                            
                            # Evaluate baseline (non-calibrated) ensemble NLL on the test set
                            nll = calibrator.compute_ensemble_nll(None, test_probs_ens, test_labels)
                            ensemble_probabilities = test_probs_ens.mean(dim=0)
                            ensemble_probabilities = ensemble_probabilities.cpu().numpy()
                        elif ens_method.endswith("temp_pool_then_calibrate"):
                            # Pool-then-calibrate temperature scaling
                            c1_prim = None
                            c2_prim = None
                            epi_scalar_prim = None
                            
                            # Time temperature calibration
                            calib_start_time = time.time()
                            mean_val_probs = val_probs_ens.mean(dim=0)
                            mean_val_logits = torch.log(mean_val_probs + 1e-12)
                            best_temp = calibrator.find_optimal_temperature(mean_val_logits, val_labels, temps)
                            c1_prim = best_temp
                            calib_time = time.time() - calib_start_time
                            
                            # Apply to test set
                            mean_test_probs_ens = test_probs_ens.mean(dim=0)
                            mean_test_logits_ens = torch.log(mean_test_probs_ens + 1e-12)
                            nll, ensemble_probabilities = calibrator.nll_at_T(mean_test_logits_ens, test_labels, best_temp)
                            ensemble_probabilities = ensemble_probabilities.cpu().numpy()
                        elif ens_method.endswith("temp_calibrate_then_pool"):
                            # JUCAL Calibrate-then-pool with c2=0 (single temperature for all members)
                            c1_prim = None
                            c2_prim = None  # Fixed at 0 for calibrate-then-pool behavior
                            epi_scalar_prim = None

                            # Time JUCAL calibration with c2=1
                            calib_start_time = time.time()
                            # Use JUCAL grid search but with c2 fixed at 1
                            c2_vals_fixed = np.array([1])  # Only c2=1 for calibrate-then-pool
                            _, best_params = calibrator.grid_search_c1_c2_precomputed_coarse_to_fine(
                                val_probs_ens, val_labels, temps, c2_vals_fixed, epi_scalar_vals
                            )
                            calib_time = time.time() - calib_start_time
                            
                            c1_prim = best_params['c1']
                            c2_prim = best_params['c2']  # Should be 0
                            epi_scalar_prim = best_params['epi_scalar']

                            # Apply JUCAL prediction with the optimized parameters
                            calibrator_results = calibrator.predict(test_probs_ens, c1_prim, c2_prim, epi_scalar_prim, 
                                                                    test_labels)
                            # Extract calibrated NLL (ensure a scalar by taking the mean over samples)
                            nll = calibrator_results['nll'].mean()
                            ensemble_probabilities = calibrator_results['ensemble_probs']  #returns a numpy array
                        else:
                            # Time ONLY the calibration (grid search), not prediction
                            calib_start_time = time.time()
                            # Evaluate non-baseline ensembles    
                            _, best_params = calibrator.grid_search_c1_c2_precomputed_coarse_to_fine(val_probs_ens, val_labels, 
                                                                                    temps, c2_vals, epi_scalar_vals)
                            calib_time = time.time() - calib_start_time
                            
                            c1_prim = best_params['c1']
                            c2_prim = best_params['c2']
                            epi_scalar_prim = best_params['epi_scalar']

                            calibrator_results = calibrator.predict(test_probs_ens, c1_prim, c2_prim, epi_scalar_prim, 
                                                                    test_labels)
                            # Extract calibrated NLL (ensure a scalar by taking the mean over samples)
                            nll = calibrator_results['nll'].mean()
                            ensemble_probabilities = calibrator_results['ensemble_probs']  #returns a numpy array

                        # Store experimental results for this ensemble type
                        safe_dataset = re.sub(r"[^a-zA-Z0-9_]", "_", dataset).strip("_").lower() #this strip is needed to remove leading and trailing underscores
                        base = f"{data_version}_{safe_dataset}_{seed}_{method}_{ens_method}_{date}"
                        experiment_path  = os.path.join(arr_dir, base + ".npz")

                        np.savez(
                            experiment_path,
                            ensemble_indices = indices,
                            ensemble_probs = ensemble_probabilities,
                            labels = test_labels.cpu().numpy()
                                 )

                        experimental_results = {
                                            'dataset': dataset,
                                            'seed': seed,
                                            'method': method,
                                            'ensemble_type': ens_method,
                                            'ensemble_size': len(indices),
                                            'ensemble_unique_size': len(set(indices)),
                                            'nll_test': float(nll),
                                            'c1': c1_prim,
                                            'c2': c2_prim,
                                            'epi_scalar': epi_scalar_prim,
                                            'ensemble_time': current_ensemble_time,
                                            'calibration_time': calib_time,
                                            'path': experiment_path
                                            #additional keys would get dropped
                                            }
  
                        #print(ensemble_probabilities.shape, flush=True)
                        row = {col: experimental_results[col] for col in header}
                        writer.writerow(row)
                        f.flush()  # Flush the file so results are immediately written to disk.
                        print("Wrote result for dataset:", dataset, 
                              "seed:", seed, "method:", method, 
                              "ensemble:", ens_method, flush=True)

    print("Code executed successfully.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Experiments")
    parser.add_argument('--data_version', type=str, choices=['mini', 'extended'], default='extended',
                        help='Specify the data version to use: "mini" or "extended"')
    args = parser.parse_args()
    main(data_version=args.data_version)
    print("Main done")