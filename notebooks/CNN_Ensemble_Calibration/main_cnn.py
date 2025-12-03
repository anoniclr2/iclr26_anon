"""
CNN Ensemble Calibration Experiments

Applies different calibration methods to pre-computed CNN ensemble predictions
for MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and MNIST hyperparameter datasets.
"""

import numpy as np
import time
import argparse
import torch
import re
import os
import sys
import csv
import gc
from typing import Dict

# Add src to path for imports
sys.path.append('../../src')
from calibrator import PrecomputedCalibrator
from cnn_data_utils import (
    load_cnn_predictions_single_split,
    load_cifar100_split,
    get_num_splits,
    get_dataset_paths
)


def main(dataset_name: str = "all") -> None:
    """
    Run CNN ensemble calibration experiments.

    Args:
        dataset_name: Which dataset to process or 'all' for all datasets
    """
    date = '2025-11-30'  # NOTE: change this when you change the code
    print('Running CNN ensemble calibration experiments')

    # Get dataset paths
    dataset_paths = get_dataset_paths()

    # Determine which datasets to process
    if dataset_name == "all":
        datasets_to_process = list(dataset_paths.items())
    elif dataset_name in dataset_paths:
        datasets_to_process = [(dataset_name, dataset_paths[dataset_name])]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_paths.keys())} or 'all'")

    print(f"Datasets to process: {[d[0] for d in datasets_to_process]}", flush=True)

    # Ensure output directories exist
    arr_dir = "cnn_experiments_data/arrays"
    os.makedirs(arr_dir, exist_ok=True)

    output_file = f'cnn_experiments_data/cnn_experimental_results_iclr_{date}.csv'
    header = ['dataset', 'split_idx', 'method', 'ensemble_type', 'ensemble_size', 'ensemble_unique_size',
              'nll_test', 'c1', 'c2', 'epi_scalar', 'ensemble_time', 'calibration_time', 'path']

    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    # Open the output file in append mode
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)

        # Loop over datasets
        for dataset, json_path in datasets_to_process:
            print(f"\n{'='*80}")
            print(f"Processing dataset: {dataset}")
            print(f"{'='*80}")

            # Get number of splits without loading all data
            num_splits = get_num_splits(str(json_path))
            print(f"Found {num_splits} splits for {dataset}")

            # Loop over splits - load one at a time to save memory
            for split_idx in range(num_splits):
                print(f"\nProcessing split: {split_idx} out of {num_splits}")

                # Load only this split to minimize memory usage
                # For CIFAR100, use individual split files if available
                if dataset == "CIFAR100":
                    val_member_probs, val_labels, test_member_probs, test_labels = (
                        load_cifar100_split(split_idx)
                    )
                else:
                    val_member_probs, val_labels, test_member_probs, test_labels = (
                        load_cnn_predictions_single_split(str(json_path), split_idx)
                    )

                print(f"Val shape: {val_member_probs.shape}, Test shape: {test_member_probs.shape}")

                # NOTE: only JUCAL (pure_logits method)
                for method in ["pure_logits"]:
                    print(f"Processing method: {method}")
                    calibrator = PrecomputedCalibrator(
                        adjusting_alpha_method=method,
                        clamping_alphas=False,
                        logits_based_adjustments=True
                    )

                    # --- Ensemble Selection ---
                    # Time ensemble selection methods
                    greedy_5_time = time.time()
                    greedy_5_indices, _ = calibrator.greedy_ensemble(
                        member_probs=val_member_probs,
                        labels=val_labels,
                        m=5,
                        no_resample=False
                    )
                    greedy_5_time = time.time() - greedy_5_time
                    print(f"Greedy-5 selection: {greedy_5_indices} (time: {greedy_5_time:.2f}s)")

                    # Parameter ranges (same as LLM experiments)
                    if method == 'pure_logits':
                        c2_vals = np.linspace(0, 10, 50)
                    else:
                        c2_vals = np.linspace(0, 3, 50)

                    temps = np.linspace(0.3, 3, 50)
                    epi_scalar_vals = np.array([1])

                    # Define ensemble methods
                    ensemble_methods = {
                        "greedy_5_baseline": greedy_5_indices,
                        "greedy_5_temp_pool_then_calibrate": greedy_5_indices,
                        "greedy_5_temp_calibrate_then_pool": greedy_5_indices,
                        "greedy_5_post_calib": greedy_5_indices,  # JUCAL
                    }

                    # Process each ensemble method
                    for ens_method, indices in ensemble_methods.items():
                        print(f"\n  Method: {ens_method}")

                        # Update validation and test ensemble probabilities based on selected indices
                        val_probs_ens = val_member_probs[indices]
                        test_probs_ens = test_member_probs[indices]

                        # All methods use greedy_5 timing
                        current_ensemble_time = greedy_5_time

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

                            print(f"    Best temperature: {best_temp:.4f}")

                            # Apply to test set
                            mean_test_probs_ens = test_probs_ens.mean(dim=0)
                            mean_test_logits_ens = torch.log(mean_test_probs_ens + 1e-12)
                            nll, ensemble_probabilities = calibrator.nll_at_T(mean_test_logits_ens, test_labels, best_temp)
                            ensemble_probabilities = ensemble_probabilities.cpu().numpy()

                        elif ens_method.endswith("temp_calibrate_then_pool"):
                            # Calibrate-then-pool with temperature scaling (c2=1)
                            c1_prim = None
                            c2_prim = None
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
                            c2_prim = best_params['c2']
                            epi_scalar_prim = best_params['epi_scalar']

                            print(f"    Best c1: {c1_prim:.4f}, c2: {c2_prim:.4f}")

                            # Apply JUCAL prediction with the optimized parameters
                            calibrator_results = calibrator.predict(
                                test_probs_ens, c1_prim, c2_prim, epi_scalar_prim, test_labels
                            )
                            # Extract calibrated NLL (ensure a scalar by taking the mean over samples)
                            nll = calibrator_results['nll'].mean()
                            ensemble_probabilities = calibrator_results['ensemble_probs']  # returns a numpy array

                        else:  # post_calib (Full JUCAL)
                            # Time ONLY the calibration (grid search), not prediction
                            calib_start_time = time.time()
                            # Evaluate non-baseline ensembles
                            _, best_params = calibrator.grid_search_c1_c2_precomputed_coarse_to_fine(
                                val_probs_ens, val_labels, temps, c2_vals, epi_scalar_vals
                            )
                            calib_time = time.time() - calib_start_time

                            c1_prim = best_params['c1']
                            c2_prim = best_params['c2']
                            epi_scalar_prim = best_params['epi_scalar']

                            print(f"    Best c1: {c1_prim:.4f}, c2: {c2_prim:.4f}")

                            calibrator_results = calibrator.predict(
                                test_probs_ens, c1_prim, c2_prim, epi_scalar_prim, test_labels
                            )
                            # Extract calibrated NLL (ensure a scalar by taking the mean over samples)
                            nll = calibrator_results['nll'].mean()
                            ensemble_probabilities = calibrator_results['ensemble_probs']  # returns a numpy array

                        # Store experimental results for this ensemble type
                        safe_dataset = re.sub(r"[^a-zA-Z0-9_]", "_", dataset).strip("_").lower()
                        base = f"{safe_dataset}_{split_idx}_{method}_{ens_method}_{date}"
                        experiment_path = os.path.join(arr_dir, base + ".npz")

                        np.savez(
                            experiment_path,
                            ensemble_indices=indices,
                            ensemble_probs=ensemble_probabilities,
                            labels=test_labels.cpu().numpy()
                        )

                        experimental_results = {
                            'dataset': dataset,
                            'split_idx': split_idx,
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
                        }

                        # Write row to CSV
                        row = {col: experimental_results[col] for col in header}
                        writer.writerow(row)
                        f.flush()  # Flush the file so results are immediately written to disk

                        print(f"    Test NLL: {nll:.4f}, Calibration time: {calib_time:.2f}s")
                        print(f"    Wrote result for dataset: {dataset}, "
                              f"split: {split_idx}, method: {method}, "
                              f"ensemble: {ens_method}", flush=True)

                        # Clean up tensors from this ensemble method to free memory
                        del val_probs_ens, test_probs_ens, ensemble_probabilities

                        # Force garbage collection after processing all ensemble methods
                        gc.collect()

    print("\n" + "="*80)
    print("Code executed successfully.")
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN Ensemble Calibration Experiments")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['MNIST', 'FMNIST', 'CIFAR', 'CIFAR100', 'MNIST10Epochs10_5L2Reg', 'MNIST40Epochs10_4L2Reg', 'all'],
        default='all',
        help='Specify which dataset to process or "all" for all datasets'
    )
    args = parser.parse_args()
    main(dataset_name=args.dataset)
    print("Main done")
