"""Calculate uncertainties for ICLR 2025 experiments - JUCAL Greedy-50 and Greedy-5"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# add project paths for imports
sys.path.append('../../src')  # for PrecomputedCalibrator
sys.path.append('finetuning_text_classifiers')

from calibrator import PrecomputedCalibrator
from llm_helper_fn import create_new_split, retrieve_data
from metadataset.ftc.metadataset import FTCMetadataset


def load_df(path: str,
            ensembles: list = ['greedy_50_post_calib', 'greedy_5_post_calib'],
            methods: list = ['pure_logits']) -> pd.DataFrame:
    """
    Load a CSV of hyperparameters and filter by ensemble type & method.
    
    Args:
        path: Path to CSV file relative to llm_experiments_data directory
        ensembles: List of ensemble types to include (now includes both greedy_50 and greedy_5)
        methods: List of methods to include
    
    Returns:
        Filtered DataFrame
    """
    base_path = Path('llm_experiments_data')
    full_path = base_path / path
    df = pd.read_csv(full_path)
    mask = df['ensemble_type'].isin(ensembles) & df['method'].isin(methods)
    return df.loc[mask, :].reset_index(drop=True)


def get_predictions(row: pd.Series, test_probs: np.ndarray, test_labels: np.ndarray) -> pd.Series:
    """
    Given a row of hyperparams and test data, run calibration & return mean uncertainties.
    
    Args:
        row: Single row from DataFrame containing hyperparameters
        test_probs: Test probabilities array
        test_labels: Test labels array
    
    Returns:
        Series containing computed uncertainty metrics
    """
    # Extract hyperparameters
    method = row['method']
    c1_prim = row['c1']
    c2_prim = row['c2']
    epi_scalar_prim = row['epi_scalar']

    # Load ensemble indices & select member probabilities
    np_data = np.load(row['path'], allow_pickle=True)
    ensemble_indices = np_data['ensemble_indices']
    member_probs = test_probs[ensemble_indices]

    # Initialize calibrator and predict
    calibrator = PrecomputedCalibrator(
        adjusting_alpha_method=method,
        clamping_alphas=False,
        logits_based_adjustments=True
    )
    results = calibrator.predict(
        precomputed_probs=member_probs,
        c1_prim=c1_prim,
        c2_prim=c2_prim,
        epi_scalar_prim=epi_scalar_prim,
        labels=test_labels
    )

    # Compute and return mean uncertainties
    return pd.Series({
        'MC Aleatoric Mean':  np.mean(results['aleatoric_mc']),
        'MC Epistemic Mean':  np.mean(results['epistemic_mc']),
        'MC Total Mean':      np.mean(results['total_mc']),
        'P Aleatoric Mean':   np.mean(results['aleatoric_p']),
        'P Epistemic Mean':   np.mean(results['epistemic_p']),
        'P Total Mean':       np.mean(results['total_p']),
    })


def main():
    """Main function to calculate uncertainties for ICLR 2025 JUCAL experiments."""
    print("Starting ICLR uncertainty calculation for JUCAL Greedy-50 and Greedy-5", flush=True)
    
    # Number of random seeds / splits
    num_seeds = 5

    # 1) Load dataset once for both metadata versions
    data_dir = '../data'
    metadataset_extended = FTCMetadataset(
        data_dir=str(data_dir), metric_name='error', data_version='extended'
    )
    metadataset_mini = FTCMetadataset(
        data_dir=str(data_dir), metric_name='error', data_version='mini'
    )

    dataset_names = metadataset_extended.get_dataset_names()
    splits = ['valid', 'test']

    # 2) Load ICLR experimental results - both JUCAL methods
    ensembles_of_interest = ['greedy_50_post_calib', 'greedy_5_post_calib']
    
    # Load mini data (single file for ICLR)
    df_mini = load_df('llm_experimental_results_mini_iclr_2025-09-21.csv', 
                      ensembles=ensembles_of_interest)
    df_mini['metadata_version'] = 'mini'

    # Load extended data
    df_extended = load_df('llm_experimental_results_extended_iclr_2025-09-21.csv',
                         ensembles=ensembles_of_interest)
    df_extended['metadata_version'] = 'extended'

    print(f"Loaded mini and extended dataframes with shapes: {df_mini.shape}, {df_extended.shape}", flush=True)
    print(f"Mini ensembles: {sorted(df_mini['ensemble_type'].unique())}", flush=True)
    print(f"Extended ensembles: {sorted(df_extended['ensemble_type'].unique())}", flush=True)

    # Combine dataframes
    all_hparams = pd.concat([df_mini, df_extended], axis=0, ignore_index=True)

    # 3) Iterate over seeds and datasets, compute uncertainties
    print(f'Dataset names: {dataset_names}', flush=True)
    print(f'Dataset names in combined df: {sorted(all_hparams["dataset"].unique())}', flush=True)
    
    results = []
    for seed in range(num_seeds):
        for dataset in dataset_names:
            print(f"Processing seed {seed}, dataset: {dataset}", flush=True)

            # Retrieve valid/test splits for extended and mini
            data_extended = retrieve_data(metadataset_extended, dataset, splits)
            data_mini = retrieve_data(metadataset_mini, dataset, splits)

            # Unpack or create new splits
            if seed == 0:
                test_memb_probs_extended = data_extended['test'][2]
                test_labels_extended     = data_extended['test'][3]

                test_memb_probs_mini = data_mini['test'][2]
                test_labels_mini     = data_mini['test'][3]
            else:
                _, _, test_memb_probs_extended, test_labels_extended = create_new_split(
                    data_extended['valid'][2], data_extended['valid'][3],
                    data_extended['test'][2],  data_extended['test'][3],
                    seed=seed
                )
                _, _, test_memb_probs_mini, test_labels_mini = create_new_split(
                    data_mini['valid'][2], data_mini['valid'][3],
                    data_mini['test'][2],  data_mini['test'][3],
                    seed=seed
                )

            # Filter hyperparameters for this seed and dataset
            df_seed = all_hparams[
                (all_hparams['seed'] == seed) &
                (all_hparams['dataset'] == dataset)
            ]

            # Compute predictions & uncertainties per hyperparam setting
            for _, row in df_seed.iterrows():
                # Select appropriate test set
                if row['metadata_version'] == 'extended':
                    test_probs, test_labels = test_memb_probs_extended, test_labels_extended
                else:
                    test_probs, test_labels = test_memb_probs_mini, test_labels_mini

                # Run prediction / calibration and compute means
                try:
                    preds = get_predictions(row, test_probs, test_labels)
                    record = {**row.to_dict(), **preds.to_dict()}
                    results.append(record)
                    print(f"  Processed {row['ensemble_type']} for {dataset} seed {seed} ({row['metadata_version']})", flush=True)
                except Exception as e:
                    print(f"  ERROR processing {row['ensemble_type']} for {dataset} seed {seed}: {e}", flush=True)
                    continue

    # 4) Save combined results
    final_df = pd.DataFrame(results)
    out_path = Path('llm_experiments_data/calibrated_uncertainties_iclr.csv')
    final_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}", flush=True)
    print(f"Final results shape: {final_df.shape}", flush=True)
    print(f"Unique ensemble types in results: {sorted(final_df['ensemble_type'].unique())}", flush=True)
    print(f"Unique metadata versions: {sorted(final_df['metadata_version'].unique())}", flush=True)


if __name__ == '__main__':
    main()