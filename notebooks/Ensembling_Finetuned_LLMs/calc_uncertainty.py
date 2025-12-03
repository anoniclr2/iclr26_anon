"""Check mini vs extended"""

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
            ensembles: list = ['greedy_50_post_calib'],
            methods: list = ['pure_logits']) -> pd.DataFrame:
    """
    Load a CSV of hyperparameters and filter by ensemble type & method.
    """
    #['greedy_unique_5_baseline', 'greedy_50_baseline', 'greedy_50_temp_baseline', 'greedy_unique_5_temp_baseline', 
    # 'greedy_unique_5_post_calib', 'greedy_50_post_calib', 'greedy_50_calib_once', 'greedy_50_calib_every_step']

    # ['pure_logits', 'convex_comb']

    base_path = Path('llm_experiments_data')
    full_path = base_path / path
    df = pd.read_csv(full_path)
    mask = df['ensemble_type'].isin(ensembles) & df['method'].isin(methods)
    return df.loc[mask, :].reset_index(drop=True)


def concatenate_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple DataFrames and drop duplicate rows.
    """
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.loc[~df.duplicated(keep='first')].reset_index(drop=True)
    return df


def get_predictions(row: pd.Series, test_probs: np.ndarray, test_labels: np.ndarray) -> pd.Series:
    """
    Given a row of hyperparams and test data, run calibration & return mean uncertainties.
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
    # Number of random seeds / splits
    print("Starting main", flush=True)
    num_seeds = 5

    # 1) Load dataset once for both metadata versions
    data_dir = '../data'
    metadataset_full = FTCMetadataset(
        data_dir=str(data_dir), metric_name='error', data_version='extended'
    )
    metadataset_mini = FTCMetadataset(
        data_dir=str(data_dir), metric_name='error', data_version='mini'
    )

    dataset_names = metadataset_full.get_dataset_names()
    splits = ['valid', 'test']

    # 2) Load and combine hyperparameter CSVs
    df_mini_1 = load_df('llm_experimental_results_mini_neurips_2025-04-20.csv')
    df_mini_2 = load_df('llm_experimental_results_mini_neurips_2025-04-24.csv')
    df_mini = concatenate_dfs([df_mini_1, df_mini_2])
    df_mini['metadata_version'] = 'mini'

    df_ftc = load_df('llm_experimental_results_ftc_neurips_2025-04-25.csv')
    df_ftc['metadata_version'] = 'extended'

    print(f"Loaded mini and full dataframes with the following columns {df_mini.columns} and {df_ftc.columns}", flush=True)

    all_hparams = pd.concat([df_mini, df_ftc], axis=0, ignore_index=True)

    # 3) Iterate over seeds and datasets, compute uncertainties
    print(f'Dataset names: {dataset_names}', flush=True)
    print(f'Dataset names in df_mini: {df_mini["dataset"].unique()}', flush=True)
    print(f'Dataset names in df_ftc: {df_ftc["dataset"].unique()}', flush=True)
    results = []
    for seed in range(num_seeds):
        for dataset in dataset_names:
            print(f"Processing seed {seed}, dataset: {dataset}", flush =True)

            # Retrieve valid/test splits for full and mini
            data_full = retrieve_data(metadataset_full, dataset, splits)
            data_mini = retrieve_data(metadataset_mini, dataset, splits)

            # Unpack or create new splits
            if seed == 0:
                #val_memb_probs_full = data_full['valid'][2]
                #val_labels_full      = data_full['valid'][3]
                test_memb_probs_full = data_full['test'][2]
                test_labels_full     = data_full['test'][3]

                #val_memb_probs_mini  = data_mini['valid'][2]
                #val_labels_mini      = data_mini['valid'][3]
                test_memb_probs_mini = data_mini['test'][2]
                test_labels_mini     = data_mini['test'][3]
            else:
                _,_, test_memb_probs_full, test_labels_full = create_new_split(
                    data_full['valid'][2], data_full['valid'][3],
                    data_full['test'][2],  data_full['test'][3],
                    seed=seed
                )
                _,_, test_memb_probs_mini, test_labels_mini = create_new_split(
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
                    test_probs, test_labels = test_memb_probs_full, test_labels_full
                else:
                    test_probs, test_labels = test_memb_probs_mini, test_labels_mini

                # Run prediction / calibration and compute means
                preds = get_predictions(row, test_probs, test_labels)
                record = {**row.to_dict(), **preds.to_dict()}
                results.append(record)

    # 4) Save combined results
    final_df = pd.DataFrame(results)
    out_path = Path('llm_experiments_data/calibrated_uncertainties.csv')
    final_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}", flush=True)


if __name__ == '__main__':
    main()