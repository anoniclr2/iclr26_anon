#!/usr/bin/env python3
"""
Standalone script to compute LLM analysis metrics.
This script replicates the metric computation from the Jupyter notebook 
for running on cluster with SLURM.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import argparse

# Add src to path for importing helper functions
sys.path.append('src/')
from plotting import get_coverage_threshold_and_size, get_auc, compute_aurac, compute_ece, compute_ece_adaptive, compute_brier_score, compute_conformal_aps


def read_file(file_path: str, base_path: Path) -> pd.DataFrame:
    """Read a file and return a DataFrame."""
    path = base_path / 'llm_experiments_data' / file_path
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    return pd.read_csv(path)


def compute_metrics_rowise(row: pd.Series, base_path: Path, target_coverage: float = 0.99) -> pd.Series:
    """Compute metrics for a single row (experiment)."""
    npfile = base_path / row['path']
    data = np.load(npfile, allow_pickle=True)
    probs, labels = data['ensemble_probs'], data['labels']

    misclassification = (probs.argmax(axis=1) != labels).mean()  # Misclassification rate
    threshold, set_size = get_coverage_threshold_and_size(probs, labels, target_coverage=target_coverage)
    auc = get_auc(probs, labels)
    aurac = compute_aurac(labels, probs)  # Using the new AURAC function
    aorac = 1 - aurac  # Area Over Rejection-Accuracy Curve
    ece = compute_ece(probs, labels, n_bins=10)  # Expected Calibration Error with 10 bins
    ece_adaptive = compute_ece_adaptive(probs, labels)  # Expected Calibration Error with adaptive binning
    brier = compute_brier_score(probs, labels)  # Brier score
    conformal_coverage, conformal_set_size = compute_conformal_aps(probs, labels, target_coverage=target_coverage)  # Conformal prediction

    return pd.Series({
        'misclassification': misclassification,
        'threshold': threshold,
        'set_size': set_size,
        'auc': auc,
        'aurac': aurac,
        'aorac': aorac,
        'ece': ece,
        'ece_adaptive': ece_adaptive,
        'brier_score': brier,
        'conformal_coverage': conformal_coverage,
        'conformal_set_size': conformal_set_size
    })


def compute_metrics(df: pd.DataFrame, base_path: Path, target_coverage: float = 0.999, 
                   output_path: Path = None) -> pd.DataFrame:
    """Compute metrics and save dataframe."""
    print(f"Computing metrics for {len(df)} experiments with target coverage {target_coverage}")
    
    df = df.copy()
    total_rows = len(df)
    
    # Compute metrics with simple progress tracking
    metrics_list = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 10 == 0 or i == total_rows - 1:
            print(f"Progress: {i+1}/{total_rows} ({(i+1)/total_rows*100:.1f}%)")
        
        metrics = compute_metrics_rowise(row, base_path=base_path, target_coverage=target_coverage)
        metrics_list.append(metrics)
    
    # Convert list of Series to DataFrame and assign to original DataFrame
    metrics_df = pd.DataFrame(metrics_list, index=df.index)
    df[['misclassification', 'threshold', 'set_size', 'auc', 'aurac', 'aorac', 'ece', 'ece_adaptive', 'brier_score',
        'conformal_coverage', 'conformal_set_size']] = metrics_df
    
    # Save the DataFrame to a CSV file
    if output_path:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compute LLM analysis metrics')
    parser.add_argument('--base-path', type=str, default='notebooks/Ensembling_Finetuned_LLMs',
                       help='Base path to data directory')
    parser.add_argument('--input-date', type=str, default='2025-09-23',
                       help='Date string for input files')
    parser.add_argument('--output-date', type=str, default=None,
                       help='Date string for output files (defaults to input-date if not specified)')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    input_date = args.input_date
    output_date = args.output_date if args.output_date else input_date

    print("=== LLM Metrics Computation ===")
    print(f"Base path: {base_path}")
    print(f"Input date: {input_date}")
    print(f"Output date: {output_date}")
    
    # Load the experimental results
    print("\nLoading experimental data...")
    try:
        df_extended = read_file(f'llm_experimental_results_extended_iclr_{input_date}.csv', base_path)
        df_mini = read_file(f'llm_experimental_results_mini_iclr_{input_date}.csv', base_path)
        print(f"Extended dataset shape: {df_extended.shape}")
        print(f"Mini dataset shape: {df_mini.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create metrics output directory
    metrics_dir = base_path / 'llm_experiments_data' / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics for 99% coverage
    print("\n=== Computing metrics for 99% coverage ===")
    compute_metrics(
        df_extended,
        base_path=base_path,
        target_coverage=0.99,
        output_path=metrics_dir / f'extended_with_metrics_cov_0p99_{output_date}.csv'
    )

    compute_metrics(
        df_mini,
        base_path=base_path,
        target_coverage=0.99,
        output_path=metrics_dir / f'mini_with_metrics_cov_0p99_{output_date}.csv'
    )

    # # Compute metrics for 99.9% coverage
    # print("\n=== Computing metrics for 99.9% coverage ===")
    # compute_metrics(
    #     df_extended,
    #     base_path=base_path,
    #     target_coverage=0.999,
    #     output_path=metrics_dir / f'extended_with_metrics_cov_0p999_{output_date}.csv'
    # )

    # compute_metrics(
    #     df_mini,
    #     base_path=base_path,
    #     target_coverage=0.999,
    #     output_path=metrics_dir / f'mini_with_metrics_cov_0p999_{output_date}.csv'
    # )
    
    print("\n=== Computation completed successfully! ===")
    print(f"All metrics saved to {metrics_dir}")


if __name__ == "__main__":
    main()