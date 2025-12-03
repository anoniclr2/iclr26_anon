#!/usr/bin/env python3
"""
Standalone script to compute CNN analysis metrics.

This script computes calibration and uncertainty metrics for CNN ensemble experiments,
following the same structure as the LLM metrics computation.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import argparse

# Add src to path for importing helper functions
sys.path.append('../../src/')
from plotting import (
    get_coverage_threshold_and_size,
    get_auc,
    compute_aurac,
    compute_ece,
    compute_ece_adaptive,
    compute_brier_score,
    compute_conformal_aps
)


def compute_misclassification(probs: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Compute misclassification rate.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        true_labels: True labels, shape (n_samples,)

    Returns:
        Misclassification rate as a float between 0 and 1
    """
    predictions = np.argmax(probs, axis=1)
    return np.mean(predictions != true_labels)


def read_file(file_path: str, base_path: Path) -> pd.DataFrame:
    """
    Read a file and return a DataFrame.

    Args:
        file_path: Relative path to the file
        base_path: Base directory path

    Returns:
        DataFrame with experimental results
    """
    path = base_path / 'cnn_experiments_data' / file_path
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    return pd.read_csv(path)


def compute_metrics_rowise(row: pd.Series, base_path: Path, target_coverage: float = 0.999) -> pd.Series:
    """
    Compute metrics for a single row (experiment).

    Args:
        row: DataFrame row containing experiment results
        base_path: Base directory path
        target_coverage: Target coverage for threshold/set size computation

    Returns:
        Series with computed metrics
    """
    npfile = base_path / row['path']
    data = np.load(npfile, allow_pickle=True)
    probs, labels = data['ensemble_probs'], data['labels']

    threshold, set_size = get_coverage_threshold_and_size(probs, labels, target_coverage=target_coverage)
    auc = get_auc(probs, labels)
    aurac = compute_aurac(labels, probs)  # Area Under Rejection-Accuracy Curve
    aorac = 1 - aurac  # Area Over Rejection-Accuracy Curve
    ece = compute_ece(probs, labels, n_bins=10)  # Expected Calibration Error with 10 bins
    ece_adaptive = compute_ece_adaptive(probs, labels)  # ECE with adaptive binning
    brier = compute_brier_score(probs, labels)  # Brier score
    conformal_coverage, conformal_set_size = compute_conformal_aps(probs, labels, target_coverage=target_coverage)  # Conformal prediction
    misclassification = compute_misclassification(probs, labels)  # Misclassification rate

    return pd.Series({
        'threshold': threshold,
        'set_size': set_size,
        'auc': auc,
        'aurac': aurac,
        'aorac': aorac,
        'ece': ece,
        'ece_adaptive': ece_adaptive,
        'brier_score': brier,
        'conformal_coverage': conformal_coverage,
        'conformal_set_size': conformal_set_size,
        'misclassification': misclassification
    })


def compute_metrics(df: pd.DataFrame, base_path: Path, target_coverage: float = 0.999,
                   output_path: Path = None) -> pd.DataFrame:
    """
    Compute metrics and save dataframe.

    Args:
        df: DataFrame with experimental results
        base_path: Base directory path
        target_coverage: Target coverage for threshold/set size computation
        output_path: Path to save output CSV

    Returns:
        DataFrame with added metrics
    """
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
    df[['threshold', 'set_size', 'auc', 'aurac', 'aorac', 'ece', 'ece_adaptive', 'brier_score',
        'conformal_coverage', 'conformal_set_size', 'misclassification']] = metrics_df

    # Save the DataFrame to a CSV file
    if output_path:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

    return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Compute CNN analysis metrics')
    parser.add_argument('--base-path', type=str, default='.',
                       help='Base path to CNN experiment directory')
    parser.add_argument('--date', type=str, default='2025-11-30',
                       help='Date string for input/output files')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    date = args.date

    print("=== CNN Metrics Computation ===")
    print(f"Base path: {base_path}")
    print(f"Date: {date}")

    # Load the experimental results
    print("\nLoading experimental data...")
    input_file = f'cnn_experimental_results_iclr_{date}.csv'
    try:
        df_cnn = read_file(input_file, base_path)
        print(f"CNN dataset shape: {df_cnn.shape}")
        print(f"Available datasets: {sorted(df_cnn['dataset'].unique())}")
        print(f"Available ensemble types: {sorted(df_cnn['ensemble_type'].unique())}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Create metrics output directory
    metrics_dir = base_path / 'cnn_experiments_data' / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # # Compute metrics for 99% coverage
    # print("\n=== Computing metrics for 99% coverage ===")
    # compute_metrics(
    #     df_cnn,
    #     base_path=base_path,
    #     target_coverage=0.99,
    #     output_path=metrics_dir / f'cnn_with_metrics_cov_0p99_{date}.csv'
    # )

    # # Compute metrics for 99.9% coverage
    # print("\n=== Computing metrics for 99.9% coverage ===")
    # compute_metrics(
    #     df_cnn,
    #     base_path=base_path,
    #     target_coverage=0.999,
    #     output_path=metrics_dir / f'cnn_with_metrics_cov_0p999_{date}.csv'
    # )

    # Compute metrics for 90% coverage
    print("\n=== Computing metrics for 90% coverage ===")
    compute_metrics(
        df_cnn,
        base_path=base_path,
        target_coverage=0.90,
        output_path=metrics_dir / f'cnn_with_metrics_cov_0p90_{date}.csv'
    )

    print("\n=== Metrics computation completed successfully! ===")
    print(f"Results saved to {metrics_dir}")


if __name__ == "__main__":
    main()
