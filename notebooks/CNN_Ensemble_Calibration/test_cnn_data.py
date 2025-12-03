"""
Test script to explore CNN ensemble prediction data structure.

Loads MNIST, FMNIST, and CIFAR prediction JSONs and prints their shapes and structure.
"""

import json
import numpy as np
from pathlib import Path


def explore_dataset(json_path: Path, dataset_name: str) -> None:
    """
    Load and explore a single dataset's prediction JSON.

    Args:
        json_path: Path to the JSON file
        dataset_name: Name of the dataset (for printing)
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"File: {json_path.name}")
    print(f"{'='*80}")

    # Load JSON
    print(f"\nLoading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Overall structure
    print(f"\nNumber of splits: {len(data)}")

    # Examine first split
    first_split = data[0]
    print(f"\nFirst split keys: {list(first_split.keys())}")
    print(f"Split index: {first_split['split_idx']}")

    # Ensemble members
    ensemble_members = first_split['ensemble_members']
    print(f"Number of ensemble members: {len(ensemble_members)}")

    # Examine first ensemble member
    first_member = ensemble_members[0]
    print(f"\nFirst ensemble member keys: {list(first_member.keys())}")

    val_preds = np.array(first_member['val_predictions'])
    print(f"\nVal predictions shape: {val_preds.shape}")
    print(f"  Sum: {val_preds[0].sum():.6f} (should be ~1.0 for probabilities)")

    test_preds = np.array(first_member['test_predictions'])
    print(f"\nTest predictions shape: {test_preds.shape}")

    # Val labels
    val_labels = np.array(first_member['val_true_labels'])
    print(f"\nVal labels shape: {val_labels.shape}")
    print(f"  First 5 labels: {val_labels[:5]}")

    test_labels = np.array(first_member['test_true_labels'])
    print(f"\nTest labels shape: {test_labels.shape}")

def main() -> None:
    """Main function to explore all CNN datasets."""
    data_dir = Path(__file__).parent / "cnn_prediction_output_data"

    datasets = {
        "MNIST": data_dir / "MNISTensemble_predictions.json",
        "Fashion-MNIST": data_dir / "FMNISTensemble_predictions.json",
        "CIFAR-10": data_dir / "CIFARensemble_predictions.json",
        "CIFAR-100": data_dir / "CIFAR100ensemble_predictions.json",
        "MNIST-10Epochs-10_5L2Reg": data_dir / "MNIST10Epochs10_5L2Regensemble_predictions.json",
        "MNIST-40Epochs-10_4L2Reg": data_dir / "MNIST40Epochs10_4L2Regensemble_predictions.json",
    }

    for name, path in datasets.items():
        if path.exists():
            explore_dataset(path, name)
        else:
            print(f"\n{name} file not found: {path}")

    print(f"\n{'='*80}")
    print("Exploration complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
