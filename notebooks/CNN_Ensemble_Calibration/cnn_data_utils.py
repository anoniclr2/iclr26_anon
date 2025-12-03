"""
CNN Data Loading Utilities

Functions for loading pre-computed CNN ensemble predictions from JSON files
and converting them to the tensor format expected by calibration code.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def load_cnn_predictions(json_path: str) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load CNN ensemble predictions from JSON and convert to tensor format.

    Args:
        json_path: Path to the JSON file containing predictions

    Returns:
        Dictionary mapping split_idx to (val_probs, val_labels, test_probs, test_labels)
        - val_probs: [num_members, num_samples, num_classes]
        - val_labels: [num_samples]
        - test_probs: [num_members, num_samples, num_classes]
        - test_labels: [num_samples]
    """
    print(f"Loading predictions from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = {}

    for split_data in data:
        split_idx = split_data['split_idx']
        members = split_data['ensemble_members']

        # Stack member predictions: [num_members, num_samples, num_classes]
        val_probs = torch.tensor(
            np.stack([np.array(m['val_predictions']) for m in members]),
            dtype=torch.float32
        )
        test_probs = torch.tensor(
            np.stack([np.array(m['test_predictions']) for m in members]),
            dtype=torch.float32
        )

        # Labels are identical across members, so just take from first member
        val_labels = torch.tensor(
            np.array(members[0]['val_true_labels']),
            dtype=torch.long
        )
        test_labels = torch.tensor(
            np.array(members[0]['test_true_labels']),
            dtype=torch.long
        )

        results[split_idx] = (val_probs, val_labels, test_probs, test_labels)

    print(f"Loaded {len(results)} splits")
    print(f"Val shape: {val_probs.shape}, Test shape: {test_probs.shape}")

    return results


def get_num_splits(json_path: str) -> int:
    """
    Get the number of splits in a JSON file without loading all data.

    Args:
        json_path: Path to the JSON file containing predictions

    Returns:
        Number of splits in the file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data)


def load_cifar100_split(split_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a single CIFAR100 split from individual split files.

    For CIFAR100, we use pre-split JSON files to minimize memory usage.
    Falls back to loading from the full JSON if split files don't exist.

    Args:
        split_idx: Index of the split to load

    Returns:
        Tuple of (val_probs, val_labels, test_probs, test_labels)
        - val_probs: [num_members, num_samples, num_classes]
        - val_labels: [num_samples]
        - test_probs: [num_members, num_samples, num_classes]
        - test_labels: [num_samples]
    """
    base_dir = Path(__file__).parent / "cnn_prediction_output_data"
    split_file = base_dir / "cifar100_splits" / f"CIFAR100_split_{split_idx}.json"

    # Try to load from individual split file first
    if split_file.exists():
        with open(split_file, 'r') as f:
            split_data = json.load(f)
    else:
        # Fallback to loading from full JSON
        full_file = base_dir / "CIFAR100ensemble_predictions.json"
        with open(full_file, 'r') as f:
            data = json.load(f)

        # Find the requested split
        split_data = None
        for split in data:
            if split['split_idx'] == split_idx:
                split_data = split
                break

        if split_data is None:
            raise ValueError(f"Split {split_idx} not found in CIFAR100 data")

    members = split_data['ensemble_members']

    # Stack member predictions: [num_members, num_samples, num_classes]
    val_probs = torch.tensor(
        np.stack([np.array(m['val_predictions']) for m in members]),
        dtype=torch.float32
    )
    test_probs = torch.tensor(
        np.stack([np.array(m['test_predictions']) for m in members]),
        dtype=torch.float32
    )

    # Labels are identical across members, so just take from first member
    val_labels = torch.tensor(
        np.array(members[0]['val_true_labels']),
        dtype=torch.long
    )
    test_labels = torch.tensor(
        np.array(members[0]['test_true_labels']),
        dtype=torch.long
    )

    return val_probs, val_labels, test_probs, test_labels


def load_cnn_predictions_single_split(
    json_path: str,
    split_idx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a single split from CNN ensemble predictions JSON.

    This function loads the full JSON but immediately processes only the
    requested split and discards the rest, reducing peak memory usage.

    Args:
        json_path: Path to the JSON file containing predictions
        split_idx: Index of the split to load

    Returns:
        Tuple of (val_probs, val_labels, test_probs, test_labels)
        - val_probs: [num_members, num_samples, num_classes]
        - val_labels: [num_samples]
        - test_probs: [num_members, num_samples, num_classes]
        - test_labels: [num_samples]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Find the split with matching index and process immediately
    for split in data:
        if split['split_idx'] == split_idx:
            members = split['ensemble_members']

            # Stack member predictions: [num_members, num_samples, num_classes]
            val_probs = torch.tensor(
                np.stack([np.array(m['val_predictions']) for m in members]),
                dtype=torch.float32
            )
            test_probs = torch.tensor(
                np.stack([np.array(m['test_predictions']) for m in members]),
                dtype=torch.float32
            )

            # Labels are identical across members, so just take from first member
            val_labels = torch.tensor(
                np.array(members[0]['val_true_labels']),
                dtype=torch.long
            )
            test_labels = torch.tensor(
                np.array(members[0]['test_true_labels']),
                dtype=torch.long
            )

            # Return immediately to allow garbage collection of the rest
            return val_probs, val_labels, test_probs, test_labels

    raise ValueError(f"Split {split_idx} not found in {json_path}")


def get_dataset_paths() -> Dict[str, Path]:
    """
    Get paths to all CNN ensemble prediction JSON files.

    Returns:
        Dictionary mapping dataset name to JSON file path
    """
    base_dir = Path(__file__).parent / "cnn_prediction_output_data"

    return {
        "MNIST": base_dir / "MNISTensemble_predictions.json",
        "FMNIST": base_dir / "FMNISTensemble_predictions.json",
        "CIFAR": base_dir / "CIFARensemble_predictions.json",
        "CIFAR100": base_dir / "CIFAR100ensemble_predictions.json",
        "MNIST10Epochs10_5L2Reg": base_dir / "MNIST10Epochs10_5L2Regensemble_predictions.json",
        "MNIST40Epochs10_4L2Reg": base_dir / "MNIST40Epochs10_4L2Regensemble_predictions.json"
    }
