"""
Split CIFAR100 ensemble predictions JSON into individual split files.

Run this script locally to create separate JSON files for each split,
which can then be uploaded to the cluster to reduce memory usage.
"""

import json
from pathlib import Path


def split_cifar100_json() -> None:
    """
    Split the large CIFAR100 JSON into individual files, one per split.
    """
    input_file = Path(__file__).parent / "cnn_prediction_output_data" / "CIFAR100ensemble_predictions.json"
    output_dir = Path(__file__).parent / "cnn_prediction_output_data" / "cifar100_splits"

    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data)} splits")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Save each split to a separate file
    for split_data in data:
        split_idx = split_data['split_idx']
        output_file = output_dir / f"CIFAR100_split_{split_idx}.json"

        print(f"Writing split {split_idx} to {output_file.name}...")
        with open(output_file, 'w') as f:
            json.dump(split_data, f)

    print(f"\nDone! Created {len(data)} split files in {output_dir}")


if __name__ == "__main__":
    split_cifar100_json()
