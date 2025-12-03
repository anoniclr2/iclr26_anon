"""Simple test of conformal prediction function."""
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append('src/')
from plotting import compute_conformal_aps

if __name__ == "__main__":
    # Larger toy example: 500 samples, 4 classes (realistic LLM test set size)
    # Simulate a confident, mostly correct model
    np.random.seed(42)
    n_samples = 500
    n_classes = 4

    probs = []
    labels = []

    for i in range(n_samples):
        # Create a probability distribution
        p = np.random.dirichlet([0.5] * n_classes)  # Random baseline

        # Sample a label from this distribution
        label = np.random.choice(n_classes, p=p)
        #print(f'label: {label}')
        probs.append(p)
        labels.append(label)

    probs = np.array(probs)
    labels = np.array(labels)

    #print("="*60)
    #print("TOY DATA")
    #print("="*60)
    #print(f"Probs shape: {probs.shape}")
    #print(f"Labels: {labels}")
    #print("\nFirst 3 samples:")
    #for i in range(10):
        #print(f"  Sample {i}: probs={probs[i]}, label={labels[i]}")

    # Run conformal prediction
    print("\n" + "="*60)
    print("RUNNING CONFORMAL PREDICTION")
    print("="*60)

    target_cov = 0.95
    coverage, avg_set_size = compute_conformal_aps(
        probs,
        labels,
        target_coverage=target_cov,
        cal_ratio=0.3
    )

    print("\nResults:")
    print(f"Target coverage: {target_cov}")
    print(f"Actual coverage: {coverage:.4f}")
    print(f"Avg set size: {avg_set_size:.2f}")
    print(f"Coverage error: {coverage - target_cov:+.4f}")
