import torch
import numpy as np
import torch.nn.functional as F
from typing import Sequence

def find_optimal_temp(logits: np.ndarray, labels: np.ndarray, temperatures: Sequence[float]) -> float:
    """
    Finds the optimal temperature that minimizes negative log-likelihood (NLL)
    using PyTorch softmax for stability and convenience.

    Args:
        logits (np.ndarray): Logits of shape (batch_size, num_classes).
        labels (np.ndarray): Ground-truth labels of shape (batch_size,).
        temperatures (Sequence[float]): Temperatures to evaluate.

    Returns:
        float: Best temperature.
    """
    logits_torch = torch.tensor(logits, dtype=torch.float32)
    labels_torch = torch.tensor(labels, dtype=torch.long)

    nlls = []
    for temp in temperatures:
        scaled_logits = logits_torch / temp
        probs = F.softmax(scaled_logits, dim=1)
        log_probs = torch.log(probs + 1e-12)
        nll = -log_probs.gather(1, labels_torch.unsqueeze(1)).squeeze(1)
        nlls.append(nll.mean().item())

    return float(temperatures[np.argmin(nlls)])


def temp_scale_logits_nll(logits: np.ndarray, labels: np.ndarray, temp: float) -> float:
    """
    Scales logits using the optimal temperature and returns calibrated nll

    Args:
        logits (np.ndarray): Raw model outputs.
        labels (np.ndarray): Ground-truth labels.
        temp (float) : Temperature.

    Returns:
        float nll.
    """

    logits_torch = torch.tensor(logits, dtype=torch.float32) / temp
    probs = F.softmax(logits_torch, dim=1)
    log_probs = torch.log(probs + 1e-12)
    labels_torch = torch.tensor(labels, dtype=torch.long)
    nll = -log_probs.gather(1, labels_torch.unsqueeze(1)).squeeze(1)
    return nll.mean().item()