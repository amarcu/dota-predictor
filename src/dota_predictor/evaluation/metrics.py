"""
Evaluation metrics for win probability models.

The primary metric for probabilistic predictions is the Brier Score,
which measures calibration and discrimination of probability estimates.
"""

from __future__ import annotations

import numpy as np
from typing import Any
import torch
from torch.utils.data import DataLoader


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate the Brier Score (mean squared error of probabilities).
    
    The Brier Score measures the accuracy of probabilistic predictions.
    It ranges from 0 (perfect) to 1 (worst).
    
    A baseline model predicting 50% always would score 0.25.
    
    Args:
        y_true: True binary outcomes (0 or 1)
        y_prob: Predicted probabilities for class 1
        
    Returns:
        Brier score (lower is better)
    """
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate the Log Loss (binary cross-entropy).
    
    Log loss heavily penalizes confident wrong predictions.
    
    Args:
        y_true: True binary outcomes (0 or 1)
        y_prob: Predicted probabilities for class 1
        eps: Small value to avoid log(0)
        
    Returns:
        Log loss (lower is better)
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate classification accuracy at a given threshold.
    
    Args:
        y_true: True binary outcomes (0 or 1)
        y_prob: Predicted probabilities for class 1
        threshold: Decision threshold
        
    Returns:
        Accuracy (higher is better)
    """
    y_pred = (y_prob >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


def calibration_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve (reliability diagram data).
    
    For a well-calibrated model, predictions of X% should correspond
    to an actual win rate of ~X%.
    
    Args:
        y_true: True binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of bins to use
        
    Returns:
        Tuple of (mean_predicted_probs, actual_fractions, bin_counts)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    mean_predicted = np.zeros(n_bins)
    actual_fraction = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mean_predicted[i] = np.mean(y_prob[mask])
            actual_fraction[i] = np.mean(y_true[mask])
            bin_counts[i] = np.sum(mask)
    
    return mean_predicted, actual_fraction, bin_counts


def expected_calibration_error(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the weighted average deviation from perfect calibration.
    
    Args:
        y_true: True binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        ECE (lower is better, 0 = perfectly calibrated)
    """
    mean_pred, actual_frac, counts = calibration_curve(y_true, y_prob, n_bins)
    total = np.sum(counts)
    
    if total == 0:
        return 0.0
    
    ece = np.sum(counts * np.abs(mean_pred - actual_frac)) / total
    return float(ece)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Evaluate a model on a dataset and return all metrics.
    
    Args:
        model: The trained model
        dataloader: DataLoader with test data
        device: Device to run inference on
        
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    model.to(device)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch.get("label", batch.get("labels")).to(device)
            durations = batch.get("duration", batch.get("durations"))
            heroes = batch.get("heroes", batch.get("hero_ids"))
            
            # Get predictions (sequence mode matches training)
            if hasattr(model, "use_hero_embedding") and model.use_hero_embedding and heroes is not None:
                outputs = model(features, heroes.to(device), return_sequence=True)
            else:
                outputs = model(features, return_sequence=True)
            
            # Handle sequence output - get prediction at each sample's duration
            if outputs.dim() == 2:
                # outputs shape: (batch, seq_len)
                if durations is not None:
                    batch_size = outputs.size(0)
                    probs = torch.zeros(batch_size, device=device)
                    for i in range(batch_size):
                        t = min(durations[i].item() - 1, outputs.size(1) - 1)
                        probs[i] = outputs[i, int(t)]
                else:
                    probs = outputs[:, -1]
            else:
                probs = outputs.squeeze()
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    
    # Calculate all metrics
    metrics = {
        "brier_score": brier_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "accuracy": accuracy(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "n_samples": len(y_true),
        "baseline_brier": 0.25,  # Always predicting 50%
    }
    
    # Add improvement over baseline
    metrics["brier_improvement"] = (0.25 - metrics["brier_score"]) / 0.25 * 100
    
    return metrics


