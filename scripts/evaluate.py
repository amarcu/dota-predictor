#!/usr/bin/env python3
"""
Evaluate a trained model and log results to the experiment tracker.

Usage:
    # Evaluate a model
    python scripts/evaluate.py --model models/checkpoints/model.pt
    
    # Evaluate and log to tracker
    python scripts/evaluate.py --model models/checkpoints/model.pt --log --name "baseline_v1"
    
    # Compare recent experiments
    python scripts/evaluate.py --compare
    
    # Show experiment summary
    python scripts/evaluate.py --summary
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from dota_predictor.models.loader import load_model_from_checkpoint


class ProcessedDataset(Dataset):
    """Dataset for pre-processed numpy arrays."""

    def __init__(self, features, heroes, labels, masks, durations):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.heroes = torch.tensor(heroes, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.durations = torch.tensor(durations, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "hero_ids": self.heroes[idx],
            "labels": self.labels[idx],
            "mask": self.masks[idx],
            "durations": self.durations[idx],
        }
from dota_predictor.evaluation.metrics import (
    brier_score,
    log_loss,
    accuracy,
    expected_calibration_error,
    calibration_curve,
)
from dota_predictor.evaluation.tracker import ExperimentTracker


def load_model(model_path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Load a model and its config from checkpoint."""
    return load_model_from_checkpoint(model_path, device=device)


def load_processed_data(data_dir: Path, split: str = "val") -> ProcessedDataset:
    """Load pre-processed numpy arrays for evaluation."""
    features = np.load(data_dir / f"{split}_features.npy")
    heroes = np.load(data_dir / f"{split}_heroes.npy")
    labels = np.load(data_dir / f"{split}_labels.npy")
    masks = np.load(data_dir / f"{split}_masks.npy")
    durations = np.load(data_dir / f"{split}_durations.npy")
    
    # Features are stored raw (unnormalized) — training does not normalize,
    # so evaluation must also use raw features to match training conditions.
    
    return ProcessedDataset(features, heroes, labels, masks, durations)


def evaluate_on_dataset(
    model: torch.nn.Module,
    data_dir: str,
    device: str = "cpu",
    batch_size: int = 64,
    split: str = "val",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Returns:
        Tuple of (metrics_dict, y_true, y_prob)
    """
    dataset = load_processed_data(Path(data_dir), split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            durations = batch["durations"].to(device)
            hero_ids = batch["hero_ids"].to(device)
            
            # Get predictions - match training exactly
            # Training uses: self.model(features, heroes, return_sequence=True)
            if hasattr(model, "use_hero_embedding") and model.use_hero_embedding:
                outputs = model(features, hero_ids, return_sequence=True)
            else:
                outputs = model(features, return_sequence=True)
            
            # outputs shape: (batch, seq_len)
            # Get prediction at last valid timestep (matching training)
            batch_size_local = outputs.size(0)
            probs = torch.stack([
                outputs[i, min(durations[i].item() - 1, 59)] 
                for i in range(batch_size_local)
            ])
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        "brier_score": brier_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "accuracy": accuracy(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "n_samples": len(y_true),
        "mean_prediction": float(np.mean(y_prob)),
        "std_prediction": float(np.std(y_prob)),
    }
    
    # Baseline comparison
    metrics["baseline_brier"] = 0.25  # Always predicting 50%
    metrics["brier_improvement_pct"] = (0.25 - metrics["brier_score"]) / 0.25 * 100
    
    return metrics, y_true, y_prob


def print_metrics(metrics: dict) -> None:
    """Print metrics in a nice format."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {metrics['n_samples']:,}")
    print()
    print("  PRIMARY METRIC (lower is better):")
    print(f"    Brier Score:     {metrics['brier_score']:.4f}")
    print(f"    (Baseline 50%):  {metrics['baseline_brier']:.4f}")
    print(f"    Improvement:     {metrics['brier_improvement_pct']:+.1f}%")
    print()
    print("  SECONDARY METRICS:")
    print(f"    Log Loss:        {metrics['log_loss']:.4f}")
    print(f"    Accuracy:        {metrics['accuracy']:.2%}")
    print(f"    ECE (calibration): {metrics['ece']:.4f}")
    print()
    print("  PREDICTION STATS:")
    print(f"    Mean prediction: {metrics['mean_prediction']:.3f}")
    print(f"    Std prediction:  {metrics['std_prediction']:.3f}")
    print("=" * 60)


def print_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Print calibration analysis."""
    mean_pred, actual_frac, counts = calibration_curve(y_true, y_prob, n_bins=10)
    
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    print(f"{'Predicted':<12} {'Actual':<12} {'Count':<10} {'Gap':<10}")
    print("-" * 60)
    
    for i in range(len(mean_pred)):
        if counts[i] > 0:
            gap = mean_pred[i] - actual_frac[i]
            print(f"{mean_pred[i]:<12.1%} {actual_frac[i]:<12.1%} {int(counts[i]):<10} {gap:+.1%}")
    
    print("=" * 60)
    print("(A well-calibrated model has 'Predicted' ≈ 'Actual')")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda, mps)")
    
    # Logging options
    parser.add_argument("--log", action="store_true",
                        help="Log results to experiment tracker")
    parser.add_argument("--name", type=str, default="evaluation",
                        help="Experiment name for logging")
    parser.add_argument("--notes", type=str, default="",
                        help="Notes about this experiment")
    parser.add_argument("--tags", type=str, nargs="*", default=[],
                        help="Tags for filtering experiments")
    
    # Tracker options
    parser.add_argument("--compare", action="store_true",
                        help="Compare recent experiments")
    parser.add_argument("--summary", action="store_true",
                        help="Show experiment summary")
    parser.add_argument("--best", action="store_true",
                        help="Show best experiment")
    
    # Output options
    parser.add_argument("--calibration", action="store_true",
                        help="Show calibration analysis")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ExperimentTracker("experiments")
    
    # Handle tracker-only commands
    if args.summary:
        print(tracker.summary())
        return
    
    if args.compare:
        print(tracker.compare())
        return
    
    if args.best:
        best = tracker.get_best("brier_score", lower_is_better=True)
        if best:
            print(f"\n🏆 Best Model: {best.name} ({best.id})")
            print(f"   Brier Score: {best.metrics['brier_score']:.4f}")
            print(f"   Accuracy: {best.metrics['accuracy']:.2%}")
            print(f"   Path: {best.model_path}")
        else:
            print("No experiments logged yet.")
        return
    
    # Evaluate model
    if not args.model:
        parser.error("--model is required for evaluation")
    
    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Loading model from {args.model}...")
    print(f"Using device: {device}")
    
    model, model_config = load_model(args.model, device)
    
    print(f"Evaluating on {args.data}...")
    metrics, y_true, y_prob = evaluate_on_dataset(model, args.data, device)
    
    # Print results
    print_metrics(metrics)
    
    if args.calibration:
        print_calibration(y_true, y_prob)
    
    # Log to tracker
    if args.log:
        # Load data config
        metadata_path = Path(args.data) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data_config = json.load(f)
        else:
            data_config = {"path": args.data}
        
        # Training config (unknown for loaded models)
        training_config = {"unknown": True}
        
        exp_id = tracker.log_experiment(
            name=args.name,
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            metrics=metrics,
            model_path=args.model,
            notes=args.notes,
            tags=args.tags,
        )
        
        print(f"\n📊 Logged to experiment tracker: {exp_id}")


if __name__ == "__main__":
    main()

