#!/usr/bin/env python3
"""
Build per-minute calibrators for live prediction.

Instead of one calibrator for end-of-game, this builds calibrators
for different game phases:
- Early game (0-10 min)
- Mid game (10-25 min)  
- Late game (25+ min)

This enables proper calibration for live betting/prediction.

Usage:
    python scripts/calibrate_per_minute.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dota_predictor.models.loader import load_model_from_checkpoint
from dota_predictor.evaluation.calibration import IsotonicCalibration


def load_model_and_data():
    """Load the trained model and validation data."""
    model_path = Path("models/checkpoints/model.pt")
    model, model_config = load_model_from_checkpoint(model_path)
    use_hero_embedding = model_config["use_hero_embedding"]
    
    # Load data (raw features — no normalization, matching training)
    features = np.load("data/processed/val_features.npy")
    labels = np.load("data/processed/val_labels.npy")
    durations = np.load("data/processed/val_durations.npy")
    masks = np.load("data/processed/val_masks.npy")
    
    # Load heroes if model uses them
    heroes = None
    if use_hero_embedding:
        heroes = np.load("data/processed/val_heroes.npy")
    
    return model, features, labels, durations, masks, heroes


def get_predictions_at_minute(model, features, heroes, minute: int) -> np.ndarray:
    """Get model predictions at a specific minute for all matches."""
    with torch.no_grad():
        features_t = torch.tensor(features, dtype=torch.float32)
        heroes_t = torch.tensor(heroes, dtype=torch.long) if heroes is not None else None
        
        # Get sequence predictions
        seq_probs = model(features_t, heroes_t, return_sequence=True)
        
        # Extract prediction at the specified minute
        probs = seq_probs[:, minute].numpy()
        
    return probs


def build_phase_calibrators():
    """Build calibrators for different game phases."""
    print("=" * 60)
    print("BUILDING PER-PHASE CALIBRATORS")
    print("=" * 60)
    
    model, features, labels, durations, masks, heroes = load_model_and_data()
    
    # Define game phases
    phases = {
        "early": (1, 10),    # Minutes 1-10
        "mid": (11, 25),     # Minutes 11-25
        "late": (26, 45),    # Minutes 26-45
    }
    
    calibrators = {}
    
    for phase_name, (min_start, min_end) in phases.items():
        print(f"\n{'='*40}")
        print(f"Phase: {phase_name.upper()} (minutes {min_start}-{min_end})")
        print(f"{'='*40}")
        
        all_probs = []
        all_labels = []
        
        # Collect predictions at each minute in this phase
        for minute in range(min_start, min_end + 1):
            # Only include matches that lasted at least this long
            valid_mask = durations > minute
            
            if valid_mask.sum() == 0:
                continue
            
            # Get predictions at this minute
            probs = get_predictions_at_minute(
                model, 
                features[valid_mask], 
                heroes[valid_mask] if heroes is not None else None,
                minute
            )
            
            all_probs.extend(probs)
            all_labels.extend(labels[valid_mask])
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        print(f"Samples: {len(all_probs):,}")
        print(f"Mean prediction: {all_probs.mean():.3f}")
        print(f"Actual win rate: {all_labels.mean():.3f}")
        
        # Analyze calibration BEFORE
        print("\nBefore calibration:")
        analyze_calibration(all_probs, all_labels)
        
        # Fit isotonic calibrator
        calibrator = IsotonicCalibration()
        calibrator.fit(all_probs, all_labels)
        
        # Analyze AFTER
        calibrated = calibrator.calibrate(all_probs)
        print("\nAfter calibration:")
        analyze_calibration(calibrated, all_labels)
        
        # Save calibrator
        output_path = f"models/checkpoints/calibrator_{phase_name}.json"
        calibrator.save(output_path)
        print(f"\nSaved to: {output_path}")
        
        calibrators[phase_name] = calibrator
    
    # Create combined config
    config = {
        "phases": {
            "early": {"minutes": [1, 10], "calibrator": "calibrator_early.json"},
            "mid": {"minutes": [11, 25], "calibrator": "calibrator_mid.json"},
            "late": {"minutes": [26, 60], "calibrator": "calibrator_late.json"},
        }
    }
    
    config_path = "models/checkpoints/calibrator_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config to: {config_path}")
    
    return calibrators


def analyze_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Analyze calibration quality."""
    bins = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            predicted = probs[mask].mean()
            actual = labels[mask].mean()
            count = mask.sum()
            error = abs(predicted - actual)
            bar = "█" * int(count / len(probs) * 50)
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: pred={predicted:.2f} actual={actual:.2f} err={error:.2f} n={count:5d} {bar}")


def main():
    calibrators = build_phase_calibrators()
    
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print("\nYou now have phase-specific calibrators:")
    print("  - Early game (min 1-10):  models/checkpoints/calibrator_early.json")
    print("  - Mid game (min 11-25):   models/checkpoints/calibrator_mid.json")
    print("  - Late game (min 26+):    models/checkpoints/calibrator_late.json")
    print("\nThese can be integrated into live prediction for better accuracy!")


if __name__ == "__main__":
    main()

