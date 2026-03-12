"""
Calibration methods for probability predictions.

Includes:
- TemperatureScaling: Simple single-parameter scaling
- IsotonicCalibration: Non-parametric, most effective
- PlattScaling: Logistic regression on logits
"""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader


class TemperatureScaling:
    """
    Temperature scaling for calibrating probability predictions.
    
    For under-confident models (predictions compressed toward 50%),
    temperature < 1 will push predictions toward 0 or 1.
    
    For over-confident models, temperature > 1 will smooth predictions.
    
    Usage:
        calibrator = TemperatureScaling()
        calibrator.fit(model_probs, true_labels)
        calibrated_probs = calibrator.calibrate(new_probs)
    """
    
    def __init__(self, lr: float = 0.01, max_iter: int = 100):
        """
        Initialize temperature scaling.
        
        Args:
            lr: Learning rate for optimization
            max_iter: Maximum iterations for fitting
        """
        self.lr = lr
        self.max_iter = max_iter
        self.temperature = 1.0
        self._fitted = False
    
    def _prob_to_logit(self, p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Convert probability to logit."""
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    
    def _logit_to_prob(self, logit: np.ndarray) -> np.ndarray:
        """Convert logit to probability."""
        return 1 / (1 + np.exp(-logit))
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Fit the temperature parameter.
        
        Args:
            probs: Predicted probabilities from the model
            labels: True binary labels
            
        Returns:
            Optimal temperature
        """
        # Convert to logits
        logits = self._prob_to_logit(probs)
        
        # Convert to tensors
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        
        # Temperature is a learnable parameter
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=self.lr, max_iter=self.max_iter)
        
        criterion = nn.BCEWithLogitsLoss()
        
        def closure():
            optimizer.zero_grad()
            # Scale logits by temperature
            scaled_logits = logits_t / temperature
            loss = criterion(scaled_logits, labels_t)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        self.temperature = temperature.item()
        self._fitted = True
        
        return self.temperature
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        Args:
            probs: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise ValueError("Must call fit() before calibrate()")
        
        # Convert to logits, scale, convert back
        logits = self._prob_to_logit(probs)
        scaled_logits = logits / self.temperature
        return self._logit_to_prob(scaled_logits)
    
    def __repr__(self) -> str:
        status = f"T={self.temperature:.4f}" if self._fitted else "not fitted"
        return f"TemperatureScaling({status})"


def fit_temperature_scaling(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> TemperatureScaling:
    """
    Fit temperature scaling on a validation set.
    
    Args:
        model: Trained model
        dataloader: Validation data loader
        device: Device to run on
        
    Returns:
        Fitted TemperatureScaling object
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
            
            # Get predictions
            if hasattr(model, "use_hero_embedding") and model.use_hero_embedding and heroes is not None:
                outputs = model(features, heroes.to(device), return_sequence=True)
            else:
                outputs = model(features, return_sequence=True)
            
            # Get prediction at last valid timestep
            batch_size = outputs.size(0)
            probs = torch.stack([
                outputs[i, min(durations[i].item() - 1, 59)] 
                for i in range(batch_size)
            ])
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    
    # Fit temperature scaling
    calibrator = TemperatureScaling()
    temperature = calibrator.fit(probs, labels)
    
    print(f"Fitted temperature: {temperature:.4f}")
    if temperature < 1:
        print(f"  → Model was under-confident, scaling will sharpen predictions")
    else:
        print(f"  → Model was over-confident, scaling will smooth predictions")
    
    return calibrator


class IsotonicCalibration:
    """
    Isotonic regression for calibrating probability predictions.
    
    This is a non-parametric method that learns a monotonic mapping
    from model probabilities to calibrated probabilities.
    
    Most effective calibration method, especially for models with
    severely compressed probability outputs.
    
    Usage:
        calibrator = IsotonicCalibration()
        calibrator.fit(model_probs, true_labels)
        calibrated_probs = calibrator.calibrate(new_probs)
        calibrator.save("calibrator.json")
    """
    
    def __init__(self):
        """Initialize isotonic calibration."""
        self._model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        self._fitted = False
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibration":
        """
        Fit the isotonic regression model.
        
        Args:
            probs: Predicted probabilities from the model
            labels: True binary labels
            
        Returns:
            Self for chaining
        """
        self._model.fit(probs, labels)
        self._fitted = True
        return self
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration to probabilities.
        
        Args:
            probs: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise ValueError("Must call fit() before calibrate()")
        
        return self._model.predict(probs)
    
    def save(self, path: str | Path) -> None:
        """
        Save calibration model to JSON file.
        
        Args:
            path: Path to save the calibrator
        """
        if not self._fitted:
            raise ValueError("Must fit before saving")
        
        # Save the isotonic regression as lookup table
        data = {
            "type": "isotonic",
            "x_thresholds": self._model.X_thresholds_.tolist(),
            "y_thresholds": self._model.y_thresholds_.tolist(),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibration":
        """
        Load calibration model from JSON file.
        
        Args:
            path: Path to the saved calibrator
            
        Returns:
            Loaded calibrator
        """
        from scipy.interpolate import interp1d
        
        with open(path) as f:
            data = json.load(f)
        
        calibrator = cls()
        x_thresholds = np.array(data["x_thresholds"])
        y_thresholds = np.array(data["y_thresholds"])
        
        calibrator._model.X_thresholds_ = x_thresholds
        calibrator._model.y_thresholds_ = y_thresholds
        calibrator._model.X_min_ = x_thresholds[0]
        calibrator._model.X_max_ = x_thresholds[-1]
        calibrator._model.increasing_ = True
        
        # Recreate the interpolation function (sklearn internal)
        calibrator._model.f_ = interp1d(
            x_thresholds, y_thresholds, 
            kind='linear', 
            bounds_error=False,
            fill_value=(y_thresholds[0], y_thresholds[-1])
        )
        
        calibrator._fitted = True
        
        return calibrator
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"IsotonicCalibration({status})"


class PhaseCalibrator:
    """
    Phase-aware calibrator for live prediction.
    
    Uses different calibrators for different game phases:
    - Early game (minutes 1-10): predictions close to 50%
    - Mid game (minutes 11-25): moderate differentiation
    - Late game (minutes 26+): strong predictive signal
    
    Usage:
        calibrator = PhaseCalibrator.load("models/checkpoints/")
        calibrated_prob = calibrator.calibrate(raw_prob, game_minute=15)
    """
    
    def __init__(self):
        """Initialize phase calibrator."""
        self._calibrators: dict[str, IsotonicCalibration] = {}
        self._phases = {
            "early": (1, 10),
            "mid": (11, 25),
            "late": (26, 60),
        }
        self._loaded = False
    
    def _get_phase(self, minute: int) -> str:
        """Determine which phase a minute belongs to."""
        for phase_name, (min_start, min_end) in self._phases.items():
            if min_start <= minute <= min_end:
                return phase_name
        return "late"  # Default to late for very long games
    
    def calibrate(self, prob: float, minute: int) -> float:
        """
        Calibrate a probability based on the current game minute.
        
        Args:
            prob: Raw model probability (0-1)
            minute: Current game minute
            
        Returns:
            Calibrated probability
        """
        if not self._loaded:
            return prob
        
        phase = self._get_phase(minute)
        calibrator = self._calibrators.get(phase)
        
        if calibrator is None:
            return prob
        
        return calibrator.calibrate(np.array([prob]))[0]
    
    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "PhaseCalibrator":
        """
        Load phase calibrators from checkpoint directory.
        
        Expects files:
        - calibrator_early.json
        - calibrator_mid.json
        - calibrator_late.json
        
        Args:
            checkpoint_dir: Directory containing calibrator files
            
        Returns:
            Loaded PhaseCalibrator
        """
        calibrator = cls()
        checkpoint_dir = Path(checkpoint_dir)
        
        phases_found = 0
        for phase_name in ["early", "mid", "late"]:
            path = checkpoint_dir / f"calibrator_{phase_name}.json"
            if path.exists():
                try:
                    calibrator._calibrators[phase_name] = IsotonicCalibration.load(path)
                    phases_found += 1
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
        
        if phases_found > 0:
            calibrator._loaded = True
            print(f"Loaded {phases_found} phase calibrators (early/mid/late)")
        
        return calibrator
    
    def __repr__(self) -> str:
        phases = list(self._calibrators.keys())
        return f"PhaseCalibrator(phases={phases})"
