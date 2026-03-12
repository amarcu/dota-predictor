"""
Baseline models for comparison.

These simpler models serve as baselines to compare against the LSTM model.
The paper also uses Linear Regression and simple Neural Networks.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dota_predictor.data.match import Match


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline using scikit-learn.

    This model uses aggregated features rather than time-series data,
    similar to the Linear Regression baseline in the paper.
    """

    def __init__(self) -> None:
        """Initialize the baseline model."""
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _extract_features(self, match: Match, minute: int | None = None) -> np.ndarray:
        """
        Extract aggregated features from a match.

        Features:
        - Total team gold/XP at specified minute (or end of game)
        - Gold/XP difference
        - Total kills/deaths
        - Hero composition stats (if available)
        """
        if minute is None:
            minute = match.duration_minutes - 1

        minute = min(minute, match.duration_minutes - 1)

        # Get team features at the specified minute
        radiant_feats = match.get_team_features_at_time(minute, "radiant")
        dire_feats = match.get_team_features_at_time(minute, "dire")

        # Combine features
        features = np.concatenate([
            radiant_feats,
            dire_feats,
            [match.duration_minutes],
        ])

        return features

    def fit(
        self,
        matches: list[Match],
        prediction_minute: int | None = None,
    ) -> "LogisticRegressionBaseline":
        """
        Fit the model on training data.

        Args:
            matches: List of Match objects
            prediction_minute: Minute at which to make predictions

        Returns:
            Self for chaining
        """
        X = np.array([
            self._extract_features(m, prediction_minute)
            for m in matches
        ])
        y = np.array([1 if m.radiant_win else 0 for m in matches])

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        return self

    def predict(
        self,
        matches: list[Match],
        prediction_minute: int | None = None,
    ) -> np.ndarray:
        """
        Predict match outcomes.

        Args:
            matches: List of Match objects
            prediction_minute: Minute at which to make predictions

        Returns:
            Array of win probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        X = np.array([
            self._extract_features(m, prediction_minute)
            for m in matches
        ])
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)[:, 1]

    def score(
        self,
        matches: list[Match],
        prediction_minute: int | None = None,
    ) -> float:
        """
        Calculate accuracy score.

        Args:
            matches: List of Match objects
            prediction_minute: Minute at which to make predictions

        Returns:
            Accuracy score
        """
        predictions = self.predict(matches, prediction_minute) > 0.5
        labels = np.array([m.radiant_win for m in matches])
        return float(np.mean(predictions == labels))


class SimpleNNBaseline(nn.Module):
    """
    Simple Neural Network baseline (non-sequential).

    Uses aggregated features similar to LogisticRegressionBaseline
    but with a neural network for non-linear relationships.
    """

    def __init__(
        self,
        input_size: int = 13,  # Based on feature extraction
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        layers: list[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        logits = self.network(x).squeeze(-1)
        return torch.sigmoid(logits)

