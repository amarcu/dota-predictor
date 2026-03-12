"""
Feature extraction utilities.

This module handles the transformation of raw match data into
features suitable for machine learning models.
"""

import numpy as np

from dota_predictor.data.match import Match
from dota_predictor.utils.config import MAX_MINUTES, NUM_HEROES


class FeatureExtractor:
    """
    Extracts and engineers features from match data.

    Based on the paper, the key features are:
    - Time-series: gold, XP, last hits, denies at each minute
    - Team aggregates: total gold/XP, leads/deficits
    - Hero picks: can be used for embeddings or one-hot encoding
    """

    def __init__(
        self,
        max_minutes: int = MAX_MINUTES,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the feature extractor.

        Args:
            max_minutes: Maximum sequence length
            normalize: Whether to normalize features
        """
        self.max_minutes = max_minutes
        self.normalize = normalize

        # Normalization statistics (computed during fitting)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, matches: list[Match]) -> "FeatureExtractor":
        """
        Compute normalization statistics from training data.

        Args:
            matches: List of Match objects for computing stats

        Returns:
            Self for chaining
        """
        all_features = []
        for match in matches:
            if match.has_time_series:
                features = self.extract_time_series(match, normalize=False)
                all_features.append(features)

        if all_features:
            stacked = np.stack(all_features, axis=0)
            self._mean = np.mean(stacked, axis=(0, 1))
            self._std = np.std(stacked, axis=(0, 1))
            self._std = np.where(self._std > 0, self._std, 1.0)

        return self

    def extract_time_series(
        self,
        match: Match,
        normalize: bool | None = None,
    ) -> np.ndarray:
        """
        Extract time-series features from a match.

        Features per timestep:
        1. Radiant total gold
        2. Radiant total XP
        3. Dire total gold
        4. Dire total XP
        5. Gold difference (Radiant - Dire)
        6. XP difference (Radiant - Dire)
        7. Radiant total last hits
        8. Dire total last hits

        Args:
            match: Match object
            normalize: Whether to normalize (uses instance default if None)

        Returns:
            Array of shape (max_minutes, 8)
        """
        features = match.get_full_time_series(self.max_minutes)

        if normalize is None:
            normalize = self.normalize

        if normalize and self._mean is not None:
            features = (features - self._mean) / self._std

        return features

    def extract_static_features(self, match: Match) -> np.ndarray:
        """
        Extract static (non-time-series) features.

        Features:
        - Match duration
        - Game mode
        - Average team MMR (if available)
        - Hero picks (as IDs for embedding)

        Args:
            match: Match object

        Returns:
            Static feature array
        """
        return np.array([
            match.duration_minutes,
            match.game_mode,
            match.radiant_score,
            match.dire_score,
        ], dtype=np.float32)

    def extract_hero_features(
        self,
        match: Match,
        num_heroes: int = NUM_HEROES,
    ) -> np.ndarray:
        """
        Extract hero pick features as one-hot encoding.

        Creates two binary vectors:
        - Radiant picks (num_heroes,)
        - Dire picks (num_heroes,)

        Args:
            match: Match object
            num_heroes: Total number of heroes

        Returns:
            One-hot encoded hero picks (2 * num_heroes,)
        """
        radiant_picks = np.zeros(num_heroes, dtype=np.float32)
        dire_picks = np.zeros(num_heroes, dtype=np.float32)

        for hero_id in match.radiant_heroes:
            if 0 < hero_id < num_heroes:
                radiant_picks[hero_id] = 1.0

        for hero_id in match.dire_heroes:
            if 0 < hero_id < num_heroes:
                dire_picks[hero_id] = 1.0

        return np.concatenate([radiant_picks, dire_picks])

    def extract_at_minute(
        self,
        match: Match,
        minute: int,
    ) -> np.ndarray:
        """
        Extract features at a specific game minute.

        Useful for multi-forward step prediction.

        Args:
            match: Match object
            minute: Target minute

        Returns:
            Feature array at the specified minute
        """
        time_series = self.extract_time_series(match)
        if minute < len(time_series):
            return time_series[minute]
        return time_series[-1]

    def compute_derived_features(self, match: Match) -> dict[str, float]:
        """
        Compute derived/engineered features.

        These are additional features that might be predictive:
        - Gold/XP velocity (rate of change)
        - Comeback potential
        - Snowball index

        Args:
            match: Match object

        Returns:
            Dictionary of derived features
        """
        derived = {}

        if not match.has_time_series:
            return derived

        # Get time series
        radiant_gold = [
            sum(p.gold_t[m] if m < len(p.gold_t) else 0 for p in match.radiant_players)
            for m in range(match.duration_minutes)
        ]
        dire_gold = [
            sum(p.gold_t[m] if m < len(p.gold_t) else 0 for p in match.dire_players)
            for m in range(match.duration_minutes)
        ]

        if len(radiant_gold) >= 2:
            # Gold velocity (last 5 minutes average)
            window = min(5, len(radiant_gold) - 1)
            radiant_velocity = (radiant_gold[-1] - radiant_gold[-1 - window]) / window
            dire_velocity = (dire_gold[-1] - dire_gold[-1 - window]) / window

            derived["radiant_gold_velocity"] = radiant_velocity
            derived["dire_gold_velocity"] = dire_velocity
            derived["gold_velocity_diff"] = radiant_velocity - dire_velocity

            # Max gold lead for each team
            gold_diff = np.array(radiant_gold) - np.array(dire_gold)
            derived["max_radiant_lead"] = float(np.max(gold_diff))
            derived["max_dire_lead"] = float(-np.min(gold_diff))

            # Lead volatility (how much the lead changes)
            derived["lead_volatility"] = float(np.std(gold_diff))

        return derived

