"""
PyTorch Dataset for Dota 2 matches.

This module provides a PyTorch-compatible Dataset class that can be used
with DataLoader for training the prediction models.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dota_predictor.utils.config import MAX_MINUTES

from dota_predictor.data.match import Match


class DotaDataset(Dataset):
    """
    PyTorch Dataset for Dota 2 match prediction.

    Each sample consists of:
    - Time-series features (game state at each minute)
    - Static features (hero picks, game mode, etc.)
    - Label (radiant_win: 0 or 1)

    This dataset is designed to support:
    1. Full match prediction (given entire game history)
    2. Early prediction (given first N minutes)
    3. Multi-step prediction (predict at multiple time points)
    """

    def __init__(
        self,
        matches: list[Match] | None = None,
        data_path: str | Path | None = None,
        max_minutes: int = MAX_MINUTES,
        min_duration_minutes: int = 15,
        normalize: bool = False,
        prediction_minute: int | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            matches: List of Match objects (if loading from memory)
            data_path: Path to JSON file with match data (alternative to matches)
            max_minutes: Maximum sequence length
            min_duration_minutes: Minimum match duration to include
            normalize: Whether to normalize features
            prediction_minute: If set, only use data up to this minute
        """
        self.max_minutes = max_minutes
        self.min_duration = min_duration_minutes
        self.normalize = normalize
        self.prediction_minute = prediction_minute

        # Load matches
        if matches is not None:
            self.matches = matches
        elif data_path is not None:
            self.matches = self._load_from_file(Path(data_path))
        else:
            self.matches = []

        # Filter matches
        self.matches = [
            m for m in self.matches
            if m.duration_minutes >= min_duration_minutes and m.has_time_series
        ]

        # Compute normalization stats if needed
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        if normalize and len(self.matches) > 0:
            self._compute_normalization_stats()

    def _load_from_file(self, path: Path) -> list[Match]:
        """Load matches from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        return [Match.from_api_response(m) for m in data]

    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for normalization."""
        all_features = []
        for match in self.matches:
            features = match.get_full_time_series(self.max_minutes)
            all_features.append(features)

        stacked = np.stack(all_features, axis=0)
        self._mean = np.mean(stacked, axis=(0, 1))
        self._std = np.std(stacked, axis=(0, 1))
        # Avoid division by zero
        self._std = np.where(self._std > 0, self._std, 1.0)

    def __len__(self) -> int:
        """Return number of matches in dataset."""
        return len(self.matches)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
            - features: Time-series features (seq_len, num_features)
            - heroes: Hero IDs for both teams (10,)
            - label: Win label (1 if radiant wins)
            - duration: Actual match duration in minutes
            - mask: Valid timestep mask (seq_len,)
        """
        match = self.matches[idx]

        # Get time-series features
        features = match.get_full_time_series(self.max_minutes)

        # Apply prediction minute cutoff if specified
        if self.prediction_minute is not None:
            cutoff = min(self.prediction_minute, match.duration_minutes)
            features[cutoff:] = 0

        # Normalize
        if self.normalize and self._mean is not None:
            features = (features - self._mean) / self._std

        # Create attention mask (1 for valid timesteps, 0 for padding)
        duration = match.duration_minutes
        if self.prediction_minute is not None:
            duration = min(duration, self.prediction_minute)
        mask = np.zeros(self.max_minutes, dtype=np.float32)
        mask[:duration] = 1.0

        # Get hero IDs
        heroes = np.array(
            match.radiant_heroes + match.dire_heroes,
            dtype=np.int64,
        )

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "heroes": torch.tensor(heroes, dtype=torch.long),
            "label": torch.tensor(1.0 if match.radiant_win else 0.0, dtype=torch.float32),
            "duration": torch.tensor(duration, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        data = []
        for match in self.matches:
            # Convert back to API format for saving
            match_dict: dict[str, Any] = {
                "match_id": match.match_id,
                "radiant_win": match.radiant_win,
                "duration": match.duration,
                "start_time": match.start_time,
                "game_mode": match.game_mode,
                "lobby_type": match.lobby_type,
                "patch": match.patch,
                "radiant_score": match.radiant_score,
                "dire_score": match.dire_score,
                "players": [],
            }
            for player in match.players:
                player_dict = {
                    "player_slot": player.player_slot,
                    "account_id": player.account_id,
                    "hero_id": player.hero_id,
                    "kills": player.kills,
                    "deaths": player.deaths,
                    "assists": player.assists,
                    "last_hits": player.last_hits,
                    "denies": player.denies,
                    "gold_per_min": player.gold_per_min,
                    "xp_per_min": player.xp_per_min,
                    "hero_damage": player.hero_damage,
                    "tower_damage": player.tower_damage,
                    "hero_healing": player.hero_healing,
                    "level": player.level,
                    "gold_t": player.gold_t,
                    "xp_t": player.xp_t,
                    "lh_t": player.lh_t,
                    "dn_t": player.dn_t,
                    "item_0": player.item_0,
                    "item_1": player.item_1,
                    "item_2": player.item_2,
                    "item_3": player.item_3,
                    "item_4": player.item_4,
                    "item_5": player.item_5,
                }
                match_dict["players"].append(player_dict)
            data.append(match_dict)

        with open(path, "w") as f:
            json.dump(data, f)

    def get_feature_dim(self) -> int:
        """Get the feature dimension."""
        return 8  # Based on get_full_time_series output

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple["DotaDataset", "DotaDataset"]:
        """
        Split dataset into train and validation sets.

        Args:
            train_ratio: Fraction of data for training
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.matches))

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_matches = [self.matches[i] for i in train_indices]
        val_matches = [self.matches[i] for i in val_indices]

        # Create new datasets with same settings
        train_ds = DotaDataset(
            matches=train_matches,
            max_minutes=self.max_minutes,
            min_duration_minutes=self.min_duration,
            normalize=self.normalize,
            prediction_minute=self.prediction_minute,
        )
        val_ds = DotaDataset(
            matches=val_matches,
            max_minutes=self.max_minutes,
            min_duration_minutes=self.min_duration,
            normalize=self.normalize,
            prediction_minute=self.prediction_minute,
        )

        # Share normalization stats
        if self._mean is not None:
            train_ds._mean = self._mean
            train_ds._std = self._std
            val_ds._mean = self._mean
            val_ds._std = self._std

        return train_ds, val_ds

