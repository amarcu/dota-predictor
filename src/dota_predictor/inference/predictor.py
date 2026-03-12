"""
Real-time prediction engine for Dota 2 matches.

This module provides the LivePredictor class that:
1. Loads a trained model
2. Accepts real-time game state updates
3. Returns win probability predictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dota_predictor.models.lstm import LSTMPredictor
from dota_predictor.models.loader import load_model_from_checkpoint
from dota_predictor.evaluation.calibration import IsotonicCalibration, PhaseCalibrator


@dataclass
class GameState:
    """
    Represents the current state of a live Dota 2 match.
    
    Accumulates per-minute snapshots of the game state for prediction.
    Supports both basic (8 features) and enhanced (20 features) modes.
    """
    
    # Hero IDs for all 10 players (radiant first 5, dire last 5)
    heroes: list[int] = field(default_factory=list)
    
    # Time-series data: dict mapping minute -> snapshot
    # Enhanced snapshot (20 features):
    # [0-7]: radiant_gold, radiant_xp, dire_gold, dire_xp, gold_diff, xp_diff, radiant_lh, dire_lh
    # [8-10]: radiant_kills, dire_kills, kill_diff
    # [11-13]: radiant_towers, dire_towers, tower_diff
    # [14-16]: radiant_barracks, dire_barracks, barracks_diff
    # [17-19]: radiant_roshan, dire_roshan, roshan_diff
    minute_snapshots: dict[int, list[float]] = field(default_factory=dict)
    
    # Current game time in seconds
    game_time: float = 0.0
    
    # Match metadata
    match_id: int | None = None
    
    # Enhanced mode (20 features vs 8)
    enhanced: bool = True
    num_features: int = 20
    
    def current_minute(self) -> int:
        """Get the current game minute."""
        return int(self.game_time / 60)
    
    def add_snapshot(self, minute: int, snapshot: list[float]) -> None:
        """Add a new minute snapshot at the specified minute."""
        self.minute_snapshots[minute] = snapshot
    
    def get_features(self, max_minutes: int = 60) -> np.ndarray:
        """
        Convert accumulated snapshots to feature array for model input.
        
        Returns:
            numpy array of shape (max_minutes, num_features) with interpolated values
        """
        features = np.zeros((max_minutes, self.num_features), dtype=np.float32)
        
        # Initialize towers and barracks to starting values
        if self.enhanced:
            features[:, 11] = 11  # radiant_towers
            features[:, 12] = 11  # dire_towers
            features[:, 14] = 6   # radiant_barracks
            features[:, 15] = 6   # dire_barracks
        
        if not self.minute_snapshots:
            return features
        
        # Get sorted minutes
        sorted_minutes = sorted(self.minute_snapshots.keys())
        
        # Fill in each minute with interpolated or latest values
        for m in range(max_minutes):
            if m in self.minute_snapshots:
                # We have exact data for this minute
                snap = self.minute_snapshots[m]
                features[m, :len(snap)] = snap
            elif m < sorted_minutes[0]:
                # Before first snapshot - use first snapshot scaled down
                first_min = sorted_minutes[0]
                first_snap = np.array(self.minute_snapshots[first_min])
                if first_min > 0:
                    features[m, :len(first_snap)] = first_snap * (m / first_min)
            else:
                # After snapshots - use last known values
                prev_min = max(sm for sm in sorted_minutes if sm <= m)
                snap = self.minute_snapshots[prev_min]
                features[m, :len(snap)] = snap
        
        return features
    
    def get_heroes_array(self) -> np.ndarray:
        """Get hero IDs as numpy array."""
        if len(self.heroes) != 10:
            heroes = self.heroes + [0] * (10 - len(self.heroes))
        else:
            heroes = self.heroes
        return np.array(heroes, dtype=np.int32)
    
    def reset(self) -> None:
        """Reset game state for a new match."""
        self.heroes = []
        self.minute_snapshots = {}
        self.game_time = 0.0
        self.match_id = None


class LivePredictor:
    """
    Real-time prediction engine for Dota 2 matches.
    
    Example usage:
        predictor = LivePredictor("models/checkpoints/model.pt")
        predictor.set_heroes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # As game progresses, update with new data each minute:
        predictor.update(
            game_time=300,  # 5 minutes
            radiant_gold=25000,
            radiant_xp=15000,
            dire_gold=23000,
            dire_xp=14000,
            radiant_lh=150,
            dire_lh=140,
        )
        
        prob = predictor.predict()
        print(f"Radiant win probability: {prob:.1%}")
    """
    
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        normalization_path: str | Path | None = None,
        calibrator_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run inference on (cuda, mps, cpu)
            normalization_path: Path to normalization stats (.npz file)
            calibrator_path: Path to isotonic calibrator (.json file)
        """
        self.model_path = Path(model_path)
        
        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Initialize game state first (before loading model)
        self.game_state = GameState()
        
        # Track last processed minute to avoid duplicate updates
        self._last_minute = -1
        
        # Load model (may update game_state settings based on model config)
        self.model = self._load_model()
        self.model.eval()
        
        # Load normalization stats if provided
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if normalization_path:
            self._load_normalization(normalization_path)
        
        # Load calibrator if provided
        self.calibrator: IsotonicCalibration | None = None
        self.phase_calibrator: PhaseCalibrator | None = None
        if calibrator_path:
            self._load_calibrator(calibrator_path)
    
    def _load_model(self) -> LSTMPredictor:
        """Load the trained model from checkpoint."""
        model, model_config = load_model_from_checkpoint(
            self.model_path, device=self.device
        )

        self._use_hero_embedding = model_config["use_hero_embedding"]
        self._input_size = model_config["input_size"]

        self.game_state.enhanced = (self._input_size == 20)
        self.game_state.num_features = self._input_size

        return model
    
    def _load_normalization(self, path: str | Path) -> None:
        """Load normalization statistics."""
        with np.load(path) as data:
            self.mean = data["mean"]
            self.std = data["std"]
    
    def _load_calibrator(self, path: str | Path) -> None:
        """Load calibrator - tries phase calibrators first, then single calibrator."""
        path = Path(path)
        
        # Try loading phase calibrators from directory
        if path.is_dir() or path.suffix == "":
            # Treat as directory containing phase calibrators
            checkpoint_dir = path if path.is_dir() else path.parent
            phase_cal = PhaseCalibrator.load(checkpoint_dir)
            if phase_cal._loaded:
                self.phase_calibrator = phase_cal
                print(f"✅ Phase calibrators loaded (early/mid/late game)")
                return
        
        # Fall back to single calibrator
        if path.exists() and path.suffix == ".json":
            self.calibrator = IsotonicCalibration.load(path)
            print(f"Loaded single calibrator from {path}")
    
    def set_heroes(self, hero_ids: list[int]) -> None:
        """
        Set the hero picks for the match.
        
        Args:
            hero_ids: List of 10 hero IDs [radiant_1-5, dire_1-5]
        """
        if len(hero_ids) != 10:
            raise ValueError(f"Expected 10 heroes, got {len(hero_ids)}")
        self.game_state.heroes = hero_ids
    
    def update(
        self,
        game_time: float,
        radiant_gold: float,
        radiant_xp: float,
        dire_gold: float,
        dire_xp: float,
        radiant_lh: float = 0,
        dire_lh: float = 0,
        radiant_kills: float = 0,
        dire_kills: float = 0,
        radiant_towers: float = 11,
        dire_towers: float = 11,
        radiant_barracks: float = 6,
        dire_barracks: float = 6,
        radiant_roshan: float = 0,
        dire_roshan: float = 0,
    ) -> bool:
        """
        Update the game state with new data.
        
        This should be called with fresh data as the game progresses.
        The predictor will automatically aggregate data per minute.
        
        Args:
            game_time: Current game time in seconds
            radiant_gold: Total gold for Radiant team
            radiant_xp: Total XP for Radiant team
            dire_gold: Total gold for Dire team
            dire_xp: Total XP for Dire team
            radiant_lh: Total last hits for Radiant team
            dire_lh: Total last hits for Dire team
            radiant_kills: Total kills for Radiant team
            dire_kills: Total kills for Dire team
            radiant_towers: Towers remaining for Radiant (0-11)
            dire_towers: Towers remaining for Dire (0-11)
            radiant_barracks: Barracks remaining for Radiant (0-6)
            dire_barracks: Barracks remaining for Dire (0-6)
            radiant_roshan: Roshan kills by Radiant
            dire_roshan: Roshan kills by Dire
            
        Returns:
            True if a new minute snapshot was recorded
        """
        self.game_state.game_time = game_time
        current_minute = self.game_state.current_minute()
        
        # Only record when we cross into a new minute
        if current_minute > self._last_minute:
            # Enhanced snapshot with 20 features
            snapshot = [
                radiant_gold,                          # 0
                radiant_xp,                            # 1
                dire_gold,                             # 2
                dire_xp,                               # 3
                radiant_gold - dire_gold,              # 4: gold_diff
                radiant_xp - dire_xp,                  # 5: xp_diff
                radiant_lh,                            # 6
                dire_lh,                               # 7
                radiant_kills,                         # 8
                dire_kills,                            # 9
                radiant_kills - dire_kills,            # 10: kill_diff
                radiant_towers,                        # 11
                dire_towers,                           # 12
                radiant_towers - dire_towers,          # 13: tower_diff
                radiant_barracks,                      # 14
                dire_barracks,                         # 15
                radiant_barracks - dire_barracks,      # 16: barracks_diff
                radiant_roshan,                        # 17
                dire_roshan,                           # 18
                radiant_roshan - dire_roshan,          # 19: roshan_diff
            ]
            self.game_state.add_snapshot(current_minute, snapshot)
            self._last_minute = current_minute
            return True
        
        return False
    
    def update_from_gsi(self, gsi_data: dict[str, Any]) -> bool:
        """
        Update game state from a GSI (Game State Integration) payload.
        
        GSI sends detailed game state as JSON. This method extracts
        all relevant features including kills, towers, and objectives.
        
        Args:
            gsi_data: The GSI JSON payload from Dota 2
            
        Returns:
            True if a new minute snapshot was recorded
        """
        # Extract game time
        map_data = gsi_data.get("map", {})
        game_time = map_data.get("clock_time", 0)
        
        # Skip if game hasn't started (negative clock time = pre-game)
        if game_time < 0:
            return False
        
        # Initialize all features
        radiant_gold = 0
        radiant_xp = 0
        dire_gold = 0
        dire_xp = 0
        radiant_lh = 0
        dire_lh = 0
        radiant_kills = 0
        dire_kills = 0
        radiant_heroes: list[int] = []
        dire_heroes: list[int] = []
        
        # Get kill scores from map data
        radiant_kills = map_data.get("radiant_score", 0)
        dire_kills = map_data.get("dire_score", 0)
        
        # Spectator format: player stats in player.team2/team3
        player_data = gsi_data.get("player", {})
        team2_players = player_data.get("team2", {})  # Radiant players
        team3_players = player_data.get("team3", {})  # Dire players
        
        # Hero info is in a separate section: hero.team2/team3
        hero_data = gsi_data.get("hero", {})
        team2_heroes = hero_data.get("team2", {}) if isinstance(hero_data, dict) else {}
        team3_heroes = hero_data.get("team3", {}) if isinstance(hero_data, dict) else {}
        
        # Debug: Log data structure once at minute 1
        current_minute = int(game_time / 60)
        if current_minute >= 1 and self._last_minute < 1:
            print(f"\n🔍 DEBUG - GSI Data Keys: {list(gsi_data.keys())}")
            print(f"   player keys: {list(player_data.keys()) if player_data else 'EMPTY'}")
            print(f"   team2_players: {bool(team2_players)}, team3_players: {bool(team3_players)}")
            print(f"   allplayers: {'allplayers' in gsi_data}")
        
        if isinstance(team2_players, dict) and isinstance(team3_players, dict) and team2_players and team3_players:
            # Spectator mode - extract player stats
            for player_id, pdata in team2_players.items():
                if isinstance(pdata, dict):
                    radiant_gold += pdata.get("net_worth", pdata.get("gold", 0))
                    xpm = pdata.get("xpm", 0)
                    radiant_xp += int(xpm * (game_time / 60)) if xpm else 0
                    radiant_lh += pdata.get("last_hits", 0)
            
            for player_id, pdata in team3_players.items():
                if isinstance(pdata, dict):
                    dire_gold += pdata.get("net_worth", pdata.get("gold", 0))
                    xpm = pdata.get("xpm", 0)
                    dire_xp += int(xpm * (game_time / 60)) if xpm else 0
                    dire_lh += pdata.get("last_hits", 0)
            
            # Extract hero IDs
            for player_id, hdata in team2_heroes.items():
                if isinstance(hdata, dict) and hdata.get("id"):
                    radiant_heroes.append(hdata.get("id", 0))
            
            for player_id, hdata in team3_heroes.items():
                if isinstance(hdata, dict) and hdata.get("id"):
                    dire_heroes.append(hdata.get("id", 0))
        
        # Fallback: Try allplayers format (playing mode)
        elif "allplayers" in gsi_data:
            players = gsi_data["allplayers"]
            for player_id, pdata in players.items():
                team = pdata.get("team_name", "")
                gold = pdata.get("gold", 0)
                xp = pdata.get("xp", 0)
                lh = pdata.get("last_hits", 0)
                hero_id = pdata.get("hero_id", 0)
                
                if team == "radiant":
                    radiant_gold += gold
                    radiant_xp += xp
                    radiant_lh += lh
                    radiant_heroes.append(hero_id)
                else:
                    dire_gold += gold
                    dire_xp += xp
                    dire_lh += lh
                    dire_heroes.append(hero_id)
        else:
            # Debug: No valid player data found
            if current_minute >= 1 and self._last_minute < 1:
                print(f"   ⚠️ No valid player data format found!")
            return False
        
        # Extract tower counts from buildings if available
        # Note: GSI doesn't always provide building data, so we estimate from game state
        radiant_towers = 11
        dire_towers = 11
        radiant_barracks = 6
        dire_barracks = 6
        radiant_roshan = 0
        dire_roshan = 0
        
        buildings = gsi_data.get("buildings", {})
        if buildings:
            # Count remaining towers
            radiant_buildings = buildings.get("radiant", {})
            dire_buildings = buildings.get("dire", {})
            
            radiant_towers = sum(
                1 for b in radiant_buildings.values()
                if isinstance(b, dict) and b.get("health", 0) > 0 and "tower" in str(b.get("name", ""))
            )
            dire_towers = sum(
                1 for b in dire_buildings.values()
                if isinstance(b, dict) and b.get("health", 0) > 0 and "tower" in str(b.get("name", ""))
            )
        
        # Set heroes if not already set
        all_heroes = radiant_heroes + dire_heroes
        if len(all_heroes) == 10 and not self.game_state.heroes:
            self.game_state.heroes = all_heroes
        
        # Debug on first update and every 10 minutes
        current_minute = int(game_time / 60)
        if current_minute >= 1 and (self._last_minute < 1 or current_minute % 10 == 0 and current_minute > self._last_minute):
            print(f"\n📊 Data snapshot at minute {current_minute}:")
            print(f"   Radiant: gold={radiant_gold:,}, xp={radiant_xp:,}, kills={radiant_kills}")
            print(f"   Dire:    gold={dire_gold:,}, xp={dire_xp:,}, kills={dire_kills}")
            print(f"   Gold diff: {radiant_gold - dire_gold:+,} (positive = Radiant lead)")
            print(f"   Heroes:  {len(all_heroes)} found")
        
        return self.update(
            game_time=game_time,
            radiant_gold=radiant_gold,
            radiant_xp=radiant_xp,
            dire_gold=dire_gold,
            dire_xp=dire_xp,
            radiant_lh=radiant_lh,
            dire_lh=dire_lh,
            radiant_kills=radiant_kills,
            dire_kills=dire_kills,
            radiant_towers=radiant_towers,
            dire_towers=dire_towers,
            radiant_barracks=radiant_barracks,
            dire_barracks=dire_barracks,
            radiant_roshan=radiant_roshan,
            dire_roshan=dire_roshan,
        )
    
    def predict(self) -> float:
        """
        Generate a win probability prediction at the current game minute.
        
        Uses sequence prediction to get the prediction at the current timestep,
        not the final hidden state. This enables meaningful mid-game predictions.
        
        Returns:
            Probability (0-1) that Radiant will win
        """
        with torch.no_grad():
            # Get features
            features = self.game_state.get_features()
            
            # Normalize if we have stats
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            
            # Convert to tensors
            features_tensor = torch.tensor(features, dtype=torch.float32)
            features_tensor = features_tensor.unsqueeze(0)  # Add batch dim
            features_tensor = features_tensor.to(self.device)
            
            # Get heroes if available and model uses them
            heroes_tensor = None
            if getattr(self, '_use_hero_embedding', False) and self.game_state.heroes:
                heroes = self.game_state.get_heroes_array()
                heroes_tensor = torch.tensor(heroes, dtype=torch.long)
                heroes_tensor = heroes_tensor.unsqueeze(0)
                heroes_tensor = heroes_tensor.to(self.device)
            
            # Use sequence prediction to get probability at current minute
            seq_probs = self.model(features_tensor, heroes_tensor, return_sequence=True)
            
            # Get prediction at current minute (clamped to valid range)
            current_minute = min(self.game_state.current_minute(), 59)
            if current_minute < 1:
                current_minute = 1  # Use minute 1 if game just started
            
            raw_probability = seq_probs[0, current_minute].item()
            probability = raw_probability
            
            # Debug: Print raw vs calibrated every 5 minutes
            if current_minute % 5 == 0 and current_minute > 0:
                print(f"\n🔍 DEBUG Minute {current_minute}: raw_prob={raw_probability:.3f}", end="")
            
            # Apply phase-aware calibration if available (preferred for live)
            if self.phase_calibrator is not None:
                probability = self.phase_calibrator.calibrate(raw_probability, current_minute)
                if current_minute % 5 == 0 and current_minute > 0:
                    print(f" → calibrated={probability:.3f}")
            # Fall back to single calibrator
            elif self.calibrator is not None:
                probability = self.calibrator.calibrate(np.array([raw_probability]))[0]
            
            return probability
    
    def predict_at_minutes(self, minutes: list[int]) -> dict[int, float]:
        """
        Get predictions at multiple time points.
        
        Uses sequence prediction to efficiently get all predictions in one pass.
        
        Args:
            minutes: List of minute marks to predict at
            
        Returns:
            Dict mapping minute -> probability
        """
        predictions = {}
        
        if not self.game_state.minute_snapshots:
            return predictions
        
        max_recorded = max(self.game_state.minute_snapshots.keys())
        
        with torch.no_grad():
            # Get features and normalize
            features = self.game_state.get_features()
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            
            features_tensor = torch.tensor(features, dtype=torch.float32)
            features_tensor = features_tensor.unsqueeze(0).to(self.device)
            
            heroes_tensor = None
            if getattr(self, '_use_hero_embedding', False) and self.game_state.heroes:
                heroes = self.game_state.get_heroes_array()
                heroes_tensor = torch.tensor(heroes, dtype=torch.long)
                heroes_tensor = heroes_tensor.unsqueeze(0).to(self.device)
            
            # Get all sequence predictions in one pass
            seq_probs = self.model(features_tensor, heroes_tensor, return_sequence=True)
            
            for minute in minutes:
                if minute <= max_recorded and minute < 60:
                    predictions[minute] = seq_probs[0, minute].item()
        
        return predictions
    
    def reset(self) -> None:
        """Reset the predictor for a new match."""
        self.game_state.reset()
        self._last_minute = -1
    
    def get_prediction_summary(self) -> dict[str, Any]:
        """
        Get a detailed summary of the current prediction.
        
        Returns:
            Dictionary with prediction details
        """
        prob = self.predict()
        current_minute = self.game_state.current_minute()
        
        # Get latest stats
        if self.game_state.minute_snapshots:
            latest_minute = max(self.game_state.minute_snapshots.keys())
            latest = self.game_state.minute_snapshots[latest_minute]
            gold_diff = latest[4]
            xp_diff = latest[5]
        else:
            gold_diff = 0
            xp_diff = 0
        
        return {
            "radiant_win_probability": prob,
            "dire_win_probability": 1 - prob,
            "game_minute": current_minute,
            "gold_advantage": gold_diff,
            "xp_advantage": xp_diff,
            "confidence": abs(prob - 0.5) * 2,  # 0 = 50/50, 1 = certain
            "prediction": "Radiant" if prob > 0.5 else "Dire",
        }

