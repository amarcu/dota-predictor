"""
Data models for Dota 2 matches.

These models represent the structured data format compatible with the
research paper arXiv:2106.01782. The key insight is that we capture
time-series data at minute intervals for gold, XP, last hits, etc.

Enhanced with:
- Objective tracking (towers, barracks, Roshan with timestamps)
- Kill events with timestamps
- Per-minute kill/objective state reconstruction
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ObjectiveEvent:
    """A single objective event (tower, barracks, roshan, etc.)."""
    
    time: int  # Game time in seconds
    event_type: str  # 'tower_kill', 'barracks_kill', 'roshan_kill', 'aegis'
    team: str  # 'radiant' or 'dire' (team that GOT the objective)
    building: str | None = None  # e.g., 'tower1_top', 'rax_melee_mid'


@dataclass
class KillEvent:
    """A single kill event."""
    
    time: int  # Game time in seconds
    killer_team: str  # 'radiant' or 'dire'


@dataclass
class MatchPlayer:
    """
    Represents a single player's data in a match.

    The time-series arrays (gold_t, xp_t, etc.) are minute-by-minute
    snapshots of the player's state. These are the primary features
    for the LSTM model.
    """

    # Basic info
    player_slot: int  # 0-4 = Radiant, 128-132 = Dire
    account_id: int | None
    hero_id: int

    # Match statistics
    kills: int
    deaths: int
    assists: int
    last_hits: int
    denies: int
    gold_per_min: int
    xp_per_min: int
    hero_damage: int
    tower_damage: int
    hero_healing: int
    level: int

    # Time series data (per minute)
    gold_t: list[int] = field(default_factory=list)  # Gold at each minute
    xp_t: list[int] = field(default_factory=list)  # XP at each minute
    lh_t: list[int] = field(default_factory=list)  # Last hits at each minute
    dn_t: list[int] = field(default_factory=list)  # Denies at each minute
    kills_t: list[dict] = field(default_factory=list)  # Kill log entries from API

    # Item build (6 item slots)
    item_0: int = 0
    item_1: int = 0
    item_2: int = 0
    item_3: int = 0
    item_4: int = 0
    item_5: int = 0

    @property
    def is_radiant(self) -> bool:
        """Check if player is on Radiant team."""
        return self.player_slot < 128

    @property
    def team_slot(self) -> int:
        """Get slot within team (0-4)."""
        return self.player_slot if self.is_radiant else self.player_slot - 128

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "MatchPlayer":
        """Create MatchPlayer from OpenDota API response."""
        return cls(
            player_slot=data.get("player_slot", 0),
            account_id=data.get("account_id"),
            hero_id=data.get("hero_id", 0),
            kills=data.get("kills", 0),
            deaths=data.get("deaths", 0),
            assists=data.get("assists", 0),
            last_hits=data.get("last_hits", 0),
            denies=data.get("denies", 0),
            gold_per_min=data.get("gold_per_min", 0),
            xp_per_min=data.get("xp_per_min", 0),
            hero_damage=data.get("hero_damage", 0),
            tower_damage=data.get("tower_damage", 0),
            hero_healing=data.get("hero_healing", 0),
            level=data.get("level", 1),
            gold_t=data.get("gold_t", []),
            xp_t=data.get("xp_t", []),
            lh_t=data.get("lh_t", []),
            dn_t=data.get("dn_t", []),
            kills_t=data.get("kills_log", []),  # Note: different field name
            item_0=data.get("item_0", 0),
            item_1=data.get("item_1", 0),
            item_2=data.get("item_2", 0),
            item_3=data.get("item_3", 0),
            item_4=data.get("item_4", 0),
            item_5=data.get("item_5", 0),
        )

    def get_time_series_features(self, max_minutes: int = 60) -> np.ndarray:
        """
        Get time-series features as a numpy array.

        Returns array of shape (max_minutes, num_features) where features are:
        [gold, xp, last_hits, denies]

        Padded with zeros if match is shorter than max_minutes.
        """
        features = np.zeros((max_minutes, 4), dtype=np.float32)

        # Get actual length
        length = min(len(self.gold_t), max_minutes)

        if length > 0:
            features[:length, 0] = self.gold_t[:length]
            features[:length, 1] = self.xp_t[:length] if self.xp_t else 0
            features[:length, 2] = self.lh_t[:length] if self.lh_t else 0
            features[:length, 3] = self.dn_t[:length] if self.dn_t else 0

        return features


@dataclass
class Match:
    """
    Represents a complete Dota 2 match.

    This is the primary data structure for training. Each match contains
    10 players (5 per team) with their time-series data, plus objective
    and kill event tracking for enhanced predictions.
    """

    match_id: int
    radiant_win: bool  # The label we predict
    duration: int  # Match duration in seconds
    start_time: int  # Unix timestamp
    game_mode: int
    lobby_type: int
    patch: int | None
    players: list[MatchPlayer]

    # Aggregate team stats (computed from players)
    radiant_score: int = 0
    dire_score: int = 0
    
    # Objective events (towers, barracks, roshan)
    objective_events: list[ObjectiveEvent] = field(default_factory=list)
    
    # Kill events
    kill_events: list[KillEvent] = field(default_factory=list)
    
    # Final objective status (bitmasks)
    tower_status_radiant: int = 2047  # All 11 towers
    tower_status_dire: int = 2047
    barracks_status_radiant: int = 63  # All 6 barracks
    barracks_status_dire: int = 63

    @property
    def duration_minutes(self) -> int:
        """Get match duration in minutes."""
        return self.duration // 60

    @property
    def radiant_players(self) -> list[MatchPlayer]:
        """Get Radiant team players."""
        return [p for p in self.players if p.is_radiant]

    @property
    def dire_players(self) -> list[MatchPlayer]:
        """Get Dire team players."""
        return [p for p in self.players if not p.is_radiant]

    @property
    def radiant_heroes(self) -> list[int]:
        """Get Radiant hero IDs."""
        return [p.hero_id for p in self.radiant_players]

    @property
    def dire_heroes(self) -> list[int]:
        """Get Dire hero IDs."""
        return [p.hero_id for p in self.dire_players]

    @property
    def has_time_series(self) -> bool:
        """Check if match has parsed time-series data."""
        return any(len(p.gold_t) > 0 for p in self.players)

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "Match":
        """Create Match from OpenDota API response."""
        players = [
            MatchPlayer.from_api_response(p) for p in data.get("players", [])
        ]
        
        # Extract kill events from player kills_log
        kill_events: list[KillEvent] = []
        for player in data.get("players", []):
            player_slot = player.get("player_slot", 0)
            is_radiant = player_slot < 128
            for kill in player.get("kills_log", []):
                kill_events.append(KillEvent(
                    time=kill.get("time", 0),
                    killer_team="radiant" if is_radiant else "dire",
                ))
        
        # Extract objective events
        objective_events: list[ObjectiveEvent] = []
        for obj in data.get("objectives", []):
            obj_type = obj.get("type", "")
            time = obj.get("time", 0)
            
            if obj_type == "building_kill":
                key = obj.get("key", "")
                # "badguys" = Dire buildings destroyed -> Radiant got objective
                team = "radiant" if "badguys" in key else "dire"
                event_type = "tower_kill" if "tower" in key else "barracks_kill"
                
                # Parse building name
                building = cls._parse_building_name(key)
                
                objective_events.append(ObjectiveEvent(
                    time=time,
                    event_type=event_type,
                    team=team,
                    building=building,
                ))
            
            elif obj_type == "CHAT_MESSAGE_ROSHAN_KILL":
                team = "radiant" if obj.get("team") == 2 else "dire"
                objective_events.append(ObjectiveEvent(
                    time=time,
                    event_type="roshan_kill",
                    team=team,
                ))

        return cls(
            match_id=data.get("match_id", 0),
            radiant_win=data.get("radiant_win", False),
            duration=data.get("duration", 0),
            start_time=data.get("start_time", 0),
            game_mode=data.get("game_mode", 0),
            lobby_type=data.get("lobby_type", 0),
            patch=data.get("patch"),
            players=players,
            radiant_score=data.get("radiant_score", 0),
            dire_score=data.get("dire_score", 0),
            objective_events=objective_events,
            kill_events=kill_events,
            tower_status_radiant=data.get("tower_status_radiant", 2047),
            tower_status_dire=data.get("tower_status_dire", 2047),
            barracks_status_radiant=data.get("barracks_status_radiant", 63),
            barracks_status_dire=data.get("barracks_status_dire", 63),
        )
    
    @staticmethod
    def _parse_building_name(key: str) -> str:
        """Parse building name from key like 'npc_dota_badguys_tower1_top'."""
        parts = key.split("_")
        if "tower" in key:
            tower_num = "tower"
            for part in parts:
                if part.startswith("tower"):
                    tower_num = part
                    break
            lane = parts[-1] if parts[-1] in ["top", "mid", "bot"] else "base"
            return f"{tower_num}_{lane}"
        elif "rax" in key:
            rax_type = "melee" if "melee" in key else "range"
            lane = parts[-1] if parts[-1] in ["top", "mid", "bot"] else "mid"
            return f"rax_{rax_type}_{lane}"
        return key

    def get_team_features_at_time(
        self,
        minute: int,
        team: str = "radiant",
    ) -> np.ndarray:
        """
        Get aggregated team features at a specific minute.

        Features per team:
        - Total gold
        - Total XP (actual per-minute values from xp_t)
        - Total last hits
        - Total denies
        - Gold/XP lead (difference from opponent)

        Args:
            minute: The minute to get features for
            team: "radiant" or "dire"

        Returns:
            numpy array of team features
        """
        players = self.radiant_players if team == "radiant" else self.dire_players
        opponent = self.dire_players if team == "radiant" else self.radiant_players

        team_gold = sum(p.gold_t[minute] if minute < len(p.gold_t) else 0 for p in players)
        team_lh = sum(p.lh_t[minute] if minute < len(p.lh_t) else 0 for p in players)
        team_dn = sum(p.dn_t[minute] if minute < len(p.dn_t) else 0 for p in players)

        opp_gold = sum(p.gold_t[minute] if minute < len(p.gold_t) else 0 for p in opponent)

        # XP calculation - use actual xp_t values (matches GSI's xpm * time semantics)
        # Both represent "current XP at this point in time"
        team_xp = sum(p.xp_t[minute] if minute < len(p.xp_t) else 0 for p in players)
        opp_xp = sum(p.xp_t[minute] if minute < len(p.xp_t) else 0 for p in opponent)

        return np.array(
            [
                team_gold,
                team_xp,
                team_lh,
                team_dn,
                team_gold - opp_gold,  # Gold lead
                team_xp - opp_xp,  # XP lead
            ],
            dtype=np.float32,
        )

    def get_kills_per_minute(self, max_minutes: int = 60) -> np.ndarray:
        """
        Get cumulative kills per team at each minute.
        
        Returns:
            Array of shape (max_minutes, 2) with [radiant_kills, dire_kills]
        """
        kills = np.zeros((max_minutes, 2), dtype=np.float32)
        
        for event in self.kill_events:
            minute = event.time // 60
            if minute < max_minutes:
                if event.killer_team == "radiant":
                    kills[minute:, 0] += 1
                else:
                    kills[minute:, 1] += 1
        
        return kills
    
    def get_towers_per_minute(self, max_minutes: int = 60) -> np.ndarray:
        """
        Get tower count per team at each minute.
        
        Returns:
            Array of shape (max_minutes, 2) with [radiant_towers, dire_towers]
        """
        towers = np.full((max_minutes, 2), 11, dtype=np.float32)
        
        for event in self.objective_events:
            if event.event_type == "tower_kill":
                minute = event.time // 60
                if minute < max_minutes:
                    # Team that got the kill -> opponent loses tower
                    if event.team == "radiant":
                        towers[minute:, 1] -= 1  # Dire loses tower
                    else:
                        towers[minute:, 0] -= 1  # Radiant loses tower
        
        return np.clip(towers, 0, 11)
    
    def get_barracks_per_minute(self, max_minutes: int = 60) -> np.ndarray:
        """
        Get barracks count per team at each minute.
        
        Returns:
            Array of shape (max_minutes, 2) with [radiant_barracks, dire_barracks]
        """
        barracks = np.full((max_minutes, 2), 6, dtype=np.float32)
        
        for event in self.objective_events:
            if event.event_type == "barracks_kill":
                minute = event.time // 60
                if minute < max_minutes:
                    if event.team == "radiant":
                        barracks[minute:, 1] -= 1
                    else:
                        barracks[minute:, 0] -= 1
        
        return np.clip(barracks, 0, 6)
    
    def get_roshan_per_minute(self, max_minutes: int = 60) -> np.ndarray:
        """
        Get cumulative Roshan kills per team at each minute.
        
        Returns:
            Array of shape (max_minutes, 2) with [radiant_rosh, dire_rosh]
        """
        roshan = np.zeros((max_minutes, 2), dtype=np.float32)
        
        for event in self.objective_events:
            if event.event_type == "roshan_kill":
                minute = event.time // 60
                if minute < max_minutes:
                    if event.team == "radiant":
                        roshan[minute:, 0] += 1
                    else:
                        roshan[minute:, 1] += 1
        
        return roshan

    def get_full_time_series(
        self,
        max_minutes: int = 60,
        enhanced: bool = False,
    ) -> np.ndarray:
        """
        Get complete time-series features for the match.

        Returns array of shape (max_minutes, num_features) where each row
        contains features for both teams at that minute.

        XP values use actual xp_t data (not xp_per_min estimation) to avoid
        data leakage and align with GSI's xpm * time semantics.

        Args:
            max_minutes: Maximum number of minutes to include
            enhanced: If True, include kills, towers, barracks, roshan (20 features)
                     If False, use basic 8 features for backward compatibility
        
        Basic features (8):
            [radiant_gold, radiant_xp, dire_gold, dire_xp,
             gold_diff, xp_diff, radiant_lh, dire_lh]
        
        Enhanced features (20):
            Basic (8) + [radiant_kills, dire_kills, kill_diff,
                        radiant_towers, dire_towers, tower_diff,
                        radiant_barracks, dire_barracks, barracks_diff,
                        radiant_roshan, dire_roshan, roshan_diff]
        """
        num_features = 20 if enhanced else 8
        features = np.zeros((max_minutes, num_features), dtype=np.float32)

        # Basic economy features
        for minute in range(min(self.duration_minutes, max_minutes)):
            radiant_feats = self.get_team_features_at_time(minute, "radiant")
            dire_feats = self.get_team_features_at_time(minute, "dire")

            features[minute, 0] = radiant_feats[0]  # Radiant gold
            features[minute, 1] = radiant_feats[1]  # Radiant XP
            features[minute, 2] = dire_feats[0]  # Dire gold
            features[minute, 3] = dire_feats[1]  # Dire XP
            features[minute, 4] = radiant_feats[4]  # Gold diff
            features[minute, 5] = radiant_feats[5]  # XP diff
            features[minute, 6] = radiant_feats[2]  # Radiant LH
            features[minute, 7] = dire_feats[2]  # Dire LH
        
        if enhanced:
            # Kill features
            kills = self.get_kills_per_minute(max_minutes)
            features[:, 8] = kills[:, 0]   # Radiant kills
            features[:, 9] = kills[:, 1]   # Dire kills
            features[:, 10] = kills[:, 0] - kills[:, 1]  # Kill diff
            
            # Tower features
            towers = self.get_towers_per_minute(max_minutes)
            features[:, 11] = towers[:, 0]  # Radiant towers
            features[:, 12] = towers[:, 1]  # Dire towers
            features[:, 13] = towers[:, 0] - towers[:, 1]  # Tower diff
            
            # Barracks features
            barracks = self.get_barracks_per_minute(max_minutes)
            features[:, 14] = barracks[:, 0]  # Radiant barracks
            features[:, 15] = barracks[:, 1]  # Dire barracks
            features[:, 16] = barracks[:, 0] - barracks[:, 1]  # Barracks diff
            
            # Roshan features
            roshan = self.get_roshan_per_minute(max_minutes)
            features[:, 17] = roshan[:, 0]  # Radiant roshan
            features[:, 18] = roshan[:, 1]  # Dire roshan
            features[:, 19] = roshan[:, 0] - roshan[:, 1]  # Roshan diff

        return features

