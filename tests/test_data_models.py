"""Tests for data models."""

import numpy as np
import pytest

from dota_predictor.data.match import Match, MatchPlayer


class TestMatchPlayer:
    """Tests for MatchPlayer class."""

    def test_from_api_response_minimal(self):
        """Test parsing minimal API response."""
        data = {
            "player_slot": 0,
            "hero_id": 1,
            "kills": 10,
            "deaths": 5,
            "assists": 15,
        }
        player = MatchPlayer.from_api_response(data)

        assert player.player_slot == 0
        assert player.hero_id == 1
        assert player.kills == 10
        assert player.deaths == 5
        assert player.assists == 15
        assert player.is_radiant is True
        assert player.team_slot == 0

    def test_is_radiant(self):
        """Test Radiant vs Dire team detection."""
        radiant_data = {"player_slot": 0, "hero_id": 1}
        dire_data = {"player_slot": 128, "hero_id": 2}

        radiant_player = MatchPlayer.from_api_response(radiant_data)
        dire_player = MatchPlayer.from_api_response(dire_data)

        assert radiant_player.is_radiant is True
        assert dire_player.is_radiant is False

    def test_time_series_features(self):
        """Test time series feature extraction."""
        data = {
            "player_slot": 0,
            "hero_id": 1,
            "gold_t": [500, 1000, 1500, 2000, 2500],
            "xp_t": [0, 100, 250, 400, 600],
            "lh_t": [0, 5, 12, 20, 30],
            "dn_t": [0, 1, 2, 3, 4],
        }
        player = MatchPlayer.from_api_response(data)
        features = player.get_time_series_features(max_minutes=10)

        assert features.shape == (10, 4)
        assert features[0, 0] == 500  # First minute gold
        assert features[4, 1] == 600  # Fifth minute XP
        assert features[5, 0] == 0  # Padded with zeros


class TestMatch:
    """Tests for Match class."""

    @pytest.fixture
    def sample_match_data(self):
        """Create sample match data."""
        players = []
        for i in range(5):  # Radiant
            players.append({
                "player_slot": i,
                "hero_id": i + 1,
                "kills": 5,
                "deaths": 3,
                "assists": 10,
                "gold_t": [500 + i * 100] * 30,
                "xp_t": [100 + i * 50] * 30,
                "lh_t": list(range(30)),
                "dn_t": list(range(0, 30, 2)),
            })
        for i in range(5):  # Dire
            players.append({
                "player_slot": 128 + i,
                "hero_id": 10 + i,
                "kills": 4,
                "deaths": 4,
                "assists": 8,
                "gold_t": [450 + i * 100] * 30,
                "xp_t": [90 + i * 50] * 30,
                "lh_t": list(range(30)),
                "dn_t": list(range(0, 30, 2)),
            })

        return {
            "match_id": 12345,
            "radiant_win": True,
            "duration": 1800,  # 30 minutes
            "start_time": 1700000000,
            "game_mode": 22,
            "lobby_type": 7,
            "players": players,
        }

    def test_from_api_response(self, sample_match_data):
        """Test parsing match from API response."""
        match = Match.from_api_response(sample_match_data)

        assert match.match_id == 12345
        assert match.radiant_win is True
        assert match.duration == 1800
        assert match.duration_minutes == 30
        assert len(match.players) == 10

    def test_team_separation(self, sample_match_data):
        """Test Radiant/Dire team separation."""
        match = Match.from_api_response(sample_match_data)

        assert len(match.radiant_players) == 5
        assert len(match.dire_players) == 5
        assert all(p.is_radiant for p in match.radiant_players)
        assert all(not p.is_radiant for p in match.dire_players)

    def test_hero_lists(self, sample_match_data):
        """Test hero ID extraction."""
        match = Match.from_api_response(sample_match_data)

        assert match.radiant_heroes == [1, 2, 3, 4, 5]
        assert match.dire_heroes == [10, 11, 12, 13, 14]

    def test_has_time_series(self, sample_match_data):
        """Test time series detection."""
        match = Match.from_api_response(sample_match_data)
        assert match.has_time_series is True

        # Test without time series
        for player in sample_match_data["players"]:
            player["gold_t"] = []
        match_no_ts = Match.from_api_response(sample_match_data)
        assert match_no_ts.has_time_series is False

    def test_team_features_at_time(self, sample_match_data):
        """Test team feature extraction."""
        match = Match.from_api_response(sample_match_data)
        features = match.get_team_features_at_time(10, "radiant")

        assert len(features) == 6
        assert features[0] > 0  # Team gold
        assert features[1] > 0  # Team XP

    def test_full_time_series(self, sample_match_data):
        """Test full time series extraction."""
        match = Match.from_api_response(sample_match_data)
        features = match.get_full_time_series(max_minutes=60)

        assert features.shape == (60, 8)
        assert np.sum(features[:30]) > 0  # First 30 minutes have data
        assert np.sum(features[30:]) == 0  # Rest is padded

