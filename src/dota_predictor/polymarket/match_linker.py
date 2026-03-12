"""
Match linker to connect Polymarket markets to live Dota 2 games.

This module links:
- Polymarket event (by team names) → OpenDota live game (match ID)
- OpenDota match ID → Dota 2 spectate command

Usage:
    from dota_predictor.polymarket.match_linker import MatchLinker
    
    linker = MatchLinker()
    
    # Find a live game matching a Polymarket event
    match = linker.find_live_match("Team Spirit", "Team Yandex")
    
    if match:
        print(f"Match ID: {match['match_id']}")
        print(f"Watch command: {linker.get_watch_command(match)}")
"""

from __future__ import annotations

import os
import re
import subprocess
import requests
from dataclasses import dataclass
from typing import Any


@dataclass
class LiveMatch:
    """Represents a live Dota 2 match."""
    
    match_id: int
    server_steam_id: str
    radiant_team: str
    dire_team: str
    league_name: str | None
    league_id: int | None
    spectators: int
    game_time: int  # seconds
    radiant_score: int
    dire_score: int
    game_number: int | None = None  # Which game in the series (1, 2, 3, etc.)
    series_id: int | None = None
    
    @property
    def teams(self) -> tuple[str, str]:
        return (self.radiant_team, self.dire_team)
    
    def __repr__(self) -> str:
        game_str = f" Game {self.game_number}" if self.game_number else ""
        return f"LiveMatch({self.match_id}: {self.radiant_team} vs {self.dire_team}{game_str})"


@dataclass
class SeriesInfo:
    """Information about a series of games between two teams."""
    
    team1: str
    team2: str
    league_id: int | None
    games: list[LiveMatch]  # Sorted by match_id (chronological)
    
    @property
    def current_game_number(self) -> int:
        """Get the current game number in the series (1-indexed)."""
        return len(self.games)
    
    def get_game(self, game_number: int) -> LiveMatch | None:
        """Get a specific game by number (1-indexed)."""
        if 1 <= game_number <= len(self.games):
            return self.games[game_number - 1]
        return None


class MatchLinker:
    """Links Polymarket markets to live Dota 2 games."""
    
    OPENDOTA_LIVE_URL = "https://api.opendota.com/api/live"
    OPENDOTA_PRO_MATCHES_URL = "https://api.opendota.com/api/proMatches"
    
    def __init__(self):
        self._live_cache: list[LiveMatch] = []
        self._cache_time: float = 0
        self._series_cache: dict[str, SeriesInfo] = {}
        # match_id -> (game_time, timestamp_when_seen) for staleness detection
        self._game_time_history: dict[int, tuple[int, float]] = {}
        
        # Load OpenDota API key from environment
        self._api_key = os.getenv("OPENDOTA_API_KEY")
    
    def _get_params(self) -> dict:
        """Get request params including API key if available."""
        if self._api_key:
            return {"api_key": self._api_key}
        return {}
    
    def get_live_matches(self, refresh: bool = False) -> list[LiveMatch]:
        """Get all live matches with team info (pro games)."""
        import time
        
        # Cache for 10 seconds (reduced from 30 for fresher data)
        if not refresh and self._live_cache and time.time() - self._cache_time < 10:
            return self._live_cache
        
        try:
            resp = requests.get(self.OPENDOTA_LIVE_URL, params=self._get_params(), timeout=10)
            resp.raise_for_status()
            games = resp.json()
        except Exception as e:
            print(f"Error fetching live games: {e}")
            return self._live_cache if self._live_cache else []
        
        matches = []
        for g in games:
            # Only include games with team info
            radiant = g.get("team_name_radiant", "")
            dire = g.get("team_name_dire", "")
            
            if not radiant and not dire and not g.get("league_id"):
                continue
            
            match_id = g.get("match_id", 0)
            game_time = g.get("game_time", 0)
            now = time.time()
            
            # Filter out likely stale matches:
            # - Game time > 90 minutes is suspicious (most pro games end by then)
            # - Game time hasn't changed for > 2 minutes (game probably ended)
            is_stale = False
            if game_time > 5400:  # > 90 minutes
                is_stale = True
            elif match_id in self._game_time_history:
                last_time, last_seen = self._game_time_history[match_id]
                if last_time == game_time:
                    # Same game_time - check how long it's been the same
                    time_unchanged_for = now - last_seen
                    if time_unchanged_for > 120:  # > 2 minutes with no change
                        is_stale = True
                    # Don't update timestamp if game_time hasn't changed
                else:
                    # Game time changed, update history
                    self._game_time_history[match_id] = (game_time, now)
            else:
                # First time seeing this match
                self._game_time_history[match_id] = (game_time, now)
            
            if is_stale:
                continue  # Skip stale matches
            
            matches.append(LiveMatch(
                match_id=match_id,
                server_steam_id=str(g.get("server_steam_id", "")),
                radiant_team=radiant or "Unknown",
                dire_team=dire or "Unknown",
                league_name=g.get("league_name"),
                league_id=g.get("league_id"),
                spectators=g.get("spectators", 0),
                game_time=game_time,
                radiant_score=g.get("radiant_score", 0),
                dire_score=g.get("dire_score", 0),
            ))
        
        self._live_cache = matches
        self._cache_time = time.time()
        
        # Update series info
        self._update_series_info(matches)
        
        return matches
    
    def _update_series_info(self, live_matches: list[LiveMatch]) -> None:
        """Update series information based on live and recent matches."""
        # Get recent pro matches to determine game number in series
        try:
            resp = requests.get(self.OPENDOTA_PRO_MATCHES_URL, params=self._get_params(), timeout=10)
            resp.raise_for_status()
            recent_matches = resp.json()
        except Exception:
            recent_matches = []
        
        # Group matches by teams + league
        series_groups: dict[str, list[dict]] = {}
        
        for m in recent_matches:
            radiant = m.get("radiant_name", "")
            dire = m.get("dire_name", "")
            league_id = m.get("league_id")
            
            if not radiant or not dire:
                continue
            
            # Create normalized key (alphabetical order)
            teams = tuple(sorted([self._normalize_team_name(radiant), 
                                   self._normalize_team_name(dire)]))
            key = f"{teams[0]}|{teams[1]}|{league_id}"
            
            if key not in series_groups:
                series_groups[key] = []
            series_groups[key].append(m)
        
        # Determine game numbers for live matches
        for live_match in live_matches:
            teams = tuple(sorted([self._normalize_team_name(live_match.radiant_team),
                                   self._normalize_team_name(live_match.dire_team)]))
            key = f"{teams[0]}|{teams[1]}|{live_match.league_id}"
            
            if key in series_groups:
                # Sort by match_id (chronological)
                series_matches = sorted(series_groups[key], key=lambda x: x.get("match_id", 0))
                
                # Find position of current match
                completed_games = len(series_matches)
                
                # If current match is in the list, count how many before it
                for i, m in enumerate(series_matches):
                    if m.get("match_id") == live_match.match_id:
                        live_match.game_number = i + 1
                        break
                else:
                    # Current match is not in recent (it's live), so it's the next game
                    live_match.game_number = completed_games + 1
                
                # Cache series info
                self._series_cache[key] = SeriesInfo(
                    team1=teams[0],
                    team2=teams[1],
                    league_id=live_match.league_id,
                    games=[live_match],  # Will be updated with full history
                )
    
    def find_live_match(
        self,
        team1: str,
        team2: str,
        game_number: int | None = None,
        fuzzy: bool = True,
    ) -> LiveMatch | None:
        """
        Find a live match matching the given teams and optionally game number.
        
        Args:
            team1: First team name
            team2: Second team name
            game_number: Specific game in series (1, 2, 3...) or None for any
            fuzzy: Use fuzzy matching (default True)
            
        Returns:
            LiveMatch if found, None otherwise
        """
        matches = self.get_live_matches(refresh=True)
        
        # Find all matches between these teams
        team_matches = []
        for match in matches:
            if self._teams_match(team1, team2, match.radiant_team, match.dire_team, fuzzy):
                team_matches.append(match)
        
        if not team_matches:
            return None
        
        # If no specific game requested, return the first (or only) live match
        if game_number is None:
            return team_matches[0]
        
        # Find match with the specific game number
        for match in team_matches:
            if match.game_number == game_number:
                return match
        
        # If we can't find the exact game, check if the live game IS that game number
        # This happens when we're watching the current game in progress
        if len(team_matches) == 1:
            match = team_matches[0]
            if match.game_number is None:
                # Assume the single live match is the game we're looking for
                # Log a warning though
                print(f"⚠️  Could not verify game number. Assuming live match is Game {game_number}")
                match.game_number = game_number
                return match
            elif match.game_number == game_number:
                return match
            else:
                print(f"⚠️  Game number mismatch! Polymarket: Game {game_number}, "
                      f"Live: Game {match.game_number}")
                return None
        
        return None
    
    def get_series_matches(
        self,
        team1: str,
        team2: str,
        include_completed: bool = True,
    ) -> list[LiveMatch]:
        """
        Get all matches in a series between two teams.
        
        Args:
            team1: First team name
            team2: Second team name
            include_completed: Include completed games from the series
            
        Returns:
            List of matches sorted by game number
        """
        matches = []
        
        # Get live matches
        live = self.get_live_matches(refresh=True)
        for m in live:
            if self._teams_match(team1, team2, m.radiant_team, m.dire_team):
                matches.append(m)
        
        if not include_completed:
            return matches
        
        # Get recent completed matches
        try:
            resp = requests.get(self.OPENDOTA_PRO_MATCHES_URL, params=self._get_params(), timeout=10)
            resp.raise_for_status()
            recent = resp.json()
            
            for m in recent:
                radiant = m.get("radiant_name", "")
                dire = m.get("dire_name", "")
                
                if self._teams_match(team1, team2, radiant, dire):
                    # Check if this match is already in our list
                    match_id = m.get("match_id")
                    if not any(lm.match_id == match_id for lm in matches):
                        matches.append(LiveMatch(
                            match_id=match_id,
                            server_steam_id="",  # Completed, no server
                            radiant_team=radiant,
                            dire_team=dire,
                            league_name=m.get("league_name"),
                            league_id=m.get("league_id"),
                            spectators=0,
                            game_time=m.get("duration", 0),
                            radiant_score=0,
                            dire_score=0,
                        ))
        except Exception:
            pass
        
        # Sort by match_id (chronological)
        matches.sort(key=lambda x: x.match_id)
        
        # Assign game numbers
        for i, m in enumerate(matches):
            m.game_number = i + 1
        
        return matches
    
    def find_match_for_polymarket(
        self, 
        event: dict | None = None,
        market: dict | None = None,
    ) -> LiveMatch | None:
        """
        Find a live match for a Polymarket event or market.
        
        Args:
            event: Polymarket event dict (from Gamma API)
            market: Specific market dict (for game-specific markets)
            
        Returns:
            LiveMatch if found
        """
        # Get title from market or event
        if market:
            title = market.get("question", "")
        elif event:
            title = event.get("title", "")
        else:
            return None
        
        # Extract team names and game number
        teams = self._extract_teams_from_title(title)
        game_number = self._extract_game_number(title)
        
        if teams:
            return self.find_live_match(teams[0], teams[1], game_number=game_number)
        
        return None
    
    def _extract_teams_from_title(self, title: str) -> tuple[str, str] | None:
        """Extract team names from a Polymarket title."""
        # Pattern: "Dota 2: Team A vs Team B (BO5)" or "Dota 2: Team A vs Team B - Game 1 Winner"
        # Remove game number suffix first
        cleaned = re.sub(r"\s*-\s*Game\s+\d+\s+Winner", "", title, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*\(BO\d+\)", "", cleaned)
        
        pattern = r"(?:Dota 2:\s*)?(.+?)\s+vs\.?\s+(.+?)$"
        match = re.match(pattern, cleaned.strip(), re.IGNORECASE)
        
        if match:
            return (match.group(1).strip(), match.group(2).strip())
        
        return None
    
    def _extract_game_number(self, title: str) -> int | None:
        """Extract game number from a market title.
        
        Examples:
            "Dota 2: Team A vs Team B - Game 1 Winner" -> 1
            "Dota 2: Team A vs Team B (BO5)" -> None (series market)
        """
        # Pattern: "Game X Winner" or "Game X"
        match = re.search(r"Game\s+(\d+)", title, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _teams_match(
        self,
        search1: str,
        search2: str,
        actual1: str,
        actual2: str,
        fuzzy: bool = True,
    ) -> bool:
        """Check if two pairs of team names match."""
        if fuzzy:
            search1 = self._normalize_team_name(search1)
            search2 = self._normalize_team_name(search2)
            actual1 = self._normalize_team_name(actual1)
            actual2 = self._normalize_team_name(actual2)
        
        # Check both orderings
        return (
            (search1 in actual1 or actual1 in search1) and
            (search2 in actual2 or actual2 in search2)
        ) or (
            (search1 in actual2 or actual2 in search1) and
            (search2 in actual1 or actual1 in search2)
        )
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for matching."""
        # Remove common prefixes/suffixes
        name = name.lower().strip()
        name = re.sub(r"^team\s+", "", name)  # Remove "Team " prefix
        name = re.sub(r"\s+esports?$", "", name)  # Remove "Esports" suffix
        name = re.sub(r"\s+gaming$", "", name)  # Remove "Gaming" suffix
        return name
    
    def get_watch_command(self, match: LiveMatch) -> str:
        """
        Get the Dota 2 console command to watch a match.
        
        Args:
            match: LiveMatch object
            
        Returns:
            Console command string
        """
        # Dota 2 console command to spectate
        return f"dota_spectator_auto_spectate_games 1; watch_server {match.server_steam_id}"
    
    def get_steam_url(self, match: LiveMatch) -> str:
        """
        Get Steam protocol URL to open Dota 2 and watch.
        
        Note: This may not work for all matches as it requires
        direct server connection support.
        """
        return f"steam://run/570//+watch_server {match.server_steam_id}"
    
    def open_in_dota(self, match: LiveMatch) -> bool:
        """
        Try to open Dota 2 and spectate the match.
        
        Returns:
            True if command was executed
        """
        import platform
        
        steam_url = self.get_steam_url(match)
        
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", steam_url], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["start", steam_url], shell=True, check=True)
            else:  # Linux
                subprocess.run(["xdg-open", steam_url], check=True)
            return True
        except Exception as e:
            print(f"Error opening Dota 2: {e}")
            print(f"Manual command: {self.get_watch_command(match)}")
            return False


def link_polymarket_to_dota(
    event_slug: str,
    market_question: str | None = None,
) -> dict[str, Any] | None:
    """
    Convenience function to link a Polymarket event/market to a live Dota 2 match.
    
    Args:
        event_slug: Polymarket event slug (e.g., "dota2-ty-ts8-2025-12-21")
        market_question: Optional market question to match specific game
                        (e.g., "Game 1 Winner" for Game 1)
        
    Returns:
        Dict with Polymarket and Dota 2 info, or None if not found
    """
    from dota_predictor.polymarket.polymarket import PolymarketClient
    
    # Get Polymarket event
    client = PolymarketClient.from_env()
    event = client.get_event_by_slug(event_slug)
    
    if not event:
        print(f"Polymarket event not found: {event_slug}")
        return None
    
    linker = MatchLinker()
    
    # If a specific market question is given, find that market
    target_market = None
    if market_question:
        for m in event.get("markets", []):
            q = m.get("question", "")
            if market_question.lower() in q.lower():
                target_market = m
                break
    
    # Find matching live game
    if target_market:
        match = linker.find_match_for_polymarket(market=target_market)
        game_number = linker._extract_game_number(target_market.get("question", ""))
    else:
        match = linker.find_match_for_polymarket(event=event)
        game_number = None
    
    if not match:
        title = target_market.get("question") if target_market else event.get("title")
        print(f"No live Dota 2 match found for: {title}")
        
        # Show series info if available
        teams = linker._extract_teams_from_title(event.get("title", ""))
        if teams:
            series = linker.get_series_matches(teams[0], teams[1])
            if series:
                print(f"\n📋 Series history ({teams[0]} vs {teams[1]}):")
                for g in series:
                    status = "🔴 LIVE" if g.server_steam_id else "✅ Completed"
                    print(f"   Game {g.game_number}: Match {g.match_id} - {status}")
        
        return None
    
    result = {
        "polymarket": {
            "event_id": event.get("id"),
            "title": event.get("title"),
            "game_id": event.get("gameId"),
            "markets": len(event.get("markets", [])),
        },
        "dota2": {
            "match_id": match.match_id,
            "server_steam_id": match.server_steam_id,
            "radiant": match.radiant_team,
            "dire": match.dire_team,
            "league": match.league_name,
            "spectators": match.spectators,
            "game_time": match.game_time,
            "game_number": match.game_number,
            "score": f"{match.radiant_score} - {match.dire_score}",
        },
        "commands": {
            "console": linker.get_watch_command(match),
            "steam_url": linker.get_steam_url(match),
        },
    }
    
    # Add market-specific info if applicable
    if target_market:
        result["market"] = {
            "question": target_market.get("question"),
            "game_number": game_number,
        }
    
    # Add series info
    teams = linker._extract_teams_from_title(event.get("title", ""))
    if teams:
        series = linker.get_series_matches(teams[0], teams[1])
        if len(series) > 1:
            result["series"] = {
                "total_games": len(series),
                "current_game": match.game_number,
                "games": [
                    {
                        "game_number": g.game_number,
                        "match_id": g.match_id,
                        "status": "live" if g.server_steam_id else "completed",
                    }
                    for g in series
                ],
            }
    
    return result


@dataclass
class MatchableGame:
    """A game that can be both bet on (Polymarket) and watched (Dota 2)."""
    
    # Polymarket data
    pm_title: str
    pm_slug: str
    pm_odds: dict[str, float]  # team -> probability
    pm_volume: float
    pm_start_time: Any  # datetime or None
    pm_event: dict  # Raw event data
    
    # Dota 2 data (optional - may not have a live match)
    dota_match: LiveMatch | None
    
    @property
    def is_market_resolved(self) -> bool:
        """Check if market is resolved (odds at 99%+ for one side)."""
        if not self.pm_odds:
            return False
        odds_values = list(self.pm_odds.values())
        if len(odds_values) >= 2:
            return max(odds_values) >= 0.99
        return False
    
    @property
    def is_live(self) -> bool:
        """Game is live if we have a Dota match AND market is not resolved."""
        if self.is_market_resolved:
            return False
        return self.dota_match is not None and bool(self.dota_match.server_steam_id)
    
    @property
    def watch_command(self) -> str | None:
        """Get the Dota 2 console command to watch this game."""
        if self.dota_match and self.dota_match.server_steam_id:
            return f"watch_server {self.dota_match.server_steam_id}"
        return None


def find_matchable_games(
    limit: int = 30,
    live_only: bool = False,
    skip_resolved: bool = True,
) -> list[MatchableGame]:
    """
    Find games that are on both Polymarket and potentially watchable in Dota 2.
    
    This is the unified function for finding games - use this instead of 
    implementing the logic separately in each script.
    
    Args:
        limit: Maximum number of Polymarket events to check
        live_only: Only return games that are currently live in Dota 2
        skip_resolved: Skip markets where odds are 99%+ (game ended)
    
    Returns:
        List of MatchableGame objects, sorted by live status then start time
    """
    from dota_predictor.polymarket.polymarket import PolymarketClient
    
    client = PolymarketClient.from_env()
    linker = MatchLinker()
    
    # Get data from both platforms
    events = client.get_dota_events(limit=limit)
    
    games = []
    
    for event in events:
        title = event.get("title", "Unknown")
        slug = event.get("slug", "")
        start_time_str = event.get("startTime") or event.get("eventStartTime")
        
        # Parse start time
        start_time = None
        if start_time_str:
            try:
                from datetime import datetime
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            except Exception:
                pass
        
        # Extract teams
        teams = linker._extract_teams_from_title(title)
        if not teams:
            continue
        
        # Get main market odds (skip game-specific and totals)
        markets = event.get("markets", [])
        odds = {}
        volume = 0.0
        for m in markets:
            q = m.get("question", "")
            if "game" not in q.lower() and "total" not in q.lower():
                parsed = client._parse_market(m)
                if parsed:
                    for token in parsed.tokens:
                        outcome = token.get("outcome", "?")
                        price = token.get("price", 0.5)
                        odds[outcome] = float(price)
                    volume = parsed.volume
                break
        
        # Check if market is resolved
        is_resolved = odds and max(odds.values()) >= 0.99
        if skip_resolved and is_resolved:
            continue
        
        # Try to find matching live game
        dota_match = linker.find_live_match(teams[0], teams[1])
        
        game = MatchableGame(
            pm_title=title,
            pm_slug=slug,
            pm_odds=odds,
            pm_volume=volume,
            pm_start_time=start_time,
            pm_event=event,
            dota_match=dota_match,
        )
        
        # Filter if live_only requested
        if live_only and not game.is_live:
            continue
        
        games.append(game)
    
    # Sort: live first, then by start time
    games.sort(key=lambda g: (
        0 if g.is_live else 1,
        g.pm_start_time.timestamp() if g.pm_start_time else float('inf'),
    ))
    
    return games


def find_first_live_game() -> MatchableGame | None:
    """
    Find the first live game that can be bet on and watched.
    
    Returns:
        MatchableGame if found, None otherwise
    """
    games = find_matchable_games(live_only=True, skip_resolved=True)
    return games[0] if games else None

