#!/usr/bin/env python3
"""
Find games that are available on both Polymarket and Dota 2.

This script cross-references:
1. Polymarket Dota 2 markets (betting available)
2. OpenDota live/recent matches (watching available)

And shows you games where you can both bet AND watch.

Usage:
    python scripts/find_games.py              # Find all matchable games
    python scripts/find_games.py --live       # Only show currently live games
    python scripts/find_games.py --upcoming   # Show upcoming scheduled games
    python scripts/find_games.py --all        # Show everything
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Load .env file for proxy settings
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dota_predictor.polymarket.match_linker import (
    MatchableGame,
    find_matchable_games,
)


def get_status(game: MatchableGame) -> str:
    """Get display status for a game."""
    if game.is_live:
        return "🔴 LIVE"
    elif game.pm_start_time:
        now = datetime.now(timezone.utc)
        if game.pm_start_time.tzinfo is None:
            start = game.pm_start_time.replace(tzinfo=timezone.utc)
        else:
            start = game.pm_start_time
        
        diff_hours = (start - now).total_seconds() / 3600
        if diff_hours < 0:
            return "⏳ STARTED (not found in Dota)"
        elif diff_hours < 1:
            return "⏰ STARTING SOON"
        elif diff_hours < 24:
            return f"📅 In {diff_hours:.1f}h"
        else:
            return f"📅 In {diff_hours/24:.1f}d"
    else:
        return "❓ Unknown"


def print_games(games: list[MatchableGame]) -> None:
    """Print the list of matchable games."""
    
    if not games:
        print("❌ No matchable games found.\n")
        return
    
    live_games = [g for g in games if g.is_live]
    upcoming_games = [g for g in games if not g.is_live]
    
    # Print live games
    if live_games:
        print("=" * 70)
        print("🔴 LIVE GAMES (can bet + watch NOW)")
        print("=" * 70)
        
        for g in live_games:
            print(f"\n  📈 {g.pm_title}")
            print(f"     Status: {get_status(g)}")
            print(f"     Polymarket slug: {g.pm_slug}")
            
            if g.pm_odds:
                odds_str = " | ".join([f"{k}: {v*100:.0f}%" for k, v in g.pm_odds.items()])
                print(f"     Odds: {odds_str}")
            
            if g.dota_match:
                m = g.dota_match
                game_str = f" (Game {m.game_number})" if m.game_number else ""
                print(f"     Dota 2 Match: {m.match_id}{game_str}")
                print(f"     Score: {m.radiant_score} - {m.dire_score} ({m.game_time // 60}m)")
                print(f"     Spectators: {m.spectators}")
                print(f"     Watch: {g.watch_command}")
    
    # Print upcoming games
    if upcoming_games:
        print("\n" + "=" * 70)
        print("📅 UPCOMING GAMES (can bet, watch when live)")
        print("=" * 70)
        
        for g in upcoming_games[:10]:  # Limit to 10
            print(f"\n  📈 {g.pm_title}")
            print(f"     Status: {get_status(g)}")
            print(f"     Polymarket slug: {g.pm_slug}")
            
            if g.pm_odds:
                odds_str = " | ".join([f"{k}: {v*100:.0f}%" for k, v in g.pm_odds.items()])
                print(f"     Odds: {odds_str}")
            
            if g.pm_volume > 0:
                print(f"     Volume: ${g.pm_volume:,.0f}")
        
        if len(upcoming_games) > 10:
            print(f"\n  ... and {len(upcoming_games) - 10} more upcoming games")
    
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find games available on both Polymarket and Dota 2"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Only show currently live games",
    )
    parser.add_argument(
        "--upcoming",
        action="store_true",
        help="Only show upcoming scheduled games",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all games (default)",
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 70)
    print("🎮 DOTA 2 GAME FINDER - Polymarket + OpenDota")
    print("=" * 70)
    print()
    print("🔍 Fetching data from both platforms...\n")
    
    # Use unified function from match_linker
    games = find_matchable_games(
        limit=30,
        live_only=args.live,
        skip_resolved=True,
    )
    
    # Filter for upcoming only if requested
    if args.upcoming:
        games = [g for g in games if not g.is_live]
    
    print_games(games)
    
    # Summary
    live_count = sum(1 for g in games if g.is_live)
    upcoming_count = len(games) - live_count
    
    print("=" * 70)
    print(f"📊 SUMMARY: {live_count} live, {upcoming_count} upcoming")
    print("=" * 70)
    print()
    print("Commands:")
    print("  make find-games                       # Run this script")
    print("  make spectate SLUG=<slug>             # Watch a specific game")
    print("  make live                             # Start GSI predictor")
    print()


if __name__ == "__main__":
    main()

