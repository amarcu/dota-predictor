#!/usr/bin/env python3
"""
Find a live Dota 2 pro game on Polymarket and launch the GSI predictor.

This script:
1. Finds a live game from Polymarket markets
2. Opens Dota 2 to spectate that game
3. Starts the GSI prediction server
4. Shows live win-probability predictions alongside Polymarket market odds

Usage:
    # Auto-find a live game and start watching
    python scripts/spectate.py

    # Watch a specific game by Polymarket slug
    python scripts/spectate.py --slug dota2-l1ga-vpp-2025-12-22

    # Watch a specific game number in a series
    python scripts/spectate.py --slug dota2-l1ga-vpp-2025-12-22 --game 2

    # Just open Dota 2 (no GSI predictor)
    python scripts/spectate.py --dota-only

    # Just start the GSI predictor (Dota 2 already running)
    python scripts/spectate.py --gsi-only
"""

import argparse
import subprocess
import sys
import time
import platform
from pathlib import Path

# Load .env file for proxy settings
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dota_predictor.polymarket.match_linker import (
    MatchableGame,
    link_polymarket_to_dota,
    find_first_live_game,
)


def open_dota_spectate(server_steam_id: str) -> bool:
    """Open Dota 2 and spectate a game."""
    print(f"\n🎮 Opening Dota 2 to spectate...")
    print(f"   Server: {server_steam_id}")
    
    system = platform.system()
    watch_cmd = f"watch_server {server_steam_id}"
    
    # Copy the watch command to clipboard for easy pasting
    try:
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=watch_cmd.encode(), check=True)
            print(f"   📋 Command copied to clipboard!")
        elif system == "Windows":
            subprocess.run(["clip"], input=watch_cmd.encode(), check=True)
            print(f"   📋 Command copied to clipboard!")
    except Exception:
        pass  # Clipboard copy is optional
    
    # Try Steam URL with watch_server command
    # Format: steam://run/570//+watch_server <id>
    steam_url = f"steam://run/570//+watch_server%20{server_steam_id}"
    
    try:
        if system == "Darwin":  # macOS
            subprocess.Popen(["open", steam_url])
        elif system == "Windows":
            subprocess.Popen(["start", steam_url], shell=True)
        else:  # Linux
            subprocess.Popen(["xdg-open", steam_url])
        
        print("   ✅ Dota 2 launched")
        print()
        print("   ┌─────────────────────────────────────────────────────┐")
        print("   │  📺 TO SPECTATE: Open console (~) and paste/type:  │")
        print(f"   │                                                     │")
        print(f"   │  {watch_cmd:<51} │")
        print("   │                                                     │")
        print("   │  (Command already copied to clipboard!)             │")
        print("   └─────────────────────────────────────────────────────┘")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"\n   Manual steps:")
        print(f"   1. Open Dota 2")
        print(f"   2. Open console (~)")
        print(f"   3. Type: {watch_cmd}")
        return False


def start_gsi_server(
    port: int = 3000,
    match_id: str | None = None,
    market_odds: dict[str, float] | None = None,
    spectate_cmd: str | None = None,
):
    """Start the GSI prediction server."""
    print(f"\n📡 Starting GSI prediction server on port {port}...")
    
    # Import here to avoid circular imports
    from dota_predictor.inference.gsi_server import GSIServer
    from dota_predictor.inference.predictor import LivePredictor
    
    model_path = Path("models/checkpoints/model.pt")
    if not model_path.exists():
        print("   ❌ No model found at models/checkpoints/model.pt")
        print("   Train a model first with: make train")
        return None
    
    print(f"   Model: {model_path}")
    if match_id:
        print(f"   Match ID: {match_id}")
    
    # Load predictor
    predictor = LivePredictor(str(model_path))
    
    # Create server with match info
    server = GSIServer(
        predictor,
        port=port,
        match_id=match_id,
        market_odds=market_odds,
        spectate_cmd=spectate_cmd,
    )
    
    return server


def run_combined(
    slug: str | None,
    game: int | None,
    dota_only: bool,
    gsi_only: bool,
    port: int,
    wait_for_game: bool = False,
    retry_interval: int = 300,  # 5 minutes
):
    """Run the combined bot."""
    
    print()
    print("=" * 70)
    print("DOTA 2 SPECTATOR - Starting Up")
    print("=" * 70)
    
    game_info = None
    server_steam_id = None
    
    # Step 1: Find/link the game
    if not gsi_only:
        if slug:
            # Link specific game
            print(f"\n📎 Linking to: {slug}")
            result = link_polymarket_to_dota(slug, market_question=f"Game {game}" if game else None)
            
            if result:
                game_info = result
                server_steam_id = result["dota2"]["server_steam_id"]
                
                print(f"\n✅ Found game:")
                print(f"   {result['polymarket']['title']}")
                print(f"   Match ID: {result['dota2']['match_id']}")
                print(f"   Score: {result['dota2']['score']}")
                
                if result.get("market"):
                    print(f"   Game: {result['market']['game_number']}")
            else:
                print("\n❌ Could not find live match for this slug")
                if not gsi_only:
                    return
        else:
            # Auto-find live game with optional retry
            while True:
                print("\n🔍 Looking for live games...")
                live_game = find_first_live_game()
                
                if live_game:
                    game_info = live_game  # MatchableGame object
                    server_steam_id = live_game.dota_match.server_steam_id
                    
                    print(f"\n✅ Found live game:")
                    print(f"   {live_game.pm_title}")
                    print(f"   Match ID: {live_game.dota_match.match_id}")
                    print(f"   Score: {live_game.dota_match.radiant_score} - {live_game.dota_match.dire_score}")
                    
                    if live_game.pm_odds:
                        odds_str = " | ".join([f"{k}: {v*100:.0f}%" for k, v in live_game.pm_odds.items()])
                        print(f"   Market Odds: {odds_str}")
                    break  # Found a game, exit loop
                else:
                    if wait_for_game:
                        from datetime import datetime
                        next_check = datetime.now().strftime("%H:%M:%S")
                        print(f"\n⏳ No live games found at {next_check}")
                        print(f"   Retrying in {retry_interval // 60} minutes... (Ctrl+C to stop)")
                        try:
                            time.sleep(retry_interval)
                        except KeyboardInterrupt:
                            print("\n\n👋 Stopping...")
                            return
                    else:
                        print("\n⚠️  No live games found matching Polymarket markets")
                        print("   Use --wait to retry every 5 minutes")
                        print("   Starting GSI server anyway - spectate manually")
                        break
    
    # Step 2: Open Dota 2
    if not gsi_only and server_steam_id:
        open_dota_spectate(server_steam_id)
        print("\n⏳ Waiting 5 seconds for Dota 2 to start...")
        time.sleep(5)
    
    # Step 3: Start GSI server
    if not dota_only:
        # Extract match_id and market_odds from game_info
        match_id = None
        market_odds = None
        
        if game_info:
            if isinstance(game_info, MatchableGame):
                # From find_first_live_game result
                if game_info.dota_match:
                    match_id = str(game_info.dota_match.match_id)
                market_odds = game_info.pm_odds
            elif isinstance(game_info, dict):
                # From link_polymarket_to_dota result
                match_id = str(game_info.get("dota2", {}).get("match_id"))
                market_odds = game_info.get("odds")
        
        watch_cmd = f"watch_server {server_steam_id}" if server_steam_id else None
        server = start_gsi_server(port, match_id=match_id, market_odds=market_odds, spectate_cmd=watch_cmd)
        
        if server:
            print("\n" + "=" * 70)
            print("🎯 BOT RUNNING - Predictions will appear when game data is received")
            print("=" * 70)
            
            if market_odds:
                print(f"\n📊 Market Odds to Beat:")
                for team, prob in market_odds.items():
                    print(f"   {team}: {prob*100:.1f}%")
            
            print(f"\n   GSI Server: http://localhost:{port}")
            print("   Press Ctrl+C to stop\n")
            
            try:
                server.start()
            except KeyboardInterrupt:
                print("\n\n👋 Stopping bot...")
                server.stop()
    else:
        print("\n✅ Dota 2 spectate started (GSI server not started)")
        print("   Run 'make live' separately if you want GSI predictions")


def main():
    parser = argparse.ArgumentParser(
        description="Find a live Dota 2 game on Polymarket and start the GSI predictor"
    )
    parser.add_argument(
        "--slug",
        type=str,
        help="Polymarket event slug to watch",
    )
    parser.add_argument(
        "--game",
        type=int,
        help="Game number in series (1, 2, 3...)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="GSI server port (default: 3000)",
    )
    parser.add_argument(
        "--dota-only",
        action="store_true",
        help="Only open Dota 2, don't start GSI server",
    )
    parser.add_argument(
        "--gsi-only",
        action="store_true",
        help="Only start GSI server, don't open Dota 2",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait and retry every 5 minutes if no live games found",
    )
    parser.add_argument(
        "--retry-interval",
        type=int,
        default=300,
        help="Seconds between retries when waiting (default: 300 = 5 min)",
    )
    
    args = parser.parse_args()
    
    run_combined(
        slug=args.slug,
        game=args.game,
        dota_only=args.dota_only,
        gsi_only=args.gsi_only,
        port=args.port,
        wait_for_game=args.wait,
        retry_interval=args.retry_interval,
    )


if __name__ == "__main__":
    main()

