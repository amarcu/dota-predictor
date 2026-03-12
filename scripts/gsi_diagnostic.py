#!/usr/bin/env python3
"""
GSI Diagnostic Tool - Logs raw GSI values for data validation.

Run this while watching/playing a Dota 2 game to capture raw GSI data.
After the game, compare with OpenDota data to verify alignment.

Usage:
    python scripts/gsi_diagnostic.py --port 3000
"""

import argparse
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


class DiagnosticHandler(BaseHTTPRequestHandler):
    """Handler that logs raw GSI data for analysis."""
    
    log_file = None
    samples = []
    last_minute = -1
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass
    
    def do_POST(self):
        """Handle GSI POST request."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body.decode('utf-8'))
            self.process_gsi_data(data)
        except json.JSONDecodeError:
            pass
        
        self.send_response(200)
        self.end_headers()
    
    def process_gsi_data(self, data: dict):
        """Extract and log relevant fields from GSI data."""
        
        # Get game time
        map_data = data.get("map", {})
        clock_time = map_data.get("clock_time", -1)
        game_time = map_data.get("game_time", -1)
        
        if clock_time < 0:
            return  # Pre-game
        
        current_minute = clock_time // 60
        
        # Only log once per minute (reduce spam)
        if current_minute == DiagnosticHandler.last_minute:
            return
        DiagnosticHandler.last_minute = current_minute
        
        # Extract player data (spectator mode: team2/team3)
        player_data = data.get("player", {})
        hero_data = data.get("hero", {})
        
        team2_players = player_data.get("team2", {})
        team3_players = player_data.get("team3", {})
        team2_heroes = hero_data.get("team2", {})
        team3_heroes = hero_data.get("team3", {})
        
        sample = {
            "timestamp": datetime.now().isoformat(),
            "clock_time": clock_time,
            "game_time": game_time,
            "minute": current_minute,
            "radiant_score": map_data.get("radiant_score", 0),
            "dire_score": map_data.get("dire_score", 0),
            "radiant": {"players": [], "total_gold": 0, "total_xp_from_hero": 0, "total_xpm_estimated": 0, "total_levels": 0},
            "dire": {"players": [], "total_gold": 0, "total_xp_from_hero": 0, "total_xpm_estimated": 0, "total_levels": 0},
        }
        
        # Process Radiant (team2)
        if isinstance(team2_players, dict):
            for player_id, pdata in team2_players.items():
                if not isinstance(pdata, dict):
                    continue
                
                # Get hero data for this player
                hdata = team2_heroes.get(player_id, {}) if isinstance(team2_heroes, dict) else {}
                
                player_info = {
                    "player_id": player_id,
                    # From Player section
                    "gold": pdata.get("gold", 0),
                    "net_worth": pdata.get("net_worth", 0),
                    "gpm": pdata.get("gpm", 0),
                    "xpm": pdata.get("xpm", 0),  # XP per minute rate
                    "last_hits": pdata.get("last_hits", 0),
                    "kills": pdata.get("kills", 0),
                    "deaths": pdata.get("deaths", 0),
                    # From Hero section
                    "hero_experience": hdata.get("experience", 0),  # Actual XP!
                    "hero_level": hdata.get("level", 0),
                    "hero_name": hdata.get("name", ""),
                }
                
                # Calculate XP estimate from XPM
                player_info["xp_estimated_from_xpm"] = int(player_info["xpm"] * (clock_time / 60))
                
                sample["radiant"]["players"].append(player_info)
                sample["radiant"]["total_gold"] += player_info["net_worth"]
                sample["radiant"]["total_xp_from_hero"] += player_info["hero_experience"]
                sample["radiant"]["total_xpm_estimated"] += player_info["xp_estimated_from_xpm"]
                sample["radiant"]["total_levels"] += player_info["hero_level"]
        
        # Process Dire (team3)
        if isinstance(team3_players, dict):
            for player_id, pdata in team3_players.items():
                if not isinstance(pdata, dict):
                    continue
                
                hdata = team3_heroes.get(player_id, {}) if isinstance(team3_heroes, dict) else {}
                
                player_info = {
                    "player_id": player_id,
                    "gold": pdata.get("gold", 0),
                    "net_worth": pdata.get("net_worth", 0),
                    "gpm": pdata.get("gpm", 0),
                    "xpm": pdata.get("xpm", 0),
                    "last_hits": pdata.get("last_hits", 0),
                    "kills": pdata.get("kills", 0),
                    "deaths": pdata.get("deaths", 0),
                    "hero_experience": hdata.get("experience", 0),
                    "hero_level": hdata.get("level", 0),
                    "hero_name": hdata.get("name", ""),
                }
                
                player_info["xp_estimated_from_xpm"] = int(player_info["xpm"] * (clock_time / 60))
                
                sample["dire"]["players"].append(player_info)
                sample["dire"]["total_gold"] += player_info["net_worth"]
                sample["dire"]["total_xp_from_hero"] += player_info["hero_experience"]
                sample["dire"]["total_xpm_estimated"] += player_info["xp_estimated_from_xpm"]
                sample["dire"]["total_levels"] += player_info["hero_level"]
        
        # Calculate diffs
        sample["gold_diff"] = sample["radiant"]["total_gold"] - sample["dire"]["total_gold"]
        sample["xp_diff_hero"] = sample["radiant"]["total_xp_from_hero"] - sample["dire"]["total_xp_from_hero"]
        sample["xp_diff_estimated"] = sample["radiant"]["total_xpm_estimated"] - sample["dire"]["total_xpm_estimated"]
        
        DiagnosticHandler.samples.append(sample)
        
        # Calculate level diff
        sample["level_diff"] = sample["radiant"]["total_levels"] - sample["dire"]["total_levels"]
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"MINUTE {current_minute} | Clock: {clock_time}s | Score: {sample['radiant_score']}-{sample['dire_score']}")
        print(f"{'='*70}")
        print(f"{'Team':<10} {'Gold':>12} {'Levels':>8} {'XP (Hero)':>12} {'XP (XPM est)':>14}")
        print(f"{'-'*60}")
        print(f"{'Radiant':<10} {sample['radiant']['total_gold']:>12,} {sample['radiant']['total_levels']:>8} {sample['radiant']['total_xp_from_hero']:>12,} {sample['radiant']['total_xpm_estimated']:>14,}")
        print(f"{'Dire':<10} {sample['dire']['total_gold']:>12,} {sample['dire']['total_levels']:>8} {sample['dire']['total_xp_from_hero']:>12,} {sample['dire']['total_xpm_estimated']:>14,}")
        print(f"{'-'*60}")
        print(f"{'Diff':<10} {sample['gold_diff']:>+12,} {sample['level_diff']:>+8} {sample['xp_diff_hero']:>+12,} {sample['xp_diff_estimated']:>+14,}")
        
        # Check if Hero.Experience and Hero.Level are available
        rad_hero_xp = sample['radiant']['total_xp_from_hero']
        dire_hero_xp = sample['dire']['total_xp_from_hero']
        rad_levels = sample['radiant']['total_levels']
        dire_levels = sample['dire']['total_levels']
        
        if rad_levels > 0 or dire_levels > 0:
            print(f"\n✅ Hero.Level is available! (Radiant: {rad_levels}, Dire: {dire_levels})")
        else:
            print(f"\n⚠️  Hero.Level is 0 - field may not be available")
        
        if rad_hero_xp == 0 and dire_hero_xp == 0:
            print(f"⚠️  Hero.Experience is 0 - field not available in spectator mode")
        else:
            rad_diff = abs(sample['radiant']['total_xp_from_hero'] - sample['radiant']['total_xpm_estimated'])
            print(f"✅ Hero.Experience available! Diff from estimate: {rad_diff:,}")
        
        # Save to file
        if DiagnosticHandler.log_file:
            with open(DiagnosticHandler.log_file, 'w') as f:
                json.dump(DiagnosticHandler.samples, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="GSI Diagnostic Tool")
    parser.add_argument("--port", type=int, default=3000, help="Port to listen on")
    parser.add_argument("--output", type=str, default="data/gsi_diagnostic.json", 
                        help="Output file for logged data")
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    DiagnosticHandler.log_file = output_path
    
    print(f"{'='*70}")
    print(f"GSI DIAGNOSTIC TOOL")
    print(f"{'='*70}")
    print(f"Listening on port {args.port}")
    print(f"Logging to: {args.output}")
    print(f"\nStart spectating a Dota 2 game to capture data.")
    print(f"Press Ctrl+C to stop and save.\n")
    
    server = HTTPServer(('localhost', args.port), DiagnosticHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n\nStopped. Captured {len(DiagnosticHandler.samples)} minute samples.")
        print(f"Data saved to: {args.output}")
        
        if DiagnosticHandler.samples:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            print(f"Match duration: {DiagnosticHandler.samples[-1]['minute']} minutes")
            print(f"Final score: {DiagnosticHandler.samples[-1]['radiant_score']}-{DiagnosticHandler.samples[-1]['dire_score']}")
            
            # Check if Hero.Experience was available
            has_hero_xp = any(s['radiant']['total_xp_from_hero'] > 0 for s in DiagnosticHandler.samples)
            if has_hero_xp:
                print(f"\n✅ Hero.Experience field is available!")
            else:
                print(f"\n❌ Hero.Experience field was NOT available (all zeros)")


if __name__ == "__main__":
    main()

