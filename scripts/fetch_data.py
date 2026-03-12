#!/usr/bin/env python3
"""
Script to fetch Dota 2 match data from OpenDota.

This script fetches pro matches with parsed data and saves them
for training the prediction model.

Usage:
    # Fetch 100 matches into a single file
    python scripts/fetch_data.py --count 100 --output data/raw/matches.json

    # Fetch matches as individual files (easier to read)
    python scripts/fetch_data.py --count 30 --output data/raw/matches --individual

    # Fetch with human-readable format
    python scripts/fetch_data.py --count 30 --output data/raw/matches --individual --readable
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dota_predictor.api.opendota import AsyncOpenDotaClient, OpenDotaClient
from dota_predictor.data.database import MatchDatabase
from dota_predictor.data.heroes import get_hero_name
from dota_predictor.data.match import Match
from dota_predictor.utils.config import Config


def fetch_pro_matches(
    client: OpenDotaClient,
    count: int = 100,
) -> list[dict]:
    """
    Fetch pro matches with parsed data (sync version - slower).

    Args:
        client: OpenDota API client
        count: Number of matches to fetch

    Returns:
        List of match data dictionaries
    """
    print(f"Collecting {count} pro match IDs...")
    match_summaries = client.collect_pro_matches(count=count)

    print(f"Fetching detailed data for {len(match_summaries)} matches...")
    match_ids = [m["match_id"] for m in match_summaries]
    matches = client.fetch_matches_batch(match_ids)

    # Filter to only matches with parsed data
    parsed_matches = [m for m in matches if _has_time_series(m)]
    print(f"Found {len(parsed_matches)} matches with parsed time-series data")

    return parsed_matches


async def fetch_pro_matches_async(
    sync_client: OpenDotaClient,
    api_key: str | None,
    count: int = 100,
    max_concurrent: int = 50,
) -> list[dict]:
    """
    Fetch pro matches with parsed data using async for speed.

    Uses sync client for ID collection (fast enough), then async for batch fetch.

    Args:
        sync_client: Sync OpenDota client for ID collection
        api_key: API key for async client
        count: Number of matches to fetch
        max_concurrent: Max concurrent requests (50 recommended with API key)

    Returns:
        List of match data dictionaries
    """
    print(f"Collecting {count} pro match IDs...")
    match_summaries = sync_client.collect_pro_matches(count=count)

    print(f"Fetching detailed data for {len(match_summaries)} matches (async, {max_concurrent} concurrent)...")
    match_ids = [m["match_id"] for m in match_summaries]

    async_client = AsyncOpenDotaClient(api_key=api_key, max_concurrent=max_concurrent)
    try:
        matches = await async_client.fetch_matches_batch(match_ids)
    finally:
        await async_client.close()

    # Filter to only matches with parsed data, excluding errors
    parsed_matches = []
    for m in matches:
        if isinstance(m, dict) and _has_time_series(m):
            parsed_matches.append(m)

    print(f"Found {len(parsed_matches)} matches with parsed time-series data")

    return parsed_matches


def _has_time_series(match_data: dict) -> bool:
    """Check if match has time-series data."""
    players = match_data.get("players", [])
    return any(
        len(p.get("gold_t", [])) > 0
        for p in players
    )


def fetch_parsed_matches(
    client: OpenDotaClient,
    count: int = 100,
) -> list[dict]:
    """
    Fetch recently parsed matches directly.

    This is more efficient as we know these have time-series data.

    Args:
        client: OpenDota API client
        count: Number of matches to fetch

    Returns:
        List of match data dictionaries
    """
    print(f"Collecting {count} parsed match IDs...")

    all_summaries = []
    last_match_id = None

    while len(all_summaries) < count:
        batch = client.get_parsed_matches(less_than_match_id=last_match_id)
        if not batch:
            break
        all_summaries.extend(batch)
        last_match_id = batch[-1]["match_id"]
        print(f"  Collected {len(all_summaries)} match IDs...")

    match_ids = [m["match_id"] for m in all_summaries[:count]]

    print(f"Fetching detailed data for {len(match_ids)} matches...")
    matches = client.fetch_matches_batch(match_ids)

    return [m for m in matches if _has_time_series(m)]


def validate_matches(matches: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Validate match data and filter invalid matches.

    Args:
        matches: List of match data

    Returns:
        Tuple of (valid_matches, error_messages)
    """
    valid = []
    errors = []

    for match in matches:
        try:
            # Try to parse into our Match model
            m = Match.from_api_response(match)

            # Validation checks
            if m.duration_minutes < 10:
                errors.append(f"Match {m.match_id}: Too short ({m.duration_minutes} min)")
                continue

            if len(m.players) != 10:
                errors.append(f"Match {m.match_id}: Wrong player count ({len(m.players)})")
                continue

            if not m.has_time_series:
                errors.append(f"Match {m.match_id}: No time-series data")
                continue

            valid.append(match)

        except Exception as e:
            errors.append(f"Match parsing error: {e}")

    return valid, errors


def match_to_readable(match_data: dict) -> dict:
    """
    Convert raw match data to a human-readable format.

    This makes it easy to understand individual matches by:
    - Adding hero names
    - Organizing players by team
    - Including time-series samples
    - Adding computed fields
    """
    duration_min = match_data["duration"] // 60
    start_time = datetime.fromtimestamp(match_data["start_time"])

    readable = {
        "_description": "Human-readable match data for analysis",
        "match_id": match_data["match_id"],
        "match_url": f"https://www.opendota.com/matches/{match_data['match_id']}",
        "date": start_time.strftime("%Y-%m-%d %H:%M"),
        "duration_minutes": duration_min,
        "winner": "Radiant" if match_data["radiant_win"] else "Dire",
        "score": {
            "radiant": match_data.get("radiant_score", 0),
            "dire": match_data.get("dire_score", 0),
        },
        "radiant_team": [],
        "dire_team": [],
        "time_series": {
            "_description": "Per-minute snapshots of game state",
            "minutes_available": 0,
            "radiant_gold": [],
            "dire_gold": [],
            "gold_advantage": [],
            "radiant_xp": [],
            "dire_xp": [],
            "xp_advantage": [],
        },
    }

    # Process players
    radiant_gold_t = []
    dire_gold_t = []
    radiant_xp_t = []
    dire_xp_t = []

    for p in match_data["players"]:
        is_radiant = p["player_slot"] < 128
        hero_id = p["hero_id"]

        player_info = {
            "hero_id": hero_id,
            "hero_name": get_hero_name(hero_id),
            "position": p["player_slot"] if is_radiant else p["player_slot"] - 128,
            "kills": p.get("kills", 0),
            "deaths": p.get("deaths", 0),
            "assists": p.get("assists", 0),
            "kda": f"{p.get('kills', 0)}/{p.get('deaths', 0)}/{p.get('assists', 0)}",
            "gold_per_min": p.get("gold_per_min", 0),
            "xp_per_min": p.get("xp_per_min", 0),
            "last_hits": p.get("last_hits", 0),
            "denies": p.get("denies", 0),
            "hero_damage": p.get("hero_damage", 0),
            "tower_damage": p.get("tower_damage", 0),
            "level": p.get("level", 1),
            "net_worth": p.get("net_worth", p.get("gold_per_min", 0) * duration_min),
        }

        if is_radiant:
            readable["radiant_team"].append(player_info)
            if p.get("gold_t"):
                radiant_gold_t.append(p["gold_t"])
            if p.get("xp_t"):
                radiant_xp_t.append(p["xp_t"])
        else:
            readable["dire_team"].append(player_info)
            if p.get("gold_t"):
                dire_gold_t.append(p["gold_t"])
            if p.get("xp_t"):
                dire_xp_t.append(p["xp_t"])

    # Compute team time-series
    if radiant_gold_t and dire_gold_t:
        min_len = min(
            min(len(g) for g in radiant_gold_t),
            min(len(g) for g in dire_gold_t),
        )
        readable["time_series"]["minutes_available"] = min_len

        for minute in range(min_len):
            r_gold = sum(g[minute] for g in radiant_gold_t)
            d_gold = sum(g[minute] for g in dire_gold_t)
            readable["time_series"]["radiant_gold"].append(r_gold)
            readable["time_series"]["dire_gold"].append(d_gold)
            readable["time_series"]["gold_advantage"].append(r_gold - d_gold)

        if radiant_xp_t and dire_xp_t:
            for minute in range(min_len):
                r_xp = sum(x[minute] for x in radiant_xp_t if minute < len(x))
                d_xp = sum(x[minute] for x in dire_xp_t if minute < len(x))
                readable["time_series"]["radiant_xp"].append(r_xp)
                readable["time_series"]["dire_xp"].append(d_xp)
                readable["time_series"]["xp_advantage"].append(r_xp - d_xp)

    return readable


def save_individual_matches(
    matches: list[dict],
    output_dir: Path,
    readable: bool = False,
) -> None:
    """
    Save each match as an individual file.

    Args:
        matches: List of match data
        output_dir: Directory to save files
        readable: If True, convert to human-readable format
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for match in matches:
        match_id = match["match_id"]
        winner = "radiant" if match["radiant_win"] else "dire"
        duration = match["duration"] // 60

        # Create descriptive filename
        filename = f"{match_id}_{winner}_win_{duration}min.json"
        filepath = output_dir / filename

        if readable:
            data = match_to_readable(match)
        else:
            data = match

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Saved {len(matches)} individual match files to {output_dir}/")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch Dota 2 match data from OpenDota API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 100 pro matches into single file
  python scripts/fetch_data.py --count 100

  # Fetch 30 matches as individual readable files
  python scripts/fetch_data.py --count 30 --individual --readable

  # Fetch parsed matches (guaranteed to have time-series)
  python scripts/fetch_data.py --count 50 --source parsed
        """,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of matches to fetch (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/matches",
        help="Output path (file for single, directory for --individual)",
    )
    parser.add_argument(
        "--source",
        choices=["pro", "parsed"],
        default="pro",
        help="Data source: 'pro' for pro matches, 'parsed' for recently parsed",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Save each match as a separate file (easier to read)",
    )
    parser.add_argument(
        "--readable",
        action="store_true",
        help="Convert to human-readable format with hero names and computed fields",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenDota API key (or set OPENDOTA_API_KEY env var)",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Save matches to SQLite database (data/matches.db)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/matches.db",
        help="Path to SQLite database (default: data/matches.db)",
    )

    args = parser.parse_args()

    # Load config
    config = Config.from_env()
    config.ensure_dirs()

    # Initialize client
    api_key = args.api_key or config.opendota_api_key
    client = OpenDotaClient(api_key=api_key)

    # Fetch matches (use async for speed with large counts)
    if args.source == "pro":
        if args.count > 100:
            # Use async for large fetches (much faster with proper rate limiting)
            # Concurrency of 20 with 40 req/s rate limit = good throughput
            max_concurrent = 20 if api_key else 5
            matches = asyncio.run(
                fetch_pro_matches_async(client, api_key, count=args.count, max_concurrent=max_concurrent)
            )
        else:
            matches = fetch_pro_matches(client, count=args.count)
    else:
        matches = fetch_parsed_matches(client, count=args.count)

    # Validate
    valid_matches, errors = validate_matches(matches)

    if errors:
        print(f"\n{len(errors)} validation errors:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Save matches
    if args.db:
        # Save to SQLite database (recommended)
        db = MatchDatabase(args.db_path)
        added, skipped = db.add_matches(valid_matches, progress=True)
        stats = db.get_stats()
        print(f"\nSaved to database: {args.db_path}")
        print(f"  Added: {added}, Skipped (duplicates): {skipped}")
        print(f"  Database total: {stats['total_matches']:,} matches")
    elif args.individual:
        # Save as individual files
        output_path = Path(args.output)
        save_individual_matches(
            valid_matches,
            output_dir=output_path,
            readable=args.readable,
        )
    else:
        # Save as single file (legacy)
        output_path = Path(args.output)
        if not str(output_path).endswith(".json"):
            output_path = Path(str(output_path) + ".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(valid_matches, f)

        print(f"\nSaved {len(valid_matches)} matches to {output_path}")

    # Print summary statistics
    if valid_matches:
        durations = [m.get("duration", 0) // 60 for m in valid_matches]
        radiant_wins = sum(1 for m in valid_matches if m.get("radiant_win"))
        print(f"\nSummary:")
        print(f"  Total matches: {len(valid_matches)}")
        print(f"  Average duration: {sum(durations) / len(durations):.1f} minutes")
        print(f"  Radiant win rate: {radiant_wins / len(valid_matches) * 100:.1f}%")


if __name__ == "__main__":
    main()

