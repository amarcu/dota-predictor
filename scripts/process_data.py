#!/usr/bin/env python3
"""
Script to process raw match data into training-ready format.

This script transforms raw OpenDota API responses into:
1. Time-series features (numpy arrays)
2. Hero compositions
3. Labels (radiant win/loss)

The processed data is optimized for fast loading during training.

Usage:
    python scripts/process_data.py --input data/raw/matches.json --output data/processed/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dota_predictor.data.database import MatchDatabase
from dota_predictor.data.match import Match

BASIC_FEATURES = [
    "radiant_gold", "radiant_xp", "dire_gold", "dire_xp",
    "gold_diff", "xp_diff", "radiant_lh", "dire_lh",
]
ENHANCED_EXTRAS = [
    "radiant_kills", "dire_kills", "kill_diff",
    "radiant_towers", "dire_towers", "tower_diff",
    "radiant_barracks", "dire_barracks", "barracks_diff",
    "radiant_roshan", "dire_roshan", "roshan_diff",
]


def _get_feature_names(enhanced: bool) -> list[str]:
    if enhanced:
        return BASIC_FEATURES + ENHANCED_EXTRAS
    return BASIC_FEATURES


def process_matches(
    matches: list[dict],
    max_minutes: int = 60,
    enhanced: bool = False,
) -> dict:
    """
    Process raw match data into training-ready arrays.

    Args:
        matches: List of raw match dictionaries
        max_minutes: Maximum sequence length
        enhanced: Use enhanced features (20) instead of basic (8)

    Returns:
        Dictionary with processed numpy arrays and metadata
    """
    # Initialize lists
    match_ids = []
    features_list = []
    heroes_list = []
    labels = []
    durations = []
    valid_masks = []

    for raw_match in matches:
        try:
            match = Match.from_api_response(raw_match)

            # Skip matches without time-series
            if not match.has_time_series:
                continue

            # Extract time-series features: (max_minutes, 8 or 20)
            features = match.get_full_time_series(max_minutes=max_minutes, enhanced=enhanced)

            # Extract hero IDs: [5 radiant + 5 dire]
            heroes = match.radiant_heroes + match.dire_heroes
            if len(heroes) != 10:
                continue

            # Create validity mask for actual match duration
            mask = np.zeros(max_minutes, dtype=np.float32)
            actual_minutes = min(match.duration_minutes, max_minutes)
            mask[:actual_minutes] = 1.0

            # Append to lists
            match_ids.append(match.match_id)
            features_list.append(features)
            heroes_list.append(heroes)
            labels.append(1.0 if match.radiant_win else 0.0)
            durations.append(match.duration_minutes)
            valid_masks.append(mask)

        except Exception as e:
            print(f"Error processing match: {e}")
            continue

    # Convert to numpy arrays
    processed = {
        "match_ids": np.array(match_ids, dtype=np.int64),
        "features": np.array(features_list, dtype=np.float32),
        "heroes": np.array(heroes_list, dtype=np.int32),
        "labels": np.array(labels, dtype=np.float32),
        "durations": np.array(durations, dtype=np.int32),
        "masks": np.array(valid_masks, dtype=np.float32),
    }

    return processed


def compute_normalization_stats(features: np.ndarray) -> dict:
    """
    Compute mean and std for feature normalization.

    Args:
        features: Array of shape (num_matches, max_minutes, num_features)

    Returns:
        Dictionary with mean and std arrays
    """
    # Compute across all matches and timesteps
    # features shape: (N, T, F)
    mean = np.mean(features, axis=(0, 1))
    std = np.std(features, axis=(0, 1))

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    return {
        "mean": mean,
        "std": std,
    }


def create_train_val_split(
    processed: dict,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Split processed data into training and validation sets.

    Args:
        processed: Processed data dictionary
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data)
    """
    np.random.seed(seed)

    n_samples = len(processed["match_ids"])
    indices = np.random.permutation(n_samples)

    n_val = int(n_samples * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_data = {key: arr[train_indices] for key, arr in processed.items()}
    val_data = {key: arr[val_indices] for key, arr in processed.items()}

    return train_data, val_data


def save_processed_data(
    data: dict,
    output_dir: Path,
    prefix: str = "",
) -> None:
    """
    Save processed data as numpy files.

    Args:
        data: Dictionary of numpy arrays
        output_dir: Output directory
        prefix: Filename prefix (e.g., "train_", "val_")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in data.items():
        filepath = output_dir / f"{prefix}{name}.npy"
        np.save(filepath, arr)

    print(f"Saved {len(data)} arrays to {output_dir}/ with prefix '{prefix}'")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process raw match data into training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files in data/processed/:
  train_features.npy   - Time-series features (N, 60, F) where F=8 or 20
  train_heroes.npy     - Hero IDs (N, 10)
  train_labels.npy     - Win labels (N,)
  train_masks.npy      - Valid timestep masks (N, 60)
  train_durations.npy  - Match durations in minutes (N,)
  train_match_ids.npy  - OpenDota match IDs (N,)
  val_*.npy            - Validation set (same structure)
  metadata.json        - Processing metadata

Feature columns:
  0: radiant_gold     - Total Radiant team gold
  1: radiant_xp       - Total Radiant team XP
  2: dire_gold        - Total Dire team gold
  3: dire_xp          - Total Dire team XP
  4: gold_diff        - Radiant - Dire gold (positive = Radiant lead)
  5: xp_diff          - Radiant - Dire XP
  6: radiant_lh       - Total Radiant last hits
  7: dire_lh          - Total Dire last hits
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/matches.json",
        help="Input file or directory with raw matches",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=60,
        help="Maximum sequence length in minutes (default: 60)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)",
    )
    parser.add_argument(
        "--enhanced-features",
        action="store_true",
        help="Use enhanced features (20 instead of 8): adds kills, towers, barracks, roshan",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Load matches from SQLite database instead of JSON file",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/matches.db",
        help="Path to SQLite database (default: data/matches.db)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Load raw matches
    if args.db:
        # Load from SQLite database (recommended)
        db_path = Path(args.db_path)
        print(f"Loading matches from database: {db_path}...")
        db = MatchDatabase(db_path)
        stats = db.get_stats()
        print(f"Database contains {stats['total_matches']:,} matches")
        matches = db.get_all_matches(has_time_series=True)
    else:
        # Load from JSON file (legacy)
        input_path = Path(args.input)
        print(f"Loading raw matches from {input_path}...")

        if input_path.is_file():
            with open(input_path) as f:
                matches = json.load(f)
        elif input_path.is_dir():
            matches = []
            for filepath in input_path.glob("*.json"):
                with open(filepath) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        matches.extend(data)
                    else:
                        matches.append(data)
        else:
            print(f"Error: {input_path} not found")
            sys.exit(1)

    print(f"Loaded {len(matches)} raw matches")

    # Process matches
    feature_type = "enhanced (20 features)" if args.enhanced_features else "basic (8 features)"
    print(f"Processing matches (max_minutes={args.max_minutes}, {feature_type})...")
    processed = process_matches(matches, max_minutes=args.max_minutes, enhanced=args.enhanced_features)

    n_processed = len(processed["match_ids"])
    print(f"Successfully processed {n_processed} matches")

    if n_processed == 0:
        print("No valid matches to process!")
        sys.exit(1)

    # Compute normalization stats
    print("Computing normalization statistics...")
    norm_stats = compute_normalization_stats(processed["features"])

    # Split into train/val
    print(f"Splitting into train/val (ratio={args.val_ratio})...")
    train_data, val_data = create_train_val_split(
        processed,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Save processed data
    print(f"\nSaving processed data to {output_dir}/...")
    save_processed_data(train_data, output_dir, prefix="train_")
    save_processed_data(val_data, output_dir, prefix="val_")

    # Save normalization stats
    norm_path = output_dir / "normalization.npz"
    np.savez(norm_path, **norm_stats)
    print(f"Saved normalization stats to {norm_path}")

    # Save metadata
    source = args.db_path if args.db else str(args.input)
    metadata = {
        "source_file": source,
        "num_raw_matches": len(matches),
        "num_processed": n_processed,
        "num_train": len(train_data["match_ids"]),
        "num_val": len(val_data["match_ids"]),
        "max_minutes": args.max_minutes,
        "feature_names": _get_feature_names(args.enhanced_features),
        "radiant_win_rate_train": float(np.mean(train_data["labels"])),
        "radiant_win_rate_val": float(np.mean(val_data["labels"])),
        "avg_duration_minutes": float(np.mean(processed["durations"])),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"  Training samples:   {metadata['num_train']}")
    print(f"  Validation samples: {metadata['num_val']}")
    n_features = len(_get_feature_names(args.enhanced_features))
    print(f"  Feature shape:      ({args.max_minutes}, {n_features})")
    print(f"  Radiant win rate:   {metadata['radiant_win_rate_train']:.1%} (train)")
    print(f"  Avg duration:       {metadata['avg_duration_minutes']:.1f} min")

    print(f"\nOutput files:")
    for f in sorted(output_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:25} ({size_kb:,.1f} KB)")


if __name__ == "__main__":
    main()

