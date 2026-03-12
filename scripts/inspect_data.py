#!/usr/bin/env python3
"""
Quick utility to inspect processed .npy files.

Usage:
    python scripts/inspect_data.py                    # Show all files summary
    python scripts/inspect_data.py train_features    # Show specific file details
    python scripts/inspect_data.py --full            # Show more sample data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def inspect_file(filepath: Path, full: bool = False) -> None:
    """Inspect a single .npy file."""
    arr = np.load(filepath)
    
    print(f"\n{'=' * 60}")
    print(f"📄 {filepath.name}")
    print(f"{'=' * 60}")
    print(f"Shape:  {arr.shape}")
    print(f"Dtype:  {arr.dtype}")
    print(f"Size:   {arr.nbytes / 1024:.1f} KB")
    print(f"Min:    {arr.min():.2f}")
    print(f"Max:    {arr.max():.2f}")
    
    if arr.dtype in [np.float32, np.float64]:
        print(f"Mean:   {arr.mean():.2f}")
        print(f"Std:    {arr.std():.2f}")
    
    print(f"\nSample values:")
    
    if arr.ndim == 1:
        n = len(arr) if full else min(10, len(arr))
        print(f"  {arr[:n].tolist()}")
    elif arr.ndim == 2:
        n_rows = arr.shape[0] if full else min(3, arr.shape[0])
        for i in range(n_rows):
            row = arr[i].tolist()
            if len(row) > 10 and not full:
                print(f"  [{i}]: {row[:10]}... ({len(row)} total)")
            else:
                print(f"  [{i}]: {row}")
    elif arr.ndim == 3:
        print(f"  Shape: (matches, minutes, features)")
        print(f"  Match 0, Minute 5:  {arr[0, 5, :].tolist()}")
        print(f"  Match 0, Minute 10: {arr[0, 10, :].tolist()}")
        print(f"  Match 0, Minute 20: {arr[0, 20, :].tolist()}")
        if full:
            print(f"\n  Full match 0:")
            for m in range(min(arr.shape[1], 30)):
                if arr[0, m, :].sum() != 0:
                    print(f"    Minute {m:2}: {arr[0, m, :].tolist()}")


def inspect_all(processed_dir: Path, full: bool = False) -> None:
    """Inspect all processed files."""
    print("=" * 60)
    print("📦 PROCESSED DATA SUMMARY")
    print("=" * 60)
    
    # Summary table
    print(f"\n{'File':<25} {'Shape':<20} {'Dtype':<10} {'Size':<10}")
    print("-" * 65)
    
    for npy_file in sorted(processed_dir.glob("*.npy")):
        arr = np.load(npy_file)
        size_str = f"{arr.nbytes / 1024:.1f} KB"
        print(f"{npy_file.name:<25} {str(arr.shape):<20} {str(arr.dtype):<10} {size_str:<10}")
    
    # Show metadata
    metadata_path = processed_dir / "metadata.json"
    if metadata_path.exists():
        print(f"\n📝 Metadata:")
        with open(metadata_path) as f:
            metadata = json.load(f)
        for key, value in metadata.items():
            if not isinstance(value, list):
                print(f"   {key}: {value}")
        print(f"   feature_names: {metadata.get('feature_names', [])}")
    
    # Show normalization
    norm_path = processed_dir / "normalization.npz"
    if norm_path.exists():
        norm = np.load(norm_path)
        print(f"\n📊 Normalization stats:")
        print(f"   Mean: {norm['mean'].tolist()}")
        print(f"   Std:  {norm['std'].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect processed .npy files")
    parser.add_argument(
        "file",
        nargs="?",
        help="Specific file to inspect (without .npy extension)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show more sample data",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/processed",
        help="Processed data directory",
    )
    
    args = parser.parse_args()
    processed_dir = Path(args.dir)
    
    if not processed_dir.exists():
        print(f"Error: {processed_dir} not found")
        sys.exit(1)
    
    if args.file:
        # Inspect specific file
        filename = args.file if args.file.endswith(".npy") else f"{args.file}.npy"
        filepath = processed_dir / filename
        if not filepath.exists():
            print(f"Error: {filepath} not found")
            print(f"Available files: {[f.name for f in processed_dir.glob('*.npy')]}")
            sys.exit(1)
        inspect_file(filepath, full=args.full)
    else:
        # Inspect all
        inspect_all(processed_dir, full=args.full)


if __name__ == "__main__":
    main()

