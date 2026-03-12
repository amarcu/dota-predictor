#!/usr/bin/env python3
"""
Predict outcomes for a specific match using OpenDota API.

This script fetches match data from OpenDota and runs predictions
at multiple time points to show how the model's prediction evolves.

Usage:
    python scripts/predict_match.py --match-id 8610327187
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from dota_predictor.api.opendota import OpenDotaClient
from dota_predictor.data.match import Match
from dota_predictor.data.heroes import get_hero_name
from dota_predictor.models.loader import load_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict match outcome at different time points"
    )
    parser.add_argument(
        "--match-id",
        type=int,
        required=True,
        help="OpenDota match ID to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/model.pt",
        help="Path to trained model",
    )
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print("Loading model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, model_config = load_model_from_checkpoint(model_path, device=device)
    input_size = model_config["input_size"]
    use_hero_embedding = model_config["use_hero_embedding"]
    
    print(f"Model: input_size={model_config['input_size']}, "
          f"hidden={model_config['hidden_size']}, "
          f"layers={model_config['num_layers']}, "
          f"heroes={'yes' if use_hero_embedding else 'no'}")
    
    # Fetch match data
    print(f"\nFetching match {args.match_id} from OpenDota...")
    client = OpenDotaClient()
    
    try:
        match_data = client.get_match(args.match_id)
    except Exception as e:
        print(f"Error fetching match: {e}")
        sys.exit(1)
    
    match = Match.from_api_response(match_data)
    
    if not match.has_time_series:
        print("Error: Match doesn't have time-series data (not parsed)")
        sys.exit(1)
    
    # Display match info
    print("\n" + "=" * 70)
    print("MATCH ANALYSIS".center(70))
    print("=" * 70)
    
    print(f"\nMatch ID: {match.match_id}")
    print(f"Duration: {match.duration_minutes} minutes")
    print(f"Result:   {'Radiant Victory 🟢' if match.radiant_win else 'Dire Victory 🔴'}")
    
    print("\nRadiant Team:")
    for i, player in enumerate(match.players[:5]):
        print(f"  {i+1}. {get_hero_name(player.hero_id)}")
    
    print("\nDire Team:")
    for i, player in enumerate(match.players[5:]):
        print(f"  {i+1}. {get_hero_name(player.hero_id)}")
    
    # Get features (enhanced=True for 20 features, matching training; no normalization)
    max_minutes = 60
    enhanced = input_size > 8
    features = match.get_full_time_series(max_minutes=max_minutes, enhanced=enhanced)
    heroes = np.array([p.hero_id for p in match.players], dtype=np.int32)
    
    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    heroes_tensor = torch.tensor(heroes, dtype=torch.long).unsqueeze(0).to(device) if use_hero_embedding else None
    
    # Predict at multiple time points
    print("\n" + "=" * 70)
    print("PREDICTION EVOLUTION".center(70))
    print("=" * 70)
    print()
    
    time_points = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    time_points = [t for t in time_points if t <= match.duration_minutes]
    
    with torch.no_grad():
        for minute in time_points:
            # Create masked features (only use data up to this minute)
            masked_features = features_tensor.clone()
            masked_features[:, minute:, :] = 0
            
            seq_probs = model(masked_features, heroes_tensor, return_sequence=True)
            prob = seq_probs[0, minute - 1].item()
            
            # Get gold diff at this minute
            gold_diff = match_data.get("radiant_gold_adv", [0] * 60)
            if minute < len(gold_diff):
                gold = gold_diff[minute]
            else:
                gold = 0
            
            # Visualization
            bar_len = 40
            radiant_bars = int(prob * bar_len)
            
            # Color indicator
            if prob > 0.55:
                indicator = "🟢"
            elif prob < 0.45:
                indicator = "🔴"
            else:
                indicator = "⚪"
            
            print(f"Min {minute:02d} | Gold: {gold:+8,} | "
                  f"[{'█' * radiant_bars}{'░' * (bar_len - radiant_bars)}] | "
                  f"{indicator} {prob:.1%}")
    
    # Final prediction vs actual
    print("\n" + "=" * 70)
    
    final_seq = model(features_tensor, heroes_tensor, return_sequence=True)
    last_minute = min(match.duration_minutes - 1, 59)
    final_prob = final_seq[0, last_minute].item()
    predicted_winner = "Radiant" if final_prob > 0.5 else "Dire"
    actual_winner = "Radiant" if match.radiant_win else "Dire"
    correct = predicted_winner == actual_winner
    
    print(f"Final Prediction: {predicted_winner} ({final_prob:.1%})")
    print(f"Actual Result:    {actual_winner}")
    print(f"Correct:          {'✅ Yes!' if correct else '❌ No'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

