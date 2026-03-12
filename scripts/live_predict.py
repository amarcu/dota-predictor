#!/usr/bin/env python3
"""
Real-time Dota 2 match prediction server.

This script starts a GSI (Game State Integration) server that receives
live game data from Dota 2 and provides win probability predictions.

Setup:
1. Copy the generated GSI config to your Dota 2 folder:
   Steam/steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration/

2. Start this server before launching Dota 2

3. Spectate a match or play a game - predictions will appear in console

Usage:
    python scripts/live_predict.py --model models/checkpoints/model.pt
    
Options:
    --model: Path to trained model checkpoint
    --port: Port to run server on (default: 3000)
    --generate-config: Generate GSI config file and exit
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dota_predictor.inference.predictor import LivePredictor
from dota_predictor.inference.gsi_server import GSIServer, create_gsi_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time Dota 2 match prediction server"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/model.pt",
        help="Path to trained model checkpoint (enhanced features + hero embeddings)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to run GSI server on",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default="dota_predictor_secret",
        help="Auth token for GSI security",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate GSI config file and print instructions",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a demo with simulated game data",
    )
    args = parser.parse_args()
    
    # Generate config and exit
    if args.generate_config:
        config_path = "gamestate_integration_predictor.cfg"
        create_gsi_config(config_path, args.port, args.auth_token)
        
        print("=" * 70)
        print("GSI CONFIG GENERATED".center(70))
        print("=" * 70)
        print()
        print(f"Created: {config_path}")
        print()
        print("To enable real-time predictions:")
        print()
        print("1. Copy this file to your Dota 2 installation:")
        print("   Steam/steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration/")
        print()
        print("2. Start the prediction server:")
        print(f"   python scripts/live_predict.py --port {args.port}")
        print()
        print("3. Launch Dota 2 and spectate a match")
        print()
        print("=" * 70)
        return
    
    # Demo mode - simulate a game
    if args.demo:
        run_demo(args.model)
        return
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print()
        print("Train a model first:")
        print("  python scripts/fetch_data.py --count 500")
        print("  python scripts/process_data.py --db --enhanced-features")
        print("  python scripts/train.py")
        sys.exit(1)
    
    # CALIBRATION DISABLED FOR LIVE PREDICTION
    # The raw model predictions were closer to Dota+ than calibrated ones.
    # Calibration is useful for evaluation metrics, not live prediction.
    calibrator_arg = None
    
    # Initialize predictor (no normalization — model was trained on raw features)
    print("Loading model...")
    predictor = LivePredictor(
        model_path=args.model,
        normalization_path=None,
        calibrator_path=calibrator_arg,
    )
    print(f"Model loaded. Using device: {predictor.device}")
    print("ℹ️  Using RAW model predictions (no calibration)")
    
    # Start GSI server
    server = GSIServer(
        predictor=predictor,
        port=args.port,
        auth_token=args.auth_token,
        verbose=True,
    )
    
    server.start(blocking=True)


def run_demo(model_path: str) -> None:
    """Run a demo with simulated game data."""
    import time
    
    print("=" * 70)
    print("DEMO MODE - Simulating a Dota 2 Match".center(70))
    print("=" * 70)
    print()
    
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model not found at {model_path}")
        print("Train a model first with: python scripts/train.py")
        sys.exit(1)
    
    print("Loading model...")
    # No normalization — model was trained on raw features
    predictor = LivePredictor(
        model_path=model_path,
        normalization_path=None,
        calibrator_path=None,
    )
    print(f"Model loaded. Device: {predictor.device}")
    print("ℹ️  Using RAW model predictions")
    print()
    
    print("Scenario: Radiant leads early, Dire makes comeback mid-game")
    print()
    
    # Simulate 30 minutes of gameplay
    # Scenario: Radiant gets early lead, Dire comes back mid-game
    print("Simulating match progression...")
    print()
    
    # Simulate every minute for proper time-series
    # Values based on actual match data: ~2k gold/min/team, ~2k xp/min/team
    simulated_data = []
    for minute in range(31):
        # Base values grow over time (realistic values based on training data)
        base_gold = 600 + minute * 2000  # ~400 gpm per player × 5
        base_xp = minute * 2000  # ~400 xpm per player × 5
        base_lh = minute * 10  # ~2 lh/min per player × 5
        
        # Scenario: Radiant ahead early, Dire comeback mid-game
        if minute <= 15:
            # Radiant building advantage
            advantage = minute * 300  # Growing radiant lead (+4.5k at min 15)
        else:
            # Dire comeback (team fights going their way)
            advantage = 4500 - (minute - 15) * 600  # Dire takes over
        
        r_gold = base_gold + advantage
        d_gold = base_gold - advantage
        r_xp = base_xp + advantage // 2
        d_xp = base_xp - advantage // 2
        r_lh = base_lh + (5 if advantage > 0 else -5)
        d_lh = base_lh + (-5 if advantage > 0 else 5)
        
        simulated_data.append((minute, r_gold, r_xp, d_gold, d_xp, r_lh, d_lh))
    
    for minute, r_gold, r_xp, d_gold, d_xp, r_lh, d_lh in simulated_data:
        predictor.update(
            game_time=minute * 60,
            radiant_gold=r_gold,
            radiant_xp=r_xp,
            dire_gold=d_gold,
            dire_xp=d_xp,
            radiant_lh=r_lh,
            dire_lh=d_lh,
        )
        
        summary = predictor.get_prediction_summary()
        prob = summary["radiant_win_probability"]
        gold_diff = summary["gold_advantage"]
        
        # Visualization
        bar_len = 40
        radiant_bars = int(prob * bar_len)
        
        if prob > 0.55:
            indicator = "🟢 Radiant"
        elif prob < 0.45:
            indicator = "🔴 Dire"
        else:
            indicator = "⚪ Even"
        
        print(f"Minute {minute:02d} | Gold: {gold_diff:+8,.0f} | "
              f"Radiant [{('█' * radiant_bars).ljust(bar_len, '░')}] Dire | "
              f"{indicator} {prob:.1%}")
        
        time.sleep(0.5)  # Pause for readability
    
    print()
    print("=" * 70)
    print("Demo complete! The model predicted Dire's comeback.")
    print("=" * 70)


if __name__ == "__main__":
    main()

