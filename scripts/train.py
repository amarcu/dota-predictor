#!/usr/bin/env python3
"""
Script to train the Dota 2 match prediction model.

Usage:
    python scripts/train.py --data data/processed --epochs 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dota_predictor.models.lstm import LSTMPredictor
from dota_predictor.utils.config import Config, HIDDEN_SIZE, NUM_LAYERS
from dota_predictor.utils.training import Trainer


def load_processed_data(data_dir: Path) -> tuple:
    """Load pre-processed numpy arrays."""
    train_features = np.load(data_dir / "train_features.npy")
    train_heroes = np.load(data_dir / "train_heroes.npy")
    train_labels = np.load(data_dir / "train_labels.npy")
    train_masks = np.load(data_dir / "train_masks.npy")
    train_durations = np.load(data_dir / "train_durations.npy")

    val_features = np.load(data_dir / "val_features.npy")
    val_heroes = np.load(data_dir / "val_heroes.npy")
    val_labels = np.load(data_dir / "val_labels.npy")
    val_masks = np.load(data_dir / "val_masks.npy")
    val_durations = np.load(data_dir / "val_durations.npy")

    return (
        train_features, train_heroes, train_labels, train_masks, train_durations,
        val_features, val_heroes, val_labels, val_masks, val_durations,
    )


class ProcessedDataset(torch.utils.data.Dataset):
    """Dataset for pre-processed numpy arrays."""

    def __init__(self, features, heroes, labels, masks, durations):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.heroes = torch.tensor(heroes, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.durations = torch.tensor(durations, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "heroes": self.heroes[idx],
            "label": self.labels[idx],
            "mask": self.masks[idx],
            "duration": self.durations[idx],
        }


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Dota 2 predictor")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=HIDDEN_SIZE,
        help=f"LSTM hidden size (default: {HIDDEN_SIZE})",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=NUM_LAYERS,
        help=f"Number of LSTM layers (default: {NUM_LAYERS})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="Directory for saving checkpoints",
    )

    args = parser.parse_args()
    data_dir = Path(args.data)

    # Load config
    config = Config.from_env()
    config.ensure_dirs()

    print("=" * 60)
    print("DOTA 2 WIN PREDICTOR - TRAINING".center(60))
    print("=" * 60)

    # Load processed data
    print("\nLoading processed data...")
    (
        train_features, train_heroes, train_labels, train_masks, train_durations,
        val_features, val_heroes, val_labels, val_masks, val_durations,
    ) = load_processed_data(data_dir)

    # Detect input size and hero ID range from data
    input_size = train_features.shape[2]
    all_heroes = np.concatenate([train_heroes, val_heroes])
    num_heroes = int(all_heroes.max())
    print(f"Feature shape: {train_features.shape}")
    print(f"Input size: {input_size} features")
    print(f"Max hero ID: {num_heroes}")
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")

    # Create datasets
    train_dataset = ProcessedDataset(train_features, train_heroes, train_labels, train_masks, train_durations)
    val_dataset = ProcessedDataset(val_features, val_heroes, val_labels, val_masks, val_durations)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    print(f"\nInitializing model...")
    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.3,
        num_heroes=num_heroes,
        use_hero_embedding=True,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=1e-5,
    )
    print(f"Using device: {trainer.device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING".center(60))
    print("=" * 60 + "\n")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=10,
        checkpoint_dir=str(checkpoint_dir),
    )

    # Final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE".center(60))
    print("=" * 60)
    print(f"\nBest validation accuracy: {max(history['val_acc']):.1%}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"\nModel saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

