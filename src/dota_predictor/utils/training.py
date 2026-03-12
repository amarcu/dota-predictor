"""Training utilities for the LSTM model."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Training utility for the prediction models.

    Handles:
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str | None = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: The model to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            device: Device to train on (auto-detected if None)
        """
        self.model = model

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = nn.BCELoss(reduction="none")  # Per-element for masking

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        show_progress: bool = True,
    ) -> tuple[float, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        iterator = tqdm(dataloader, desc="Training") if show_progress else dataloader

        for batch in iterator:
            features = batch["features"].to(self.device)
            heroes = batch["heroes"].to(self.device)
            labels = batch["label"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            
            # Train with sequence prediction - predict at EACH timestep
            # This teaches the model to use current game state, not just final state
            seq_predictions = self.model(features, heroes, return_sequence=True)
            # seq_predictions: (batch, seq_len)
            
            # Expand labels to match sequence: same label at each timestep
            seq_labels = labels.unsqueeze(1).expand_as(seq_predictions)
            
            # Compute loss only at valid timesteps (where mask = 1)
            loss = self.criterion(seq_predictions, seq_labels)
            loss = (loss * mask).sum() / mask.sum()  # Average over valid timesteps
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(labels)
            # For accuracy, use the last valid timestep prediction
            durations = batch["duration"].to(self.device)
            final_preds = torch.stack([
                seq_predictions[i, min(d-1, 59)] for i, d in enumerate(durations)
            ])
            predictions_binary = (final_preds > 0.5).float()
            correct += (predictions_binary == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        show_progress: bool = False,
    ) -> tuple[float, float]:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        iterator = tqdm(dataloader, desc="Validating") if show_progress else dataloader
        criterion = nn.BCELoss(reduction="none")  # Per-element loss for masking

        for batch in iterator:
            features = batch["features"].to(self.device)
            heroes = batch["heroes"].to(self.device)
            labels = batch["label"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Use sequence prediction for validation too
            seq_predictions = self.model(features, heroes, return_sequence=True)
            seq_labels = labels.unsqueeze(1).expand_as(seq_predictions)
            
            loss = criterion(seq_predictions, seq_labels)
            loss = (loss * mask).sum() / mask.sum()

            total_loss += loss.item() * len(labels)
            
            # Accuracy at last valid timestep
            durations = batch["duration"].to(self.device)
            final_preds = torch.stack([
                seq_predictions[i, min(d-1, 59)] for i, d in enumerate(durations)
            ])
            predictions_binary = (final_preds > 0.5).float()
            correct += (predictions_binary == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Epochs without improvement before stopping
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        best_val_loss = float("inf")
        patience_counter = 0

        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Logging
            print(  # noqa: T201
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if checkpoint_dir:
                    self.save_checkpoint(
                        checkpoint_path / "model.pt",
                        epoch=epoch,
                        val_loss=val_loss,
                    )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")  # noqa: T201
                    break

        return self.history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        **extra_info: Any,
    ) -> None:
        """Save a model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            **extra_info,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        return checkpoint

