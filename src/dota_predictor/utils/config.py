"""
Configuration management for the project.

All model architecture defaults live here as the single source of truth.
Scripts and modules should import these rather than hardcoding values.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ── Model architecture defaults (single source of truth) ──────────────
INPUT_SIZE = 20
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_HEROES = 160
HERO_EMBEDDING_DIM = 32
MAX_MINUTES = 60


@dataclass
class Config:
    """
    Project configuration.

    Loads settings from environment variables and provides defaults.
    """

    # API Keys
    opendota_api_key: str | None = None
    steam_api_key: str | None = None

    # Directories
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    models_dir: Path = field(default_factory=lambda: Path("./models"))

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 50
    max_minutes: int = MAX_MINUTES

    # Model architecture
    input_size: int = INPUT_SIZE
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    num_heroes: int = NUM_HEROES
    hero_embedding_dim: int = HERO_EMBEDDING_DIM

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "Config":
        """
        Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file

        Returns:
            Config instance
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            opendota_api_key=os.getenv("OPENDOTA_API_KEY"),
            steam_api_key=os.getenv("STEAM_API_KEY"),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            models_dir=Path(os.getenv("MODELS_DIR", "./models")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_minutes": self.max_minutes,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_heroes": self.num_heroes,
            "hero_embedding_dim": self.hero_embedding_dim,
        }

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)

