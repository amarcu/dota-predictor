"""
Shared model loading from checkpoint files.

Centralizes the logic for detecting model architecture from saved state_dict
weights, eliminating the need for hardcoded architecture params at load time.
"""

from pathlib import Path

import torch

from dota_predictor.models.lstm import LSTMPredictor
from dota_predictor.utils.config import (
    DROPOUT,
    HERO_EMBEDDING_DIM,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_HEROES,
    NUM_LAYERS,
)


def load_model_from_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[LSTMPredictor, dict]:
    """
    Load an LSTMPredictor from a checkpoint, auto-detecting architecture.

    Inspects the saved state_dict tensors to determine input_size, hidden_size,
    num_layers, hero embedding dimensions, etc. Falls back to defaults from
    config.py if detection fails.

    Args:
        path: Path to checkpoint file (.pt)
        device: Device to load model onto

    Returns:
        Tuple of (model, config_dict) where config_dict contains the detected
        architecture parameters.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    saved_config = checkpoint.get("config", {})

    # Detect input_size from LSTM input weight: shape is (4*hidden, input_size)
    lstm_weight = state_dict.get("lstm.weight_ih_l0")
    input_size = lstm_weight.shape[1] if lstm_weight is not None else saved_config.get("input_size", INPUT_SIZE)

    # Detect hidden_size from LSTM hidden weight: shape is (4*hidden, hidden)
    lstm_hh = state_dict.get("lstm.weight_hh_l0")
    hidden_size = lstm_hh.shape[1] if lstm_hh is not None else saved_config.get("hidden_size", HIDDEN_SIZE)

    # Detect num_layers by counting LSTM weight matrices
    num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l"))
    if num_layers == 0:
        num_layers = saved_config.get("num_layers", NUM_LAYERS)

    # Detect hero embedding
    use_hero_embedding = "hero_embedding.weight" in state_dict
    if use_hero_embedding:
        num_heroes = state_dict["hero_embedding.weight"].shape[0] - 1
        hero_embedding_dim = state_dict["hero_embedding.weight"].shape[1]
    else:
        num_heroes = saved_config.get("num_heroes", NUM_HEROES)
        hero_embedding_dim = saved_config.get("hero_embedding_dim", HERO_EMBEDDING_DIM)

    model = LSTMPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=DROPOUT,
        use_hero_embedding=use_hero_embedding,
        num_heroes=num_heroes,
        hero_embedding_dim=hero_embedding_dim,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    model_config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "use_hero_embedding": use_hero_embedding,
        "num_heroes": num_heroes if use_hero_embedding else None,
        "hero_embedding_dim": hero_embedding_dim if use_hero_embedding else None,
    }

    return model, model_config
