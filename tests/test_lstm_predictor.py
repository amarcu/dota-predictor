"""Tests for LSTMPredictor and LivePredictor when hero IDs are not available."""

from pathlib import Path

import pytest
import torch

from dota_predictor.inference.predictor import LivePredictor
from dota_predictor.models.lstm import LSTMPredictor

CHECKPOINT = Path(__file__).parent.parent / "models" / "checkpoints" / "model.pt"


def _model() -> LSTMPredictor:
    model = LSTMPredictor(
        input_size=20,
        hidden_size=16,
        num_layers=2,
        num_heroes=145,
        use_hero_embedding=True,
    )
    model.eval()
    return model


def test_sequence_output_without_heroes_equals_padding_heroes():
    model = _model()
    features = torch.randn(2, 60, 20)
    padding = torch.zeros(2, 10, dtype=torch.long)

    with torch.no_grad():
        without = model(features, None, return_sequence=True)
        with_padding = model(features, padding, return_sequence=True)

    assert without.shape == (2, 60)
    assert torch.equal(without, with_padding)


def test_final_output_without_heroes():
    model = _model()
    features = torch.randn(2, 60, 20)

    with torch.no_grad():
        probs = model(features, None)

    assert probs.shape == (2,)
    assert bool(((probs >= 0) & (probs <= 1)).all())


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="shipped checkpoint not present")
def test_live_predictor_predicts_before_heroes_are_known():
    predictor = LivePredictor(CHECKPOINT, device="cpu")
    predictor.update(
        game_time=600,
        radiant_gold=22000,
        radiant_xp=21000,
        dire_gold=18000,
        dire_xp=19000,
        radiant_lh=120,
        dire_lh=100,
    )

    prob = predictor.predict()

    assert 0.0 <= prob <= 1.0
