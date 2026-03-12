"""Real-time inference module for Dota 2 predictions."""

from dota_predictor.inference.predictor import LivePredictor
from dota_predictor.inference.gsi_server import GSIServer

__all__ = ["LivePredictor", "GSIServer"]

