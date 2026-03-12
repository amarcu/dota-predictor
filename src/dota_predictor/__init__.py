"""
Dota 2 Match Outcome Predictor

A machine learning system for predicting Dota 2 match outcomes using
time-series data from matches. Based on research from arXiv:2106.01782.
"""

__version__ = "0.1.0"
__author__ = "Alexandru Marcu"

from dota_predictor.api.opendota import OpenDotaClient
from dota_predictor.data.match import Match, MatchPlayer
from dota_predictor.data.dataset import DotaDataset

__all__ = [
    "OpenDotaClient",
    "Match",
    "MatchPlayer",
    "DotaDataset",
]

