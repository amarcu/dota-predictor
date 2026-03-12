"""
Polymarket integration module.

Provides read-only access to Polymarket prediction markets for Dota 2,
and utilities to cross-reference live markets with OpenDota match data.

Classes:
    PolymarketClient - Gamma/CLOB API client for fetching markets and odds
    MatchLinker - Links Polymarket events to live Dota 2 matches
"""

from dota_predictor.polymarket.polymarket import PolymarketClient
from dota_predictor.polymarket.match_linker import MatchLinker

__all__ = [
    "PolymarketClient",
    "MatchLinker",
]
