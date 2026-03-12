"""
Polymarket CLOB API client wrapper.

This module provides a simplified interface for interacting with Polymarket's
prediction markets, specifically for reading esports/Dota 2 market data.

Usage:
    from dota_predictor.polymarket.polymarket import PolymarketClient
    
    client = PolymarketClient.from_env()
    
    # Find Dota 2 markets
    markets = client.find_dota_markets()
    
    # Get current odds
    odds = client.get_market_odds(market_id)

Environment variables:
    HTTP_PROXY or HTTPS_PROXY - Optional proxy for API calls (e.g., socks5://127.0.0.1:1080)
    POLYMARKET_PROXY - Alternative proxy setting specific to Polymarket
"""

from __future__ import annotations

import os
import socket
import requests
from dataclasses import dataclass, field
from typing import Any

_DOH_CACHE: dict[str, str] = {}
_original_getaddrinfo = socket.getaddrinfo


def _doh_getaddrinfo(host: str, port: object, *args: object, **kwargs: object) -> list:
    """socket.getaddrinfo replacement that redirects blocked hosts via DoH IP cache."""
    return _original_getaddrinfo(_DOH_CACHE.get(host, host), port, *args, **kwargs)


def _enable_doh_for(hostname: str) -> bool:
    """
    Resolve `hostname` via Cloudflare DNS-over-HTTPS and patch socket.getaddrinfo
    so that all subsequent connections to that hostname go directly to the resolved IP.

    The URL and TLS SNI remain unchanged (correct hostname), so HTTPS certificates
    validate normally. This mirrors how browsers bypass ISP-level DNS blocks.

    Returns True if DoH resolution succeeded, False otherwise.
    """
    if hostname in _DOH_CACHE:
        return True
    try:
        # 1.1.1.1 is a bare IP — no DNS lookup needed to reach Cloudflare's DoH
        resp = requests.get(
            "https://1.1.1.1/dns-query",
            params={"name": hostname, "type": "A"},
            headers={"Accept": "application/dns-json"},
            timeout=5,
        )
        for answer in resp.json().get("Answer", []):
            if answer.get("type") == 1:  # A record
                _DOH_CACHE[hostname] = answer["data"]
                socket.getaddrinfo = _doh_getaddrinfo  # type: ignore[assignment]
                return True
    except Exception:
        pass
    return False


# Try to import the official client
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    HAS_POLYMARKET = True
except ImportError:
    HAS_POLYMARKET = False
    ClobClient = None
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Market:
    """Represents a Polymarket market."""
    
    id: str
    condition_id: str
    question: str
    description: str
    outcomes: list[str] = field(default_factory=list)
    tokens: list[dict] = field(default_factory=list)  # Full token info
    end_date: str | None = None
    volume: float = 0.0
    liquidity: float = 0.0
    slug: str = ""
    game_start_time: str | None = None
    
    @property
    def is_dota(self) -> bool:
        """Check if this is a Dota 2 related market."""
        text = f"{self.question} {self.description}".lower()
        dota_terms = ["dota", "ti ", "the international", "esl one", "dreamleague", 
                      "team spirit", "team liquid", "og ", "nigma", "secret"]
        return any(term in text for term in dota_terms)
    
    @property 
    def token_ids(self) -> dict[str, str]:
        """Get token IDs for each outcome."""
        result = {}
        for token in self.tokens:
            outcome = token.get("outcome", "Unknown")
            token_id = token.get("token_id", "")
            result[outcome] = token_id
        return result
    
    def get_token_id(self, outcome: str) -> str | None:
        """Get token ID for a specific outcome."""
        return self.token_ids.get(outcome)
    
    def __repr__(self) -> str:
        q = self.question[:50] if len(self.question) > 50 else self.question
        return f"Market(id={self.id[:8]}..., question='{q}...')"


@dataclass
class Order:
    """Represents an order on Polymarket."""
    
    id: str
    market_id: str
    token_id: str
    side: str  # "BUY" or "SELL"
    price: float  # 0.01 to 0.99
    size: float  # Amount in USDC
    status: str  # "OPEN", "FILLED", "CANCELLED"
    filled_size: float = 0.0
    
    @property
    def implied_probability(self) -> float:
        """Convert price to probability."""
        return self.price


class PolymarketClient:
    """
    Client for interacting with Polymarket's CLOB API.
    
    Provides simplified methods for:
    - Finding markets (especially Dota 2/esports) via Gamma API
    - Getting current odds
    - Placing and managing bets via CLOB API
    """
    
    # API endpoints
    CLOB_HOST = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"
    
    # Polygon chain ID
    POLYGON_CHAIN_ID = 137
    
    # Dota 2 tag ID (for filtering via Gamma API)
    DOTA2_TAG_ID = "102366"
    
    def __init__(
        self,
        private_key: str | None = None,
        proxy_address: str | None = None,
        signature_type: int = 1,  # 1=Email/Magic, 2=Browser Wallet
        http_proxy: str | None = None,  # e.g., socks5://127.0.0.1:1080
    ):
        """
        Initialize Polymarket client.
        
        Args:
            private_key: Your private key (from https://reveal.magic.link/polymarket)
            proxy_address: Your proxy address (shown on Polymarket profile)
            signature_type: 1 for Email/Magic login, 2 for Browser Wallet
            http_proxy: Optional proxy URL for API calls (e.g., socks5://127.0.0.1:1080)
        """
        self._private_key = private_key
        self._proxy_address = proxy_address
        self._signature_type = signature_type
        self._client = None
        self._markets_cache: dict[str, Market] = {}
        self._api_creds_set = False
        
        # Setup HTTP session with optional proxy or automatic DoH fallback
        gamma_host = self.GAMMA_API.replace("https://", "")
        self._session = requests.Session()
        if http_proxy:
            self._session.proxies = {"http": http_proxy, "https": http_proxy}
            print(f"📡 Polymarket: Using proxy {http_proxy}")
        else:
            # Try system DNS; if blocked, resolve via Cloudflare DoH and patch
            # socket.getaddrinfo so requests uses the resolved IP while keeping
            # SNI and certificate validation against the real hostname.
            try:
                socket.getaddrinfo(gamma_host, 443)
            except socket.gaierror:
                if _enable_doh_for(gamma_host):
                    print(f"📡 Polymarket: System DNS blocked — using DoH fallback ({_DOH_CACHE[gamma_host]})")
        
        # Initialize CLOB client if credentials provided
        if private_key and HAS_POLYMARKET:
            if proxy_address:
                self._client = ClobClient(
                    self.CLOB_HOST,
                    key=private_key,
                    chain_id=self.POLYGON_CHAIN_ID,
                    signature_type=signature_type,
                    funder=proxy_address,
                )
            else:
                # Direct EOA trading
                self._client = ClobClient(
                    self.CLOB_HOST,
                    key=private_key,
                    chain_id=self.POLYGON_CHAIN_ID,
                )
    
    def _ensure_api_creds(self) -> None:
        """Ensure API credentials are set for trading."""
        if self._client and not self._api_creds_set:
            self._client.set_api_creds(self._client.create_or_derive_api_creds())
            self._api_creds_set = True
    
    @classmethod
    def from_env(cls) -> "PolymarketClient":
        """
        Create client from environment variables.
        
        Required for trading:
        - POLYMARKET_PRIVATE_KEY - Your private key
        - POLYMARKET_PROXY_ADDRESS - Your proxy address (optional for EOA)
        - POLYMARKET_SIGNATURE_TYPE - 1 or 2 (default: 1)
        
        Optional proxy for network issues (Dota 2 + Polymarket conflict):
        - POLYMARKET_PROXY - e.g., socks5://127.0.0.1:40000
        - Or use HTTPS_PROXY / HTTP_PROXY
        
        For read-only (fetching markets), no credentials needed.
        """
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS")
        signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))
        
        # Load HTTP proxy (for routing API traffic while keeping Dota 2 direct)
        http_proxy = (
            os.getenv("POLYMARKET_PROXY") or
            os.getenv("HTTPS_PROXY") or
            os.getenv("HTTP_PROXY")
        )
        
        return cls(
            private_key=private_key,
            proxy_address=proxy_address,
            signature_type=signature_type,
            http_proxy=http_proxy,
        )
    
    @property
    def can_trade(self) -> bool:
        """Check if client is configured for trading."""
        return self._client is not None and self._private_key is not None
    
    def get_markets(self, limit: int = 100, closed: bool = False) -> list[Market]:
        """
        Get all active markets from Gamma API.
        
        Args:
            limit: Maximum number of markets to fetch
            closed: Include closed markets
            
        Returns:
            List of Market objects
        """
        markets = []
        offset = 0
        
        while len(markets) < limit:
            batch_limit = min(50, limit - len(markets))
            url = f"{self.GAMMA_API}/markets"
            params = {
                "limit": batch_limit,
                "offset": offset,
                "closed": str(closed).lower(),
            }
            
            try:
                resp = self._session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    break
                
                for m in data:
                    market = self._parse_market(m)
                    if market:
                        markets.append(market)
                        self._markets_cache[market.id] = market
                
                offset += batch_limit
                
                if len(data) < batch_limit:
                    break
                    
            except Exception as e:
                print(f"Error fetching markets: {e}")
                break
        
        return markets
    
    ESPORTS_TAG_ID = "64"
    
    def find_dota_markets(self) -> list[Market]:
        """
        Find all Dota 2 related markets using Gamma API.
        
        Returns:
            List of Dota 2 markets
        """
        # Use the known Dota 2 tag ID
        markets = self.get_markets_by_tag(self.DOTA2_TAG_ID)
        
        if markets:
            return markets
        
        # Fallback: try events endpoint with the dota-2 series
        try:
            events = self.get_dota_events()
            markets = []
            for event in events:
                for m in event.get("markets", []):
                    parsed = self._parse_market(m)
                    if parsed:
                        markets.append(parsed)
            return markets
        except Exception:
            pass
        
        return []
    
    def get_dota_events(self, limit: int = 50) -> list[dict]:
        """
        Get Dota 2 events directly.
        
        Returns:
            List of Dota 2 event dictionaries
        """
        events = []
        offset = 0
        
        while len(events) < limit:
            batch_limit = min(50, limit - len(events))
            
            try:
                resp = self._session.get(
                    f"{self.GAMMA_API}/events",
                    params={
                        "tag_id": self.DOTA2_TAG_ID,
                        "limit": batch_limit,
                        "offset": offset,
                        "closed": "false",
                    },
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    break
                
                events.extend(data)
                offset += batch_limit
                
                if len(data) < batch_limit:
                    break
                    
            except Exception as e:
                print(f"Error fetching Dota 2 events: {e}")
                break
        
        return events
    
    def get_sports_tags(self) -> list[dict]:
        """
        Get all sports tags from Gamma API.
        
        Returns:
            List of sport tag dictionaries with id, label, etc.
        """
        try:
            resp = self._session.get(f"{self.GAMMA_API}/sports", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []
    
    def get_markets_by_tag(self, tag_id: int | str, limit: int = 100) -> list[Market]:
        """
        Get markets filtered by tag ID.
        
        Args:
            tag_id: The tag ID to filter by
            limit: Maximum markets to return
            
        Returns:
            List of Market objects
        """
        markets = []
        offset = 0
        
        while len(markets) < limit:
            batch_limit = min(50, limit - len(markets))
            url = f"{self.GAMMA_API}/markets"
            params = {
                "tag_id": tag_id,
                "limit": batch_limit,
                "offset": offset,
                "closed": "false",
            }
            
            try:
                resp = self._session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    break
                
                for m in data:
                    market = self._parse_market(m)
                    if market:
                        markets.append(market)
                
                offset += batch_limit
                if len(data) < batch_limit:
                    break
                    
            except Exception as e:
                print(f"Error fetching markets by tag: {e}")
                break
        
        return markets
    
    def get_events_by_tag(self, tag_id: int | str, limit: int = 100) -> list[dict]:
        """
        Get events (which contain markets) filtered by tag.
        
        Args:
            tag_id: The tag ID to filter by
            limit: Maximum events to return
            
        Returns:
            List of event dictionaries
        """
        events = []
        offset = 0
        
        while len(events) < limit:
            batch_limit = min(50, limit - len(events))
            url = f"{self.GAMMA_API}/events"
            params = {
                "tag_id": tag_id,
                "limit": batch_limit,
                "offset": offset,
                "closed": "false",
            }
            
            try:
                resp = self._session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    break
                
                events.extend(data)
                offset += batch_limit
                
                if len(data) < batch_limit:
                    break
                    
            except Exception as e:
                print(f"Error fetching events: {e}")
                break
        
        return events
    
    def get_market_by_slug(self, slug: str) -> Market | None:
        """
        Get a specific market by its slug.
        
        The slug is the path segment in the Polymarket URL, e.g.:
        https://polymarket.com/event/fed-decision-in-october
                                    ↑ slug
        
        Args:
            slug: The market slug
            
        Returns:
            Market object or None
        """
        try:
            resp = self._session.get(f"{self.GAMMA_API}/markets/slug/{slug}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_market(data)
        except Exception:
            return None
    
    def get_event_by_slug(self, slug: str) -> dict | None:
        """
        Get an event by its slug.
        
        Args:
            slug: The event slug
            
        Returns:
            Event dictionary or None
        """
        try:
            resp = self._session.get(f"{self.GAMMA_API}/events/slug/{slug}", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None
    
    def find_esports_markets(self) -> list[Market]:
        """
        Find all esports related markets.
        
        Returns:
            List of esports markets
        """
        all_markets = self.get_markets(limit=500)
        
        esports_terms = [
            "dota", "league of legends", "lol", "csgo", "cs2", "counter-strike",
            "valorant", "overwatch", "esports", "esport", "tournament", "major",
            "ti ", "the international", "worlds", "msi", "dreamhack"
        ]
        
        esports_markets = []
        for m in all_markets:
            text = f"{m.question} {m.description}".lower()
            if any(term in text for term in esports_terms):
                esports_markets.append(m)
        
        return esports_markets
    
    def get_market(self, market_id: str) -> Market | None:
        """
        Get a specific market by ID.
        
        Args:
            market_id: The market ID (condition_id)
            
        Returns:
            Market object or None if not found
        """
        if market_id in self._markets_cache:
            return self._markets_cache[market_id]
        
        try:
            resp = self._session.get(f"{self.GAMMA_API}/markets/{market_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            market = self._parse_market(data)
            if market:
                self._markets_cache[market_id] = market
            return market
        except Exception:
            return None
    
    def get_order_book(self, token_id: str) -> dict[str, Any]:
        """
        Get the order book for a specific token.
        
        Args:
            token_id: The token ID (YES or NO outcome)
            
        Returns:
            Order book with bids and asks
        """
        try:
            resp = self._session.get(
                f"{self.CLOB_HOST}/book",
                params={"token_id": token_id},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {"bids": [], "asks": []}
    
    def get_market_odds(self, market: Market | str) -> dict[str, float]:
        """
        Get current odds for a market.
        
        Args:
            market: Market object or market ID
            
        Returns:
            Dict mapping outcome -> probability (0-1)
        """
        if isinstance(market, str):
            market = self.get_market(market)
        if not market:
            return {}
        
        odds = {}
        for outcome, token_id in market.token_ids.items():
            try:
                book = self.get_order_book(token_id)
                # Best ask is the price to buy YES
                if book.get("asks"):
                    best_ask = float(book["asks"][0]["price"])
                    odds[outcome] = best_ask
                elif book.get("bids"):
                    best_bid = float(book["bids"][0]["price"])
                    odds[outcome] = best_bid
                else:
                    # Fallback to token price from market data
                    for t in market.tokens:
                        if t.get("outcome") == outcome:
                            odds[outcome] = float(t.get("price", 0.5))
                            break
            except Exception:
                pass
        
        return odds
    
    def place_bet(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float,
        order_type: str = "GTC",  # GTC = Good-Till-Cancelled
    ) -> Order | None:
        """
        Place a limit order bet.
        
        Args:
            token_id: The token ID to bet on
            side: "BUY" to buy YES tokens, "SELL" to sell
            price: Price per share (0.01 to 0.99, represents probability)
            size: Number of shares/tokens to buy
            order_type: "GTC" (Good-Till-Cancelled) or "FOK" (Fill-Or-Kill)
            
        Returns:
            Order object if successful, None otherwise
            
        Example:
            # Buy 5 YES tokens at 0.43 (43% implied probability)
            order = client.place_bet(
                token_id="114304...",
                side="BUY",
                price=0.43,
                size=5.0,
            )
        """
        if not self.can_trade:
            print("❌ Client not configured for trading. Set POLYMARKET_PRIVATE_KEY.")
            return None
        
        if not HAS_POLYMARKET:
            print("❌ py-clob-client not installed. pip install py-clob-client")
            return None
        
        try:
            # Ensure API credentials are set
            self._ensure_api_creds()
            
            # Create order args
            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY if side.upper() == "BUY" else SELL,
                token_id=token_id,
            )
            
            # Create and sign the order
            signed_order = self._client.create_order(order_args)
            
            # Submit order
            if order_type.upper() == "GTC":
                result = self._client.post_order(signed_order, OrderType.GTC)
            else:
                result = self._client.post_order(signed_order, OrderType.FOK)
            
            print(f"✅ Order placed: {result}")
            
            return Order(
                id=result.get("orderID", result.get("id", "")),
                market_id="",
                token_id=token_id,
                side=side.upper(),
                price=price,
                size=size,
                status="OPEN",
            )
            
        except Exception as e:
            print(f"❌ Error placing bet: {e}")
            return None
    
    def place_bet_usd(
        self,
        token_id: str,
        side: str,
        price: float,
        usd_amount: float,
    ) -> Order | None:
        """
        Place a bet specifying USD amount instead of shares.
        
        Args:
            token_id: The token ID to bet on
            side: "BUY" or "SELL"
            price: Price per share (0.01 to 0.99)
            usd_amount: Amount in USD to spend
            
        Returns:
            Order object if successful
        """
        # Calculate number of shares from USD amount
        size = usd_amount / price
        return self.place_bet(token_id, side, price, size)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: The order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            self._client.cancel(order_id)
            return True
        except Exception:
            return False
    
    def get_open_orders(self) -> list[Order]:
        """
        Get all open orders.
        
        Returns:
            List of open Order objects
        """
        try:
            raw_orders = self._client.get_orders()
            orders = []
            for o in raw_orders:
                if o.get("status") == "OPEN":
                    orders.append(Order(
                        id=o.get("orderID", ""),
                        market_id=o.get("marketID", ""),
                        token_id=o.get("tokenID", ""),
                        side=o.get("side", ""),
                        price=float(o.get("price", 0)),
                        size=float(o.get("originalSize", 0)),
                        status="OPEN",
                        filled_size=float(o.get("filledSize", 0)),
                    ))
            return orders
        except Exception:
            return []
    
    def get_positions(self) -> dict[str, float]:
        """
        Get current positions (balances).
        
        Returns:
            Dict mapping token_id -> position size
        """
        try:
            # This depends on the specific API structure
            # May need adjustment based on actual API response
            return {}
        except Exception:
            return {}
    
    def _parse_market(self, raw: dict) -> Market | None:
        """Parse raw API response into Market object."""
        try:
            import json as _json
            
            # Handle both nested tokens and flat outcomes/prices format
            tokens = raw.get("tokens", [])
            
            if not tokens:
                # Parse from flat format (Gamma API sports markets)
                outcomes_raw = raw.get("outcomes", [])
                prices_raw = raw.get("outcomePrices", [])
                token_ids_raw = raw.get("clobTokenIds", [])
                
                # Parse JSON strings if needed
                if isinstance(outcomes_raw, str):
                    outcomes_raw = _json.loads(outcomes_raw)
                if isinstance(prices_raw, str):
                    prices_raw = _json.loads(prices_raw)
                if isinstance(token_ids_raw, str):
                    token_ids_raw = _json.loads(token_ids_raw)
                
                # Build tokens list
                tokens = []
                for i, outcome in enumerate(outcomes_raw):
                    tokens.append({
                        "outcome": outcome,
                        "price": float(prices_raw[i]) if i < len(prices_raw) else 0.5,
                        "token_id": token_ids_raw[i] if i < len(token_ids_raw) else "",
                    })
            
            outcomes = [t.get("outcome", "Unknown") for t in tokens]
            
            return Market(
                id=str(raw.get("id", raw.get("conditionId", ""))),
                condition_id=raw.get("conditionId", raw.get("condition_id", "")),
                question=raw.get("question", ""),
                description=raw.get("description", ""),
                outcomes=outcomes,
                tokens=tokens,
                end_date=raw.get("endDateIso", raw.get("end_date_iso")),
                volume=float(raw.get("volumeNum", raw.get("volume", 0)) or 0),
                liquidity=float(raw.get("liquidityNum", raw.get("liquidity", 0)) or 0),
                slug=raw.get("slug", ""),
                game_start_time=raw.get("gameStartTime", raw.get("game_start_time")),
            )
        except Exception as e:
            print(f"Error parsing market: {e}")
            return None


# Convenience function for quick access
def get_client() -> PolymarketClient:
    """Get a Polymarket client from environment variables."""
    return PolymarketClient.from_env()

