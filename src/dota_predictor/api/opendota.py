"""
OpenDota API Client

OpenDota provides parsed match data from Dota 2, including time-series
data like gold and XP over time which is essential for our prediction models.

API Documentation: https://docs.opendota.com/
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import requests
from tqdm import tqdm


@dataclass
class RateLimiter:
    """Simple rate limiter for API requests."""

    calls_per_minute: int = 60
    _last_call: float = 0.0
    _call_count: int = 0
    _window_start: float = 0.0

    def wait_if_needed(self) -> None:
        """Block if we're exceeding rate limits."""
        current_time = time.time()

        # Reset window if minute has passed
        if current_time - self._window_start >= 60:
            self._window_start = current_time
            self._call_count = 0

        # Wait if we've exceeded calls per minute
        if self._call_count >= self.calls_per_minute:
            sleep_time = 60 - (current_time - self._window_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._window_start = time.time()
            self._call_count = 0

        self._call_count += 1
        self._last_call = time.time()


class OpenDotaClient:
    """
    Client for the OpenDota API.

    OpenDota provides rich match data including:
    - Match metadata (duration, game mode, patch, etc.)
    - Player data (heroes, items, stats)
    - Time-series data (gold_t, xp_t, lh_t, etc.) - minute-by-minute snapshots
    - Parsed events (kills, objectives, etc.)

    The time-series data is particularly valuable for our LSTM model as it
    captures the game state evolution over time.
    """

    BASE_URL = "https://api.opendota.com/api"

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the OpenDota client.

        Args:
            api_key: Optional API key for increased rate limits.
                     Get one at https://www.opendota.com/api-keys
        """
        self.api_key = api_key
        self.rate_limiter = RateLimiter(
            calls_per_minute=3000 if api_key else 30  # API key tier: 3000/min, free: 30/min
        )
        self.session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Get or create a requests session."""
        if self.session is None:
            self.session = requests.Session()
            if self.api_key:
                self.session.params = {"api_key": self.api_key}  # type: ignore
        return self.session

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make a rate-limited request to the API."""
        self.rate_limiter.wait_if_needed()

        session = self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"
        response = session.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def get_match(self, match_id: int) -> dict[str, Any]:
        """
        Get detailed match data including time-series.

        This is the primary method for getting training data. The response includes:
        - radiant_win: bool - The label we want to predict
        - duration: int - Match duration in seconds
        - players: list - 10 players with stats and time-series
        - Each player has gold_t, xp_t, lh_t, dn_t (minute-by-minute arrays)

        Args:
            match_id: The match ID to fetch

        Returns:
            Match data dictionary
        """
        return self._request(f"matches/{match_id}")

    def get_pro_matches(self, less_than_match_id: int | None = None) -> list[dict[str, Any]]:
        """
        Get recent professional matches.

        Pro matches are higher quality and more consistent, making them
        better for training. Returns ~100 matches per call.

        Args:
            less_than_match_id: Get matches before this ID (for pagination)

        Returns:
            List of match summaries (need to call get_match for full data)
        """
        params = {}
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        return self._request("proMatches", params)

    def get_public_matches(
        self,
        min_rank: int | None = None,
        less_than_match_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get recent public matches.

        Args:
            min_rank: Minimum rank tier (80 = Immortal, 70 = Divine, etc.)
            less_than_match_id: Get matches before this ID

        Returns:
            List of match summaries
        """
        params = {}
        if min_rank:
            params["min_rank"] = min_rank
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        return self._request("publicMatches", params)

    def get_parsed_matches(
        self,
        less_than_match_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get recently parsed matches.

        Parsed matches have the full time-series data we need.

        Args:
            less_than_match_id: Get matches before this ID

        Returns:
            List of parsed match summaries
        """
        params = {}
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        return self._request("parsedMatches", params)

    def request_parse(self, match_id: int) -> dict[str, Any]:
        """
        Request a match to be parsed.

        If a match isn't parsed, we won't have time-series data.
        This queues the match for parsing.

        Args:
            match_id: Match to parse

        Returns:
            Job status
        """
        session = self._get_session()
        url = f"{self.BASE_URL}/request/{match_id}"
        response = session.post(url)
        response.raise_for_status()
        return response.json()

    def get_heroes(self) -> list[dict[str, Any]]:
        """
        Get list of all heroes.

        Returns hero metadata including ID, name, roles, etc.
        Useful for creating hero embeddings or one-hot encodings.
        """
        return self._request("heroes")

    def get_hero_stats(self) -> list[dict[str, Any]]:
        """
        Get hero statistics.

        Includes pick rates, win rates, and other aggregate stats.
        """
        return self._request("heroStats")

    def fetch_matches_batch(
        self,
        match_ids: list[int],
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple matches with progress bar.

        Args:
            match_ids: List of match IDs to fetch
            show_progress: Whether to show progress bar

        Returns:
            List of match data dictionaries
        """
        matches = []
        iterator = tqdm(match_ids, desc="Fetching matches") if show_progress else match_ids

        for match_id in iterator:
            try:
                match = self.get_match(match_id)
                matches.append(match)
            except requests.HTTPError as e:
                if show_progress:
                    print(f"Error fetching match {match_id}: {e}")  # noqa: T201
                continue

        return matches

    def collect_pro_matches(
        self,
        count: int = 100,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Collect a number of pro match IDs.

        This gets match summaries first, then you can use fetch_matches_batch
        to get the full match data.

        Args:
            count: Number of matches to collect
            show_progress: Whether to show progress

        Returns:
            List of match summaries
        """
        matches: list[dict[str, Any]] = []
        last_match_id = None

        pbar = tqdm(total=count, desc="Collecting pro matches") if show_progress else None

        while len(matches) < count:
            batch = self.get_pro_matches(less_than_match_id=last_match_id)
            if not batch:
                break

            matches.extend(batch)
            last_match_id = batch[-1]["match_id"]

            if pbar:
                pbar.update(len(batch))

        if pbar:
            pbar.close()

        return matches[:count]


class AsyncRateLimiter:
    """Token bucket rate limiter for async operations."""

    def __init__(self, requests_per_second: float = 40.0) -> None:
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._last_request_time = 0.0

    async def acquire(self) -> None:
        """Wait until we can make a request."""
        async with self._lock:
            now = time.time()
            time_since_last = now - self._last_request_time
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self._last_request_time = time.time()


class AsyncOpenDotaClient:
    """
    Async version of the OpenDota client for faster batch operations.

    Use this when you need to fetch many matches concurrently.
    """

    BASE_URL = "https://api.opendota.com/api"

    def __init__(
        self,
        api_key: str | None = None,
        max_concurrent: int = 20,
        requests_per_second: float | None = None,
    ) -> None:
        """
        Initialize async client.

        Args:
            api_key: Optional API key
            max_concurrent: Max concurrent requests (default 20)
            requests_per_second: Rate limit (default: 40/s with key, 0.5/s without)
        """
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: aiohttp.ClientSession | None = None

        # Rate limit: 3000/min = 50/s with key, but use 40/s for safety buffer
        if requests_per_second is None:
            requests_per_second = 40.0 if api_key else 0.5
        self.rate_limiter = AsyncRateLimiter(requests_per_second)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self) -> None:
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> Any:
        """Make a rate-limited async request with retry on 429."""
        await self.rate_limiter.acquire()

        async with self.semaphore:
            session = await self._get_session()
            url = f"{self.BASE_URL}/{endpoint}"

            if self.api_key:
                params = params or {}
                params["api_key"] = self.api_key

            for attempt in range(retries):
                async with session.get(url, params=params) as response:
                    if response.status == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                        await asyncio.sleep(wait_time)
                        continue
                    response.raise_for_status()
                    return await response.json()

            # Final attempt failed
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=429,
                message="Rate limit exceeded after retries",
            )

    async def get_match(self, match_id: int) -> dict[str, Any]:
        """Get match data asynchronously."""
        return await self._request(f"matches/{match_id}")

    async def fetch_matches_batch(
        self,
        match_ids: list[int],
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple matches concurrently.

        Args:
            match_ids: List of match IDs
            show_progress: Whether to show progress

        Returns:
            List of match data (in order)
        """
        tasks = [self.get_match(mid) for mid in match_ids]

        if show_progress:
            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching"):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    print(f"Error: {e}")  # noqa: T201
            return results
        else:
            return await asyncio.gather(*tasks, return_exceptions=True)  # type: ignore
