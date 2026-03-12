"""
SQLite-based match storage for Dota Predictor.

This provides a unified, efficient storage system that:
- Stores all matches in a single database file
- Supports incremental updates (add new matches without re-processing)
- Allows efficient queries (by ID, date, duration, etc.)
- Deduplicates automatically (match_id is unique)

Usage:
    from dota_predictor.data.database import MatchDatabase

    db = MatchDatabase("data/matches.db")

    # Add matches (deduplicates automatically)
    db.add_matches(matches_list)

    # Query matches
    all_matches = db.get_all_matches()
    recent = db.get_matches(limit=1000, min_duration=20)

    # Get match by ID
    match = db.get_match(8610327187)

    # Statistics
    stats = db.get_stats()
    print(f"Total: {stats['total_matches']}, Avg duration: {stats['avg_duration_min']}")
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class MatchDatabase:
    """SQLite-based match storage."""

    def __init__(self, db_path: str | Path = "data/matches.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id INTEGER PRIMARY KEY,
                    radiant_win BOOLEAN NOT NULL,
                    duration INTEGER NOT NULL,
                    start_time INTEGER NOT NULL,
                    radiant_score INTEGER,
                    dire_score INTEGER,
                    game_mode INTEGER,
                    lobby_type INTEGER,
                    has_time_series BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data JSON NOT NULL
                )
            """)

            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_start_time
                ON matches(start_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_duration
                ON matches(duration)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_radiant_win
                ON matches(radiant_win)
            """)

            conn.commit()

    def add_match(self, match_data: dict) -> bool:
        """
        Add a single match to the database.

        Args:
            match_data: Raw match data from OpenDota API

        Returns:
            True if match was added, False if it already exists
        """
        match_id = match_data.get("match_id")
        if not match_id:
            return False

        # Check if match has time-series data
        has_time_series = any(
            len(p.get("gold_t", [])) > 0
            for p in match_data.get("players", [])
        )

        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO matches
                    (match_id, radiant_win, duration, start_time, radiant_score,
                     dire_score, game_mode, lobby_type, has_time_series, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match_id,
                        match_data.get("radiant_win", False),
                        match_data.get("duration", 0),
                        match_data.get("start_time", 0),
                        match_data.get("radiant_score"),
                        match_data.get("dire_score"),
                        match_data.get("game_mode"),
                        match_data.get("lobby_type"),
                        has_time_series,
                        json.dumps(match_data),
                    ),
                )
                return conn.total_changes > 0
        except sqlite3.Error:
            return False

    def add_matches(self, matches: list[dict], progress: bool = True) -> tuple[int, int]:
        """
        Add multiple matches to the database.

        Args:
            matches: List of match data dictionaries
            progress: Show progress during import

        Returns:
            Tuple of (added_count, skipped_count)
        """
        added = 0
        skipped = 0

        with self._get_conn() as conn:
            for i, match_data in enumerate(matches):
                match_id = match_data.get("match_id")
                if not match_id:
                    skipped += 1
                    continue

                has_time_series = any(
                    len(p.get("gold_t", [])) > 0
                    for p in match_data.get("players", [])
                )

                try:
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO matches
                        (match_id, radiant_win, duration, start_time, radiant_score,
                         dire_score, game_mode, lobby_type, has_time_series, data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            match_id,
                            match_data.get("radiant_win", False),
                            match_data.get("duration", 0),
                            match_data.get("start_time", 0),
                            match_data.get("radiant_score"),
                            match_data.get("dire_score"),
                            match_data.get("game_mode"),
                            match_data.get("lobby_type"),
                            has_time_series,
                            json.dumps(match_data),
                        ),
                    )
                    if cursor.rowcount > 0:
                        added += 1
                    else:
                        skipped += 1
                except sqlite3.Error:
                    skipped += 1

                if progress and (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(matches)} matches...")

            conn.commit()

        return added, skipped

    def get_match(self, match_id: int) -> dict | None:
        """
        Get a specific match by ID.

        Args:
            match_id: The match ID

        Returns:
            Match data dictionary or None
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT data FROM matches WHERE match_id = ?",
                (match_id,),
            ).fetchone()

            if row:
                return json.loads(row["data"])
            return None

    def get_matches(
        self,
        limit: int | None = None,
        offset: int = 0,
        min_duration: int | None = None,
        max_duration: int | None = None,
        radiant_win: bool | None = None,
        has_time_series: bool = True,
        order_by: str = "start_time",
        descending: bool = True,
    ) -> list[dict]:
        """
        Query matches with filters.

        Args:
            limit: Maximum matches to return
            offset: Skip this many matches
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            radiant_win: Filter by winner
            has_time_series: Only return matches with time-series data
            order_by: Column to sort by
            descending: Sort descending if True

        Returns:
            List of match data dictionaries
        """
        conditions = []
        params: list[Any] = []

        if min_duration is not None:
            conditions.append("duration >= ?")
            params.append(min_duration)

        if max_duration is not None:
            conditions.append("duration <= ?")
            params.append(max_duration)

        if radiant_win is not None:
            conditions.append("radiant_win = ?")
            params.append(radiant_win)

        if has_time_series:
            conditions.append("has_time_series = 1")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order = "DESC" if descending else "ASC"

        query = f"""
            SELECT data FROM matches
            WHERE {where_clause}
            ORDER BY {order_by} {order}
        """

        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [json.loads(row["data"]) for row in rows]

    def get_all_matches(self, has_time_series: bool = True) -> list[dict]:
        """
        Get all matches from the database.

        Args:
            has_time_series: Only return matches with time-series data

        Returns:
            List of all match data dictionaries
        """
        return self.get_matches(has_time_series=has_time_series)

    def get_match_ids(self, has_time_series: bool = True) -> list[int]:
        """
        Get all match IDs in the database.

        Args:
            has_time_series: Only return IDs of matches with time-series data

        Returns:
            List of match IDs
        """
        where = "WHERE has_time_series = 1" if has_time_series else ""
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT match_id FROM matches {where} ORDER BY match_id"
            ).fetchall()
            return [row["match_id"] for row in rows]

    def count(self, has_time_series: bool = True) -> int:
        """
        Count matches in the database.

        Args:
            has_time_series: Only count matches with time-series data

        Returns:
            Number of matches
        """
        where = "WHERE has_time_series = 1" if has_time_series else ""
        with self._get_conn() as conn:
            row = conn.execute(f"SELECT COUNT(*) as cnt FROM matches {where}").fetchone()
            return row["cnt"] if row else 0

    def exists(self, match_id: int) -> bool:
        """Check if a match exists in the database."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM matches WHERE match_id = ?",
                (match_id,),
            ).fetchone()
            return row is not None

    def get_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_matches,
                    SUM(has_time_series) as with_time_series,
                    SUM(radiant_win) as radiant_wins,
                    AVG(duration) as avg_duration,
                    MIN(start_time) as oldest_match,
                    MAX(start_time) as newest_match,
                    MIN(duration) as min_duration,
                    MAX(duration) as max_duration
                FROM matches
            """).fetchone()

            if not row or row["total_matches"] == 0:
                return {
                    "total_matches": 0,
                    "with_time_series": 0,
                    "radiant_wins": 0,
                    "radiant_win_rate": 0.0,
                    "avg_duration_min": 0.0,
                    "min_duration_min": 0,
                    "max_duration_min": 0,
                    "oldest_match": None,
                    "newest_match": None,
                }

            total = row["total_matches"]
            radiant_wins = row["radiant_wins"] or 0

            return {
                "total_matches": total,
                "with_time_series": row["with_time_series"] or 0,
                "radiant_wins": radiant_wins,
                "radiant_win_rate": radiant_wins / total if total > 0 else 0.0,
                "avg_duration_min": (row["avg_duration"] or 0) / 60,
                "min_duration_min": (row["min_duration"] or 0) / 60,
                "max_duration_min": (row["max_duration"] or 0) / 60,
                "oldest_match": (
                    datetime.fromtimestamp(row["oldest_match"]).isoformat()
                    if row["oldest_match"]
                    else None
                ),
                "newest_match": (
                    datetime.fromtimestamp(row["newest_match"]).isoformat()
                    if row["newest_match"]
                    else None
                ),
            }

    def delete_match(self, match_id: int) -> bool:
        """Delete a match from the database."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM matches WHERE match_id = ?",
                (match_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def vacuum(self) -> None:
        """Compact the database file."""
        with self._get_conn() as conn:
            conn.execute("VACUUM")

