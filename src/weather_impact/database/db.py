"""SQLite connection handling and schema for saved scenarios."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from weather_impact.config.settings import SCENARIO_DB_PATH

PathLike = Union[str, Path]

SCHEMA = """
CREATE TABLE IF NOT EXISTS scenarios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    crop_type       TEXT    NOT NULL,
    season          TEXT    NOT NULL,
    temperature_c   REAL    NOT NULL,
    rainfall_mm     REAL    NOT NULL,
    humidity_pct    REAL    NOT NULL,
    predicted_yield REAL    NOT NULL,
    risk_score      REAL    NOT NULL,
    risk_level      TEXT    NOT NULL
);
"""


@contextmanager
def connect(path: Optional[PathLike] = None) -> Iterator[sqlite3.Connection]:
    """Open the scenario database, create the schema if needed, and commit or roll back."""
    db_path = Path(path) if path is not None else Path(SCENARIO_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
