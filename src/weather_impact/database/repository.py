"""Save, list, and delete What-If and prediction scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Union

from weather_impact.config.settings import MAX_SCENARIO_NAME_LENGTH
from weather_impact.database.db import connect
from weather_impact.utils.exceptions import InvalidInputError

PathLike = Union[str, Path]
VALID_SOURCES = ("prediction", "what_if")


@dataclass(frozen=True)
class Scenario:
    """One saved scenario. Fields match the scenarios table."""

    name: str
    source: str
    crop_type: str
    season: str
    temperature_c: float
    rainfall_mm: float
    humidity_pct: float
    predicted_yield: float
    risk_score: float
    risk_level: str
    id: Optional[int] = None
    created_at: str = field(default="")

    def as_dict(self) -> dict:
        return asdict(self)


def save_scenario(scenario: Scenario, path: Optional[PathLike] = None) -> int:
    """Insert a scenario and return its new id."""
    name = scenario.name.strip()
    if not name:
        raise InvalidInputError("Scenario name must not be empty.")
    if len(name) > MAX_SCENARIO_NAME_LENGTH:
        raise InvalidInputError(f"Scenario name must be at most {MAX_SCENARIO_NAME_LENGTH} characters.")
    if scenario.source not in VALID_SOURCES:
        raise InvalidInputError(f"Source must be one of: {', '.join(VALID_SOURCES)}.")

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO scenarios (name, created_at, source, crop_type, season, temperature_c,
                                   rainfall_mm, humidity_pct, predicted_yield, risk_score, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                created_at,
                scenario.source,
                scenario.crop_type,
                scenario.season,
                float(scenario.temperature_c),
                float(scenario.rainfall_mm),
                float(scenario.humidity_pct),
                float(scenario.predicted_yield),
                float(scenario.risk_score),
                scenario.risk_level,
            ),
        )
        return int(cursor.lastrowid)


def list_scenarios(path: Optional[PathLike] = None) -> List[Scenario]:
    """Return all saved scenarios, newest first."""
    with connect(path) as conn:
        rows = conn.execute("SELECT * FROM scenarios ORDER BY id DESC").fetchall()
    return [
        Scenario(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            source=row["source"],
            crop_type=row["crop_type"],
            season=row["season"],
            temperature_c=row["temperature_c"],
            rainfall_mm=row["rainfall_mm"],
            humidity_pct=row["humidity_pct"],
            predicted_yield=row["predicted_yield"],
            risk_score=row["risk_score"],
            risk_level=row["risk_level"],
        )
        for row in rows
    ]


def delete_scenarios(ids: Sequence[int], path: Optional[PathLike] = None) -> int:
    """Delete scenarios by id and return how many were removed."""
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    placeholders = ", ".join("?" for _ in ids)
    with connect(path) as conn:
        cursor = conn.execute(f"DELETE FROM scenarios WHERE id IN ({placeholders})", ids)
        return int(cursor.rowcount)
