"""Synthetic apiary database generator.

Why synthetic data
------------------
A real system would read a keeper's actual hive log; for a self-contained,
reproducible demo we generate a realistic SQLite catalogue instead: a handful
of hives with a season's worth of inspection history (brood pattern, queen
sightings, mite counts, temperament, stores) and a few treatment records.
A fixed RNG seed makes every run -- and therefore every evaluation -- repeatable.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from apiary_network import config

REGIONS = ["Pacific Northwest", "Northeast", "Mid-Atlantic", "Upper Midwest"]

HIVES = [
    ("A1", "Hive A1 - east fence line", "Northeast"),
    ("A2", "Hive A2 - east fence line", "Northeast"),
    ("B1", "Hive B1 - garden boxes", "Northeast"),
    ("C1", "Hive C1 - back orchard", "Mid-Atlantic"),
    ("C2", "Hive C2 - back orchard", "Mid-Atlantic"),
    ("D1", "Hive D1 - rooftop", "Upper Midwest"),
]

BROOD_PATTERNS = ["solid and tight", "spotty", "solid with some drone comb", "excellent, wall to wall"]
TEMPERAMENTS = ["calm", "a bit defensive", "calm", "calm", "testy near the bottom box"]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS hives;
        DROP TABLE IF EXISTS inspections;
        DROP TABLE IF EXISTS treatments;

        CREATE TABLE hives (
            hive_id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            region TEXT NOT NULL,
            install_date TEXT NOT NULL
        );

        CREATE TABLE inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hive_id TEXT NOT NULL REFERENCES hives(hive_id),
            inspection_date TEXT NOT NULL,
            brood_pattern TEXT,
            queen_seen INTEGER,
            eggs_seen INTEGER,
            mite_count REAL,
            temperament TEXT,
            stores_lbs REAL,
            notes TEXT
        );

        CREATE TABLE treatments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hive_id TEXT NOT NULL REFERENCES hives(hive_id),
            treatment_date TEXT NOT NULL,
            treatment_type TEXT NOT NULL,
            outdoor_temp_f REAL
        );
        """
    )


def seed(db_path: Path | None = None, seed_value: int = 42) -> Path:
    """Populate (or repopulate) the apiary database with deterministic demo data."""
    rng = random.Random(seed_value)
    db_path = db_path or config.DB_PATH
    conn = _connect(db_path)
    _create_schema(conn)

    season_start = date.today() - timedelta(days=120)

    for hive_id, label, region in HIVES:
        install_date = season_start - timedelta(days=rng.randint(180, 720))
        conn.execute(
            "INSERT INTO hives (hive_id, label, region, install_date) VALUES (?, ?, ?, ?)",
            (hive_id, label, region, install_date.isoformat()),
        )

        # 6-10 inspections spaced roughly every 1-2 weeks across the season.
        n_inspections = rng.randint(6, 10)
        inspection_date = season_start
        mite_trend = rng.uniform(0.5, 1.5)  # some hives trend worse than others
        for i in range(n_inspections):
            inspection_date = inspection_date + timedelta(days=rng.randint(9, 16))
            mite_count = round(max(0.0, mite_trend * (i + 1) * rng.uniform(0.3, 0.9)), 1)
            conn.execute(
                """INSERT INTO inspections
                   (hive_id, inspection_date, brood_pattern, queen_seen, eggs_seen,
                    mite_count, temperament, stores_lbs, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    hive_id,
                    inspection_date.isoformat(),
                    rng.choice(BROOD_PATTERNS),
                    rng.choice([1, 1, 1, 0]),
                    rng.choice([1, 1, 0]),
                    mite_count,
                    rng.choice(TEMPERAMENTS),
                    round(rng.uniform(15, 45), 1),
                    "Routine check.",
                ),
            )

        # One hive (B1) gets a treatment on record so the safety-gate demo has
        # a "recently treated" case to reason about.
        if hive_id == "B1":
            conn.execute(
                "INSERT INTO treatments (hive_id, treatment_date, treatment_type, outdoor_temp_f) "
                "VALUES (?, ?, ?, ?)",
                (hive_id, (season_start + timedelta(days=20)).isoformat(), "oxalic_acid", 68.0),
            )

    conn.commit()
    conn.close()
    return db_path


if __name__ == "__main__":
    path = seed()
    print(f"Seeded apiary database at {path}")
