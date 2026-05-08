"""
Persistent data store: Parquet partitioned by year/month + JSON manifest/vocab.

Layout:
    data/
        manifest.json
        _runs/runs.jsonl
        events/year=YYYY/month=MM/events.parquet
        questions/year=YYYY/month=MM/questions.parquet
        registrants/year=YYYY/month=MM/registrants.parquet
    vocab/
        company_domains.json    {domain: {company, is_personal}}
        topics.json             {schema_version, topics: [...]}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, VOCAB_DIR


# ── Paths ────────────────────────────────────────────────────────────────────

MANIFEST_PATH = DATA_DIR / "manifest.json"
RUNS_PATH = DATA_DIR / "_runs" / "runs.jsonl"


def _partition_path(table: str, year: int, month: int) -> Path:
    return DATA_DIR / table / f"year={year}" / f"month={month:02d}" / f"{table}.parquet"


# ── Table I/O ─────────────────────────────────────────────────────────────────

def read_table(table: str) -> pd.DataFrame:
    """Read all Parquet partitions for a table into a single DataFrame."""
    base = DATA_DIR / table
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception:
            pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _upsert_partitioned(df: pd.DataFrame, table: str, pk: str) -> None:
    """
    Upsert rows into year/month partitioned Parquet files.

    Reads ALL existing partitions, merges with incoming rows, deduplicates
    globally on pk (last/incoming wins), then re-partitions and rewrites.
    This prevents stale duplicates when a row's timestamp changes and it
    moves to a different partition on re-ingest.
    """
    if df.empty:
        return

    df = df.copy()
    ts_cols = {
        "events": "start_ts_pacific",
        "questions": "created_ts_pacific",
        "registrants": "created_ts_pacific",
    }
    ts_col = ts_cols.get(table)

    if ts_col and ts_col in df.columns:
        df = df.assign(**{ts_col: pd.to_datetime(df[ts_col], utc=True, errors="coerce")})

    # Global merge: read all partitions, concat, dedup (last = incoming wins)
    existing = read_table(table)
    merged = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    merged = merged.drop_duplicates(subset=[pk], keep="last")

    # Assign partition keys based on timestamp (null → current month)
    now = datetime.now(timezone.utc)
    if ts_col and ts_col in merged.columns:
        parsed = pd.to_datetime(merged[ts_col], utc=True, errors="coerce")
        merged = merged.assign(
            _year=parsed.dt.year.fillna(now.year).astype(int),
            _month=parsed.dt.month.fillna(now.month).astype(int),
        )
    else:
        merged["_year"] = now.year
        merged["_month"] = now.month

    # Clear all existing partition files before rewriting
    base = DATA_DIR / table
    for old_file in base.rglob("*.parquet"):
        old_file.unlink()

    for (year, month), group in merged.groupby(["_year", "_month"]):
        group = group.drop(columns=["_year", "_month"], errors="ignore")
        path = _partition_path(table, int(year), int(month))
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(path, index=False)


def upsert_events(df: pd.DataFrame) -> None:
    _upsert_partitioned(df, "events", "event_id")


def upsert_questions(df: pd.DataFrame) -> None:
    _upsert_partitioned(df, "questions", "question_id")


def upsert_registrants(df: pd.DataFrame) -> None:
    _upsert_partitioned(df, "registrants", "event_user_id")


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"schema_version": 1, "last_run_at": None, "events": {}}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def log_run(record: dict) -> None:
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    runs = []
    with open(RUNS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return runs


# ── Vocab ─────────────────────────────────────────────────────────────────────

def load_company_domains() -> dict:
    """Load {domain: {company, is_personal}} mapping."""
    path = VOCAB_DIR / "company_domains.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_company_domains(cache: dict) -> None:
    path = VOCAB_DIR / "company_domains.json"
    with open(path, "w") as f:
        json.dump(dict(sorted(cache.items())), f, indent=2)


def load_topics() -> list[str]:
    path = VOCAB_DIR / "topics.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("topics", []) if isinstance(data, dict) else list(data)


def save_topics(topics: list[str]) -> None:
    path = VOCAB_DIR / "topics.json"
    with open(path, "w") as f:
        json.dump({"schema_version": 1, "topics": sorted(set(topics))}, f, indent=2)
