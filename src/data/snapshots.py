from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def canonical_market_data(frame: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    """Return a deterministic, validated table for hashing and persistence."""
    columns = list(value_columns)
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing market data columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Market data snapshot cannot be empty")

    observations = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    if observations.has_duplicates:
        raise ValueError("Market data contains duplicate observation timestamps")

    normalized = frame.loc[:, columns].copy()
    normalized.index = observations
    normalized = normalized.sort_index()
    for column in columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[columns].isna().any().any():
        raise ValueError("Market data contains missing or non-numeric values")

    normalized.index = normalized.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    normalized.index.name = "observed_at"
    return normalized


def canonical_csv_bytes(frame: pd.DataFrame, value_columns: Iterable[str]) -> bytes:
    normalized = canonical_market_data(frame, value_columns)
    return normalized.to_csv(lineterminator="\n", float_format="%.12g").encode("utf-8")


def build_snapshot_revision_report(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    value_columns: Iterable[str],
) -> dict[str, object]:
    """Describe exact observation-level changes between two validated snapshots."""
    columns = list(value_columns)
    previous_normalized = canonical_market_data(previous, columns)
    current_normalized = canonical_market_data(current, columns)
    previous_hash = hashlib.sha256(canonical_csv_bytes(previous, columns)).hexdigest()
    current_hash = hashlib.sha256(canonical_csv_bytes(current, columns)).hexdigest()

    previous_rows = previous_normalized.to_dict(orient="index")
    current_rows = current_normalized.to_dict(orient="index")
    observations = sorted(set(previous_rows) | set(current_rows))
    changes: list[dict[str, object]] = []

    for observed_at in observations:
        previous_row = previous_rows.get(observed_at)
        current_row = current_rows.get(observed_at)
        for column in columns:
            old_value = None if previous_row is None else float(previous_row[column])
            new_value = None if current_row is None else float(current_row[column])
            if old_value == new_value:
                continue
            if previous_row is None:
                change_type = "added"
            elif current_row is None:
                change_type = "removed"
            else:
                change_type = "changed"
            changes.append(
                {
                    "observed_at": observed_at,
                    "column": column,
                    "change_type": change_type,
                    "old_value": old_value,
                    "new_value": new_value,
                }
            )

    return {
        "previous_sha256": previous_hash,
        "current_sha256": current_hash,
        "change_count": len(changes),
        "changes": changes,
    }


def build_snapshot_manifest(
    frame: pd.DataFrame,
    *,
    value_columns: Iterable[str],
    provider: str,
    identifier: str,
    meaning: str,
    unit: str,
    observation_timezone: str,
    requested_start: datetime,
    requested_end: datetime,
    source_url: str,
    retrieved_at: datetime | None = None,
    code_commit_sha: str | None = None,
) -> dict[str, object]:
    columns = list(value_columns)
    normalized = canonical_market_data(frame, columns)
    payload = canonical_csv_bytes(frame, columns)
    retrieval_time = retrieved_at or datetime.now(timezone.utc)
    if retrieval_time.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone")

    return {
        "provider": provider,
        "identifier": identifier,
        "meaning": meaning,
        "unit": unit,
        "observation_timezone": observation_timezone,
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "retrieved_at": retrieval_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "row_count": len(normalized),
        "min_observation": normalized.index.min(),
        "max_observation": normalized.index.max(),
        "source_url": source_url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "code_commit_sha": code_commit_sha,
        "columns": columns,
    }


def write_immutable_snapshot(
    directory: Path,
    frame: pd.DataFrame,
    manifest: dict[str, object],
    *,
    value_columns: Iterable[str],
) -> tuple[Path, Path]:
    payload = canonical_csv_bytes(frame, value_columns)
    actual_hash = hashlib.sha256(payload).hexdigest()
    expected_hash = manifest.get("sha256")
    if actual_hash != expected_hash:
        raise ValueError("Snapshot manifest hash does not match market data")

    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / f"{actual_hash}.csv"
    manifest_path = directory / f"{actual_hash}.json"
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    _write_once(data_path, payload)
    _write_once(manifest_path, manifest_payload)
    return data_path, manifest_path


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite immutable snapshot: {path}")
        return
    path.write_bytes(payload)
