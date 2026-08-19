from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.snapshots import (
    build_snapshot_manifest,
    canonical_csv_bytes,
    write_immutable_snapshot,
)


def _frame(order=(0, 1)):
    values = pd.DataFrame(
        {"close": [39000.0, 39125.5]},
        index=pd.to_datetime(["2026-08-18T06:00:00Z", "2026-08-19T06:00:00Z"]),
    )
    return values.iloc[list(order)]


def _manifest(frame):
    return build_snapshot_manifest(
        frame,
        value_columns=["close"],
        provider="Yahoo Finance via yfinance",
        identifier="^N225",
        meaning="Nikkei Stock Average index level",
        unit="index points",
        observation_timezone="UTC-normalized from provider timestamps",
        requested_start=datetime(2026, 8, 18, tzinfo=timezone.utc),
        requested_end=datetime(2026, 8, 20, tzinfo=timezone.utc),
        source_url="https://finance.yahoo.com/quote/%5EN225/history/",
        retrieved_at=datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc),
        code_commit_sha="abc123",
    )


def test_snapshot_hash_is_independent_of_input_row_order():
    first = _frame((0, 1))
    reversed_frame = _frame((1, 0))

    assert canonical_csv_bytes(first, ["close"]) == canonical_csv_bytes(reversed_frame, ["close"])
    assert _manifest(first)["sha256"] == _manifest(reversed_frame)["sha256"]


def test_manifest_records_provenance_and_observation_range():
    manifest = _manifest(_frame())

    assert manifest["provider"] == "Yahoo Finance via yfinance"
    assert manifest["identifier"] == "^N225"
    assert manifest["unit"] == "index points"
    assert manifest["row_count"] == 2
    assert manifest["min_observation"] == "2026-08-18T06:00:00Z"
    assert manifest["max_observation"] == "2026-08-19T06:00:00Z"
    assert len(manifest["sha256"]) == 64
    assert manifest["code_commit_sha"] == "abc123"


def test_snapshot_rejects_duplicate_observation_times():
    frame = pd.DataFrame(
        {"close": [39000.0, 39100.0]},
        index=pd.to_datetime(["2026-08-19T06:00:00Z", "2026-08-19T06:00:00Z"]),
    )

    with pytest.raises(ValueError, match="duplicate observation"):
        _manifest(frame)


def test_snapshot_rejects_missing_values():
    frame = pd.DataFrame(
        {"close": [39000.0, None]},
        index=pd.to_datetime(["2026-08-18T06:00:00Z", "2026-08-19T06:00:00Z"]),
    )

    with pytest.raises(ValueError, match="missing or non-numeric"):
        _manifest(frame)


def test_immutable_snapshot_uses_content_hash_and_refuses_changed_manifest(tmp_path):
    frame = _frame()
    manifest = _manifest(frame)

    data_path, manifest_path = write_immutable_snapshot(
        tmp_path,
        frame,
        manifest,
        value_columns=["close"],
    )
    assert data_path.name == f"{manifest['sha256']}.csv"
    assert manifest_path.name == f"{manifest['sha256']}.json"

    changed = dict(manifest)
    changed["provider"] = "different provider"
    with pytest.raises(RuntimeError, match="Refusing to overwrite immutable snapshot"):
        write_immutable_snapshot(tmp_path, frame, changed, value_columns=["close"])
