import json

import pandas as pd

from src.analysis import valuation


class _FakeTicker:
    def __init__(self, frame):
        self._frame = frame

    def history(self, **_kwargs):
        return self._frame.copy()


def test_fetch_nikkei_data_persists_close_snapshot(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "Open": [39000.0, 39100.0],
            "Close": [39125.5, 39200.0],
        },
        index=pd.to_datetime(["2026-07-31T06:00:00Z", "2026-08-31T06:00:00Z"]),
    )
    monkeypatch.setattr(valuation.yf, "Ticker", lambda _ticker: _FakeTicker(frame))

    result = valuation.fetch_nikkei_data(
        1,
        snapshot_directory=tmp_path,
        code_commit_sha="test-sha",
    )

    assert result.equals(frame)
    csv_files = list(tmp_path.glob("*.csv"))
    manifest_files = list(tmp_path.glob("*.json"))
    assert len(csv_files) == 1
    assert len(manifest_files) == 1

    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert manifest["provider"] == "Yahoo Finance via yfinance"
    assert manifest["identifier"] == "^N225"
    assert manifest["meaning"] == "Nikkei Stock Average index level"
    assert manifest["unit"] == "index points"
    assert manifest["row_count"] == 2
    assert manifest["code_commit_sha"] == "test-sha"
    assert csv_files[0].stem == manifest["sha256"]


def test_fetch_jgb_history_persists_validated_yield_snapshot(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {"Close": [1.62, 1.64]},
        index=pd.to_datetime(["2026-08-18T06:00:00Z", "2026-08-19T06:00:00Z"]),
    )
    monkeypatch.setattr(valuation.yf, "Ticker", lambda _ticker: _FakeTicker(frame))

    result = valuation.fetch_jgb_yield_history(
        "^JP10Y",
        pd.Timestamp("2026-08-18"),
        pd.Timestamp("2026-08-19"),
        snapshot_directory=tmp_path,
        code_commit_sha="test-sha",
    )

    assert result["jgb_yield"].tolist() == [1.62, 1.64]
    manifest_file = next(tmp_path.glob("*.json"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["identifier"] == "^JP10Y"
    assert manifest["unit"] == "percentage points"
    assert manifest["row_count"] == 2
    assert manifest["code_commit_sha"] == "test-sha"
