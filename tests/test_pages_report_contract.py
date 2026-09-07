from pathlib import Path


def test_pages_report_uses_point_in_time_valuation_and_evidence_state():
    source = Path("scripts/generate_report.py").read_text(encoding="utf-8")

    assert "apply_point_in_time_valuation" in source
    assert "fetch_jgb_yield_history" in source
    assert "config.valuation.jgb_yield" not in source
    assert "HISTORICAL SNAPSHOT — VERIFY FIRST" in source
    assert "VALUATION EVIDENCE UNAVAILABLE" in source
    assert "No stale yield is substituted" in source
    assert "Point-in-time fair PER" in source
    assert "Current Price" not in source
