import json

import pytest

from src.analysis.reporting import summarize_numbers, write_json_report


def test_summary_uses_linear_quantiles_and_handles_singletons():
    result = summarize_numbers([4, 1, 3, 2])
    assert result["count"] == 4
    assert result["mean"] == result["median"] == 2.5
    assert result["min"] == 1
    assert result["max"] == 4
    assert result["p05"] == pytest.approx(1.15)
    assert result["p25"] == pytest.approx(1.75)
    assert result["p75"] == pytest.approx(3.25)
    assert result["p95"] == pytest.approx(3.85)
    assert result["p99"] == pytest.approx(3.97)
    single = summarize_numbers([7])
    assert single["count"] == 1
    assert all(value == 7 for key, value in single.items() if key != "count")


def test_empty_summary_has_null_measurements_and_strict_json():
    result = summarize_numbers([])
    assert result["count"] == 0
    assert all(value is None for key, value in result.items() if key != "count")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        summarize_numbers([value])


def test_json_output_roundtrips_unicode_and_rejects_nan_before_writing(tmp_path):
    output = tmp_path / "nested" / "report.json"
    report = {"word": "caf\u00e9", "summary": summarize_numbers([1, 2])}
    assert write_json_report(report, output) == output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(ValueError):
        write_json_report({"bad": float("nan")}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report
