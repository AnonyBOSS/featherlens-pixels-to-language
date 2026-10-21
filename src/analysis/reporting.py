"""Small, JSON-safe summaries shared by statistical analyses."""

import json
import math
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Any


def summarize_numbers(values: Iterable[Real]) -> dict[str, int | float | None]:
    """Describe finite numbers; quantiles interpolate at index (n - 1) * p.

    Empty inputs have count zero and null measurements, never NaN or infinity.
    This is a descriptive summary, not a confidence interval or a fitted policy.
    """
    numbers = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("summary values must be real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("summary values must be finite")
        numbers.append(number)
    numbers.sort()

    def quantile(probability: float) -> float | None:
        if not numbers:
            return None
        position = (len(numbers) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)

    return {
        "count": len(numbers),
        "min": numbers[0] if numbers else None,
        "max": numbers[-1] if numbers else None,
        "mean": fmean(numbers) if numbers else None,
        "median": quantile(0.5),
        **{name: quantile(p) for name, p in (
            ("p05", 0.05), ("p25", 0.25), ("p75", 0.75),
            ("p95", 0.95), ("p99", 0.99),
        )},
    }


def write_json_report(report: Mapping[str, Any], output: str | Path) -> Path:
    """Save a UTF-8 report, rejecting non-standard JSON numbers before writing."""
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")
    return path
