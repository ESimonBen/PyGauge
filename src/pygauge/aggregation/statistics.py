# statistics.py
from math import floor, ceil
from collections.abc import Sequence

def percentile(values: Sequence[int | float], percent: float):
    if not values:
        raise ValueError("Cannot calculate percentile of empty data")

    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * percent / 100

    lower = floor(position)
    upper = ceil(position)

    if lower == upper:
        return float(ordered[lower])

    fraction = position - lower

    return (ordered[lower] * (1 - fraction)) + (ordered[upper] * fraction)