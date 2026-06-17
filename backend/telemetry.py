"""Fastest-lap telemetry for the graph feature (separate from the RAG path).

FastF1 exposes Speed / Throttle / Brake / nGear / RPM / DRS vs Distance, plus
X/Y position. (No steering-angle channel exists in F1's data.) We take each
driver's fastest lap and resample every channel onto a shared distance grid so
multiple drivers overlay cleanly on a common x-axis.

Sessions are loaded with telemetry=True (slow the first time, then disk-cached)
and kept in an in-process cache so repeat requests are fast.
"""

from pathlib import Path

import numpy as np
import fastf1

# Cache FIRST, before any other fastf1 call (see project gotchas).
CACHE_DIR = Path(__file__).resolve().parent / "f1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Number of points on the shared distance grid per lap.
GRID_POINTS = 400

# Loaded sessions, keyed by (year, gp, session_type) — telemetry load is expensive.
_SESSION_CACHE: dict[tuple, "fastf1.core.Session"] = {}


def _fmt_laptime(td) -> str:
    total = td.total_seconds()
    minutes, seconds = divmod(total, 60)
    return f"{int(minutes)}:{seconds:06.3f}"


def get_session(year: int, gp: str, session_type: str = "R"):
    key = (year, gp, session_type)
    if key not in _SESSION_CACHE:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=True, weather=False, messages=False)  # telemetry needed here
        _SESSION_CACHE[key] = session
    return _SESSION_CACHE[key]


def fastest_lap_telemetry(year: int, gp: str, session_type: str, driver: str) -> dict:
    """One driver's fastest-lap telemetry, resampled onto a shared distance grid.

    Raises ValueError if the driver has no usable lap in the session.
    """
    session = get_session(year, gp, session_type)
    driver = driver.upper()

    driver_laps = session.laps.pick_drivers(driver)
    if driver_laps.empty:
        raise ValueError(f"No laps for driver {driver} in {year} {gp} {session_type}.")
    lap = driver_laps.pick_fastest()
    if lap is None or lap.empty:
        raise ValueError(f"No fastest lap for driver {driver}.")

    tel = lap.get_telemetry()  # merges car + position data, adds Distance
    dist = tel["Distance"].to_numpy(dtype=float)
    grid = np.linspace(float(dist.min()), float(dist.max()), GRID_POINTS)

    def resample(col, dtype=float):
        return np.interp(grid, dist, tel[col].to_numpy(dtype=dtype))

    return {
        "driver": driver,
        "lap_time": _fmt_laptime(lap["LapTime"]),
        "lap_number": int(lap["LapNumber"]),
        "compound": str(lap["Compound"]),
        "distance": [round(v, 1) for v in grid],
        "speed": [round(v, 1) for v in resample("Speed")],
        "throttle": [round(v, 1) for v in resample("Throttle")],
        "brake": [round(v) for v in resample("Brake", dtype=float)],  # 0/1
        "x": [round(v, 1) for v in resample("X")],
        "y": [round(v, 1) for v in resample("Y")],
    }


def compare_telemetry(year: int, gp: str, session_type: str, drivers: list[str]) -> dict:
    """Fastest-lap telemetry for one or more drivers, for overlay charts."""
    results, errors = [], {}
    for code in drivers:
        try:
            results.append(fastest_lap_telemetry(year, gp, session_type, code))
        except ValueError as exc:
            errors[code.upper()] = str(exc)
    return {"year": year, "grand_prix": gp, "session_type": session_type,
            "drivers": results, "errors": errors}
