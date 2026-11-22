"""Tiny clients for the public, no-login F1 data APIs used by ingestion.

Why not FastF1 here: livetiming.formula1.com (FastF1's source) now answers 401
Unauthorized unless you sign in with an F1TV account. These two APIs are free
and anonymous:

  - OpenF1  (api.openf1.org)       — per-lap timing, sectors, speed traps, tyre stints
  - Jolpica (api.jolpi.ca/ergast)  — classified results: grid, status, points, times
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

OPENF1_BASE = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# Our grand_prix names (stored in metadata, shown in the UI) -> Jolpica circuitId.
# OpenF1 matches our names directly on its `location` field.
# Keys are OpenF1 `location` names (all 24 rounds of 2025, in calendar order).
JOLPICA_CIRCUITS = {
    "Melbourne": "albert_park",
    "Shanghai": "shanghai",
    "Suzuka": "suzuka",
    "Sakhir": "bahrain",
    "Jeddah": "jeddah",
    "Miami Gardens": "miami",
    "Imola": "imola",
    "Monaco": "monaco",
    "Barcelona": "catalunya",
    "Montréal": "villeneuve",
    "Spielberg": "red_bull_ring",
    "Silverstone": "silverstone",
    "Spa-Francorchamps": "spa",
    "Budapest": "hungaroring",
    "Zandvoort": "zandvoort",
    "Monza": "monza",
    "Baku": "baku",
    "Marina Bay": "marina_bay",
    "Austin": "americas",
    "Mexico City": "rodriguez",
    "São Paulo": "interlagos",
    "Las Vegas": "vegas",
    "Lusail": "losail",
    "Yas Island": "yas_marina",
}


def _get_json(url: str, retries: int = 5):
    """GET with backoff — both APIs rate-limit anonymous clients (429)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "f1-rag-ingest"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"unreachable: {url}")


def openf1(endpoint: str, **params) -> list[dict]:
    time.sleep(0.4)  # stay under OpenF1's free-tier request rate
    return _get_json(f"{OPENF1_BASE}/{endpoint}?{urllib.parse.urlencode(params)}")


def openf1_session_key(year: int, gp: str, session_name: str) -> int:
    sessions = openf1("sessions", year=year, location=gp, session_name=session_name)
    if len(sessions) != 1:
        raise ValueError(f"OpenF1: expected 1 {session_name} session for {year} {gp!r}, "
                         f"got {len(sessions)}. Check the grand prix name.")
    return sessions[0]["session_key"]


def jolpica_race_results(year: int, gp: str) -> list[dict]:
    if gp not in JOLPICA_CIRCUITS:
        raise ValueError(f"No Jolpica circuitId for {gp!r}; add it to JOLPICA_CIRCUITS.")
    data = _get_json(f"{JOLPICA_BASE}/{year}/circuits/{JOLPICA_CIRCUITS[gp]}/results.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        raise ValueError(f"Jolpica: no results for {year} {gp}.")
    return races[0]["Results"]
