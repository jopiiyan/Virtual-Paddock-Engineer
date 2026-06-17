"""Driver-name detection: free text ("Hamilton", "lewis") → code ("HAM").

The alias map is built from FastF1 session results (FullName / LastName /
FirstName / Abbreviation) so it stays correct per season, then cached to
`driver_aliases.json` so the API never has to load FastF1 at startup.

Regenerate the cache after ingesting new races:
    python -m backend.drivers                      # default: 2025 Silverstone R
    python -m backend.drivers 2025 Silverstone R   # explicit
"""

import json
import re
import sys
from pathlib import Path

ALIAS_CACHE = Path(__file__).resolve().parent / "driver_aliases.json"

# First names / words too ambiguous to use as driver aliases in this domain.
# "max" is excluded on purpose: our stint summaries say "max speed", which would
# otherwise false-match Verstappen. "verstappen"/"VER" still detect him fine.
_STOPWORDS = {"the", "and", "max"}


def build_alias_map(year: int, gp: str, session_type: str = "R") -> dict[str, str]:
    """Return {alias_lowercased: CODE} from a FastF1 session's results.

    Aliases per driver: full name, last name, first name, abbreviation. Last
    names and full names are the reliable hooks; first names help for casual
    phrasing ("how was lando").
    """
    import fastf1

    cache_dir = Path(__file__).resolve().parent / "f1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, laps=False, messages=False)

    aliases: dict[str, str] = {}
    for _, row in session.results.iterrows():
        code = str(row["Abbreviation"]).upper()
        # Names only — NOT the code. Codes (GAS, LAW, HAD, ...) collide with
        # common English words when lowercased, so they're matched separately
        # in detect_driver(), requiring an uppercase token in the raw text.
        for raw in (row["FullName"], row["LastName"], row["FirstName"]):
            key = str(raw).strip().lower()
            if key and key not in _STOPWORDS:
                aliases[key] = code
    return aliases


def load_alias_map() -> dict[str, str]:
    """Load the cached alias map; empty dict if it hasn't been generated yet."""
    if ALIAS_CACHE.exists():
        return json.loads(ALIAS_CACHE.read_text())
    return {}


def detect_drivers(text: str, alias_map: dict[str, str] | None = None) -> list[str]:
    """Return every distinct driver code mentioned in free text (order-preserving).

    - Explicit uppercase 3-letter codes (HAM, VER) matched against known codes.
    - Name aliases matched whole-word, case-insensitively; longer aliases first
      so "lewis hamilton" counts once.
    """
    if alias_map is None:
        alias_map = load_alias_map()
    if not alias_map:
        return []

    codes = set(alias_map.values())
    found: list[str] = []

    def add(code: str) -> None:
        if code not in found:
            found.append(code)

    for token in re.findall(r"\b[A-Z]{3}\b", text):
        if token in codes:
            add(token)

    lowered = text.lower()
    for alias in sorted(alias_map, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            add(alias_map[alias])
    return found


def detect_driver(text: str, alias_map: dict[str, str] | None = None) -> str | None:
    """Single driver code when exactly one is mentioned, else None.

    Zero or multiple matches → None, so the caller searches across all drivers
    (right for broad questions and "HAM vs NOR"-style comparisons).
    """
    drivers = detect_drivers(text, alias_map)
    return drivers[0] if len(drivers) == 1 else None


def main() -> None:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    gp = sys.argv[2] if len(sys.argv) > 2 else "Silverstone"
    session_type = sys.argv[3] if len(sys.argv) > 3 else "R"

    aliases = build_alias_map(year, gp, session_type)
    ALIAS_CACHE.write_text(json.dumps(aliases, indent=2, sort_keys=True))
    codes = sorted(set(aliases.values()))
    print(f"Wrote {len(aliases)} aliases for {len(codes)} drivers to {ALIAS_CACHE.name}")
    print("Drivers:", ", ".join(codes))


if __name__ == "__main__":
    main()
