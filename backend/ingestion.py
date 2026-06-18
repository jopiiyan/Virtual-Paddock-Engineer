"""FastF1 -> LangChain Documents -> Supabase pgvector.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

# Cache FIRST, before any other fastf1 call — non-negotiable (see gotchas).
# Anchored to this package dir so it works regardless of the current directory.
CACHE_DIR = Path(__file__).resolve().parent / "f1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


SESSION_LABELS = {"R": "Race", "Q": "Qualifying", "S": "Sprint", "FP1": "Practice 1",
                  "FP2": "Practice 2", "FP3": "Practice 3"}


def fmt_laptime(td) -> str:
    """Format a lap-time Timedelta as M:SS.sss (e.g. 1:28.412)."""
    total = td.total_seconds()
    minutes, seconds = divmod(total, 60)
    return f"{int(minutes)}:{seconds:06.3f}"


def fmt_sector(td) -> str:
    """Format a sector Timedelta as SS.sss seconds, or 'n/a' if missing."""
    if td is None or pd.isna(td):
        return "n/a"
    return f"{td.total_seconds():.3f}s"


def trap_speed(stint_laps, col) -> str:
    """Max speed (km/h) seen at one speed trap over the stint, or 'n/a'."""
    val = stint_laps[col].max()
    return f"{int(round(val))} km/h" if np.isfinite(val) else "n/a"


def build_stint_documents(year: int, gp: str, session_type: str = "R") -> list[Document]:
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False)   # we only need lap data

    laps = session.laps.pick_quicklaps()   # drops in/out/safety-car laps

    docs = []
    for driver in laps["Driver"].unique():
        driver_laps = laps.pick_drivers(driver)
        for stint_num, stint_laps in driver_laps.groupby("Stint"):
            if len(stint_laps) < 3:        # too short to say anything useful
                continue

            compound = stint_laps["Compound"].iloc[0]

            # Lap-time stats over the stint (Timedelta → "M:SS.sss")
            avg_laptime = fmt_laptime(stint_laps["LapTime"].mean())
            best_laptime = fmt_laptime(stint_laps["LapTime"].min())

            # Average sector times — decomposes the lap to show WHERE pace is found.
            s1 = fmt_sector(stint_laps["Sector1Time"].mean())
            s2 = fmt_sector(stint_laps["Sector2Time"].mean())
            s3 = fmt_sector(stint_laps["Sector3Time"].mean())

            # Per-trap max speeds (km/h). ST is the main straight-line trap.
            i1, i2 = trap_speed(stint_laps, "SpeedI1"), trap_speed(stint_laps, "SpeedI2")
            fl, st = trap_speed(stint_laps, "SpeedFL"), trap_speed(stint_laps, "SpeedST")

            # Degradation = slope of lap time (s) vs lap number, i.e. seconds lost per lap.
            # Positive slope = tyres slowing down over the stint.
            fit = stint_laps[["LapNumber", "LapTime"]].dropna()
            if len(fit) >= 2:
                x = fit["LapNumber"].to_numpy(dtype=float)
                y = fit["LapTime"].dt.total_seconds().to_numpy(dtype=float)
                degradation = float(np.polyfit(x, y, 1)[0])
            else:
                degradation = 0.0

            session_label = SESSION_LABELS.get(session_type, session_type)
            summary = (
                f"In the {year} {gp} {session_label}, {driver} ran stint {int(stint_num)} "
                f"on {compound} tyres over {len(stint_laps)} laps. "
                f"Average lap {avg_laptime}, best lap {best_laptime}. "
                f"Average sector times: S1 {s1}, S2 {s2}, S3 {s3}. "
                f"Top speeds at the traps: I1 {i1}, I2 {i2}, FL {fl}, ST {st}. "
                f"Tyre degradation {degradation:+.3f} s/lap "
                f"({'tyres slowing' if degradation > 0 else 'pace holding/improving'})."
            )

            docs.append(Document(  ##always page_content and metadata
                page_content=summary,
                metadata={
                    "driver": driver,           # 'HAM' — uppercase, matters for filtering
                    "year": year,
                    "grand_prix": gp,
                    "session_type": session_type,  # 'R'/'Q'/... — powers the session dropdown filter
                    "stint": int(stint_num),
                    "compound": compound,
                },
            ))
    return docs


def fmt_racetime(td) -> str | None:
    """Format the winner's total race time as H:MM:SS.sss (or M:SS.sss)."""
    if td is None or pd.isna(td):
        return None
    total = td.total_seconds()
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{seconds:06.3f}"
    return f"{int(minutes)}:{seconds:06.3f}"


def build_result_documents(year: int, gp: str, session_type: str = "R") -> list[Document]:
    """One Document per driver giving their race finishing result.

    page_content is natural-language (winner / position / finishing time or gap /
    grid / points); metadata carries driver + grand_prix so it filters alongside
    the stint documents. The winner's row holds the total race time; everyone
    else's Time is the gap to the winner.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=False, telemetry=False, weather=False)  # results only

    session_label = SESSION_LABELS.get(session_type, session_type)
    results = session.results
    # The winner's Time is the absolute race time; everyone else's Time is the
    # gap to the winner. So a finisher's own finishing time = winner_td + gap.
    winner_td = results.iloc[0]["Time"] if len(results) else None
    winner_time = fmt_racetime(winner_td)

    docs = []
    for _, row in results.iterrows():
        code = str(row["Abbreviation"]).upper()
        name = str(row["FullName"])
        team = str(row["TeamName"])
        status = str(row["Status"])
        points = row["Points"]
        grid = row["GridPosition"]
        grid_txt = f" from grid P{int(grid)}" if pd.notna(grid) else ""
        points_txt = f", scoring {int(points)} points" if pd.notna(points) else ""

        pos = row["Position"]
        finished = pd.notna(pos)
        pos_int = int(pos) if finished else None

        if pos_int == 1:
            timing = f"won the race in a finishing time of {winner_time}" if winner_time else "won the race"
        elif finished and status == "Finished" and pd.notna(row["Time"]):
            gap = row["Time"].total_seconds()
            finish_time = fmt_racetime(winner_td + row["Time"]) if winner_td is not None and pd.notna(winner_td) else None
            timing = (
                f"finished P{pos_int} in a finishing time of {finish_time} (+{gap:.3f}s behind the winner)"
                if finish_time else f"finished P{pos_int}, +{gap:.3f}s behind the winner"
            )
        elif finished:
            # Lapped or classified-but-not-on-lead-lap (Status carries '+1 Lap' etc.)
            timing = f"finished P{pos_int} ({status})"
        else:
            timing = f"did not finish (Status: {status})"

        summary = (
            f"In the {year} {gp} {session_label}, {code} ({name}, {team}) {timing}"
            f"{grid_txt}{points_txt}."
        )

        docs.append(Document(
            page_content=summary,
            metadata={
                "driver": code,
                "year": year,
                "grand_prix": gp,
                "session_type": session_type,
                "doc_type": "result",          # distinguishes result docs from stint docs
                "position": pos_int,
            },
        ))
    return docs


def main() -> None:
    # Optional: load a local .env if python-dotenv is installed.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_KEY"]
    year = int(os.environ.get("INGEST_YEAR", "2025"))
    gp = os.environ.get("INGEST_GP", "Silverstone")

    docs = build_stint_documents(year, gp) + build_result_documents(year, gp)
    print(f"Built {len(docs)} documents (stints + results) for {year} {gp}.")

    supabase = create_client(supabase_url, supabase_key)  #a template
    SupabaseVectorStore.from_documents(   ##automatically map page_content to content, metadat to metadata, embedding to embedding
        docs,
        OllamaEmbeddings(model="nomic-embed-text"),
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )
    print(f"Inserted {len(docs)} documents into Supabase 'documents' table.")


if __name__ == "__main__":
    main()
