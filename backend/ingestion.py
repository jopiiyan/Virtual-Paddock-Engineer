"""Phase 1b — FastF1 → LangChain Documents → Supabase pgvector.

One Document per driver per stint. `page_content` is a natural-language summary
(embedded + semantically matched); `metadata` is for exact filtering only.

Run as a script to ingest the default race into Supabase:
    python -m backend.ingestion        # from the project root
Requires SUPABASE_URL + SUPABASE_SERVICE_KEY in the environment (or a .env file).
"""

import os
from pathlib import Path

import numpy as np
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


def fmt_laptime(td) -> str:
    """Format a lap-time Timedelta as M:SS.sss (e.g. 1:28.412)."""
    total = td.total_seconds()
    minutes, seconds = divmod(total, 60)
    return f"{int(minutes)}:{seconds:06.3f}"


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

            # Average lap time over the stint (Timedelta → "M:SS.sss")
            avg_laptime = fmt_laptime(stint_laps["LapTime"].mean())

            # Top speed seen on any of the speed traps (km/h)
            speed_cols = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
            max_speed_val = stint_laps[speed_cols].max(numeric_only=True).max()
            max_speed = int(round(max_speed_val)) if np.isfinite(max_speed_val) else None

            # Degradation = slope of lap time (s) vs lap number, i.e. seconds lost per lap.
            # Positive slope = tyres slowing down over the stint.
            fit = stint_laps[["LapNumber", "LapTime"]].dropna()
            if len(fit) >= 2:
                x = fit["LapNumber"].to_numpy(dtype=float)
                y = fit["LapTime"].dt.total_seconds().to_numpy(dtype=float)
                degradation = float(np.polyfit(x, y, 1)[0])
            else:
                degradation = 0.0

            speed_txt = f"{max_speed} km/h" if max_speed is not None else "n/a"
            summary = (
                f"In the {year} {gp}, {driver} ran stint {int(stint_num)} on {compound} "
                f"tyres over {len(stint_laps)} laps: average lap {avg_laptime}, "
                f"top speed {speed_txt}, tyre degradation {degradation:+.3f} s/lap."
            )

            docs.append(Document(  ##always page_content and metadata
                page_content=summary,
                metadata={
                    "driver": driver,           # 'HAM' — uppercase, matters for filtering
                    "year": year,
                    "grand_prix": gp,
                    "stint": int(stint_num),
                    "compound": compound,
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

    docs = build_stint_documents(year, gp)
    print(f"Built {len(docs)} stint documents for {year} {gp}.")

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
