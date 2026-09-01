"""Build the ACS income extract used by the real-data experiments.

Downloads the 2018 ACS 1-Year public-use microdata person files from the Census Bureau,
one state at a time, keeps ten columns, applies the income-task filter, and writes a
single parquet file. Puerto Rico is excluded, as in the folktables definition of the
task, and the pooled national file is skipped because it would double count the states.

Only the ten columns are retained and the filter is applied while each archive is still
in memory, so the archive is discarded immediately and nothing near the full 5.7 GB of
raw data is ever written to disk. The result is about 10 MB and takes a few minutes to
build over a normal connection.

    python fetch_acs_national.py

writes `data/acs2018_income.parquet`, which `exp_central_real.py` and
`exp_empirical_fullrecord.py` read.
"""

from __future__ import annotations

import io
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year"
COLUMNS = ["AGEP", "COW", "SCHL", "MAR", "SEX", "RAC1P", "WKHP", "OCCP",
           "PINCP", "PWGTP"]
QUERY = "(AGEP > 16) & (PINCP > 100) & (WKHP > 0) & (PWGTP >= 1) & OCCP.notna()"
# "us" is the pooled national file and would double count the states;
# "pr" is Puerto Rico, which the folktables income task does not include.
SKIP = {"us", "pr"}
OUT = Path(__file__).resolve().parent / "data" / "acs2018_income.parquet"


def state_codes():
    with urllib.request.urlopen(f"{BASE}/", timeout=60) as handle:
        page = handle.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r"csv_p([a-z]{2})\.zip", page)))


def fetch(code):
    with urllib.request.urlopen(f"{BASE}/csv_p{code}.zip", timeout=600) as handle:
        blob = handle.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as stream:
            frame = pd.read_csv(stream, usecols=COLUMNS, low_memory=False)
    return frame.query(QUERY, engine="python")[COLUMNS], len(blob)


def main():
    codes = [c for c in state_codes() if c not in SKIP]
    print(f"fetching {len(codes)} state files, keeping {len(COLUMNS)} columns", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    kept, downloaded = [], 0
    for index, code in enumerate(codes, start=1):
        try:
            frame, size = fetch(code)
        except Exception as error:                      # a missing state is not fatal
            print(f"  [{index:2d}/{len(codes)}] {code}: {error}", flush=True)
            continue
        kept.append(frame)
        downloaded += size
        print(f"  [{index:2d}/{len(codes)}] {code}: {len(frame):>7,} usable rows, "
              f"{downloaded / 1e6:6.0f} MB so far, {time.perf_counter() - started:5.0f}s",
              flush=True)
    pooled = pd.concat(kept, ignore_index=True)
    pooled.to_parquet(OUT, index=False)
    print(f"\nwrote {len(pooled):,} rows to {OUT}", flush=True)
    print(f"file size {OUT.stat().st_size / 1e6:.0f} MB; "
          f"downloaded {downloaded / 1e6:.0f} MB in "
          f"{(time.perf_counter() - started) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    sys.exit(main())
