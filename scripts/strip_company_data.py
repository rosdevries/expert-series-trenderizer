"""
Run before committing data updates to the public repo.
Removes company-identifying columns from question Parquet files.
"""
from pathlib import Path
import pandas as pd

STRIP_COLS = {"inferred_company", "email_domain"}
q_dir = Path(__file__).parent.parent / "data" / "questions"

for f in sorted(q_dir.rglob("*.parquet")):
    df = pd.read_parquet(f)
    dropped = [c for c in STRIP_COLS if c in df.columns]
    if dropped:
        df.drop(columns=dropped).to_parquet(f, index=False)
        print(f"Stripped {dropped} from {f.relative_to(q_dir.parent.parent)}")

print("Done. Safe to commit.")
