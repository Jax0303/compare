from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd

def h_rcl_summary(table: pd.DataFrame, table_id: str | None = None, max_rows: int = 50) -> str:
    # Hierarchical Row/Column/Path-like textualization
    # For simplicity, we dump header hierarchy from MultiIndex if present.
    # Then list top-N rows with (row path) -> (key:value...) pairs.
    df = table.copy()
    if max_rows:
        df = df.head(max_rows)
    parts: List[str] = []
    if isinstance(df.columns, pd.MultiIndex):
        parts.append("COLUMNS: " + " | ".join(["/".join(map(str, t)) for t in df.columns.tolist()[:50]]))
    else:
        parts.append("COLUMNS: " + " | ".join(map(str, df.columns.tolist()[:50])))
    for i, row in df.iterrows():
        row_path = str(i) if not isinstance(i, tuple) else "/".join(map(str, i))
        kv = "; ".join(f"{c}={row[c]}" for c in df.columns[:10])
        parts.append(f"ROW {row_path} -> {kv}")
    return f"TABLE<{table_id or 'unknown'}>\n" + "\n".join(parts)
