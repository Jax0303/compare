from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
from ..utils.tables import h_rcl_summary

def build_hybrid_representation(doc: Dict[str, Any]) -> str:
    text = doc.get("text","")
    tables = doc.get("tables", [])
    parts: List[str] = [text[:2000]]
    for i, t in enumerate(tables[:3]):
        try:
            df = pd.DataFrame(t.get("rows", []), columns=t.get("columns"))
            parts.append(h_rcl_summary(df, table_id=f"{doc.get('doc_id')}_t{i}"))
        except Exception:
            continue
    return "\n\n".join(parts)
