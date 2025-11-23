# utils/schema_mapper.py
"""
Infer schema and provide a basic mapping of expected columns.
If dataset has different column names, users can edit mapping in UI.
"""
import pandas as pd
from typing import Dict

COMMON_COLUMNS = {
    "date": ["date", "order date", "order_date", "sale_date", "timestamp", "dt"],
    "region": ["region", "state", "market"],
    "store": ["store", "store_id", "outlet"],
    "product_id": ["product_id", "sku", "sku code", "sku_code", "item_id"],
    "product_name": ["product_name", "item_name", "product"],
    "category": ["category", "product_category"],
    "units_sold": ["units", "units_sold", "quantity", "qty", "pcs"],
    "revenue": ["revenue", "amount", "gross amt", "gross_amt", "sale_value", "net_sales", "sales"]
}

def infer_schema_and_map(path: str) -> Dict[str, str]:
    """
    Read header and map logical names to the exact header column name in the CSV.
    Returns mapping: logical_name -> actual_column_name (or None)
    """
    # Try reading headers robustly
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception:
        try:
            df = pd.read_excel(path, nrows=0)
        except Exception:
            # as a last resort return None mappings
            return {k: None for k in COMMON_COLUMNS.keys()}

    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]
    mapping = {}
    for logical, candidates in COMMON_COLUMNS.items():
        found = None
        for cand in candidates:
            if cand in cols_lower:
                idx = cols_lower.index(cand)
                found = cols[idx]
                break
        mapping[logical] = found
    return mapping
