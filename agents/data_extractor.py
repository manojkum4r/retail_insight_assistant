# agents/data_extractor.py
"""
Data Extraction Agent (fixed)
- Ensures any internally-built SQL has {path} replaced by the real local path.
- Validates whether referenced columns exist before executing SQL.
- Builds safe fallback SQL when needed.
"""
import pandas as pd
import json
from typing import Dict, Any

from core.duckdb_executor import DuckDBExecutor

class DataExtractionAgent:
    def __init__(self, executor: DuckDBExecutor = None):
        self.executor = executor or DuckDBExecutor()

    def execute_plan(self, plan: Dict[str, Any], data_path: str) -> Dict[str, Any]:
        """
        plan: dict from LanguageToQueryAgent (may include 'sql' or not)
        data_path: local file path (e.g., /mnt/data/Sale Report.csv)
        """
        # 1) If plan provides SQL, use it; if not, build one
        sql_template = plan.get("sql") or plan.get("sql_template")
        if sql_template:
            sql_to_run = sql_template
        else:
            sql_to_run = self._build_sql(plan, data_path)

        # 2) Always substitute {path} placeholder with the actual file path
        try:
            sql_to_run = sql_to_run.replace("{path}", data_path.replace("'", "''"))
        except Exception:
            # best-effort: leave as-is if replace fails
            pass

        # 3) Safety check: ensure SQL references existing columns (simple heuristic)
        if not self._sql_references_existing_columns(sql_to_run, data_path):
            # if SQL still looks unsafe, build an always-safe SQL using actual columns
            sql_to_run = self._build_sql(plan, data_path)
            sql_to_run = sql_to_run.replace("{path}", data_path.replace("'", "''"))

        # 4) Execute the SQL
        df = self.executor.query(sql_to_run)
        rows = len(df)
        preview = df.head(50)
        return {"sql": sql_to_run, "rows_scanned": rows, "table": df, "table_preview": preview, "raw": df.to_dict(orient="records")}

    def _sql_references_existing_columns(self, sql: str, data_path: str) -> bool:
        """
        Heuristic check: if the SQL references quoted column names that exist in the CSV header,
        or if any unquoted words match existing column names (case-insensitive), return True.
        """
        try:
            df_cols = pd.read_csv(data_path, nrows=0).columns.tolist()
            df_cols_lower = [c.lower() for c in df_cols]
            # find quoted identifiers "Column Name"
            import re
            quoted = re.findall(r'"([^"]+)"', sql)
            for q in quoted:
                if q in df_cols:
                    return True
            # find simple tokens that could be column names
            tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_ ]*)\b', sql)
            for t in tokens:
                if t.strip().lower() in df_cols_lower:
                    return True
            return False
        except Exception:
            return False

    def _build_sql(self, plan: Dict[str,Any], data_path: str) -> str:
        """
        Build a safe SQL using the actual columns available in the CSV.
        Priority:
         - If revenue-like + category exists -> top categories by revenue
         - If revenue-like + SKU exists -> top SKUs
         - If category + stock-like exists -> categories by stock (inventory)
         - Fallback: counts on first column
        The returned SQL uses read_csv_auto('{path}') placeholder which will be replaced by execute_plan.
        """
        # Inspect available columns
        try:
            df0 = pd.read_csv(data_path, nrows=0)
            cols = df0.columns.tolist()
        except Exception:
            # if reading header fails, return a very generic SQL (may fail downstream)
            return "SELECT COUNT(*) as cnt FROM read_csv_auto('{path}')"

        # helper finders
        def find_col(candidates):
            for cand in candidates:
                for orig in cols:
                    if orig.strip().lower() == cand:
                        return orig
            return None

        revenue_col = find_col(["amount", "revenue", "gross amt", "gross_amt", "sale_value", "net_sales", "sales", "amount (in rs)"])
        qty_col = find_col(["qty", "quantity", "units", "pcs", "units_sold", "stock"])
        category_col = None
        for orig in cols:
            if "category" in orig.lower():
                category_col = orig
                break
        sku_col = find_col(["sku", "sku code", "sku_code", "product_id", "product id", "article code"])
        stock_col = find_col(["stock", "instock", "available", "qty"])

        # Build SQL options
        if revenue_col and category_col:
            sql = f"""
            SELECT "{category_col}" AS category,
                   SUM(CAST("{revenue_col}" AS DOUBLE)) AS total_revenue,
                   SUM(CAST(COALESCE("{qty_col}",0) AS DOUBLE)) AS total_qty
            FROM read_csv_auto('{{path}}')
            GROUP BY 1
            ORDER BY total_revenue DESC
            LIMIT 20
            """
            return sql

        if revenue_col and sku_col:
            sql = f"""
            SELECT "{sku_col}" AS sku,
                   SUM(CAST("{revenue_col}" AS DOUBLE)) AS total_revenue,
                   SUM(CAST(COALESCE("{qty_col}",0) AS DOUBLE)) AS total_qty
            FROM read_csv_auto('{{path}}')
            GROUP BY 1
            ORDER BY total_revenue DESC
            LIMIT 20
            """
            return sql

        if category_col and stock_col:
            sql = f"""
            SELECT "{category_col}" AS category,
                   SUM(CAST("{stock_col}" AS DOUBLE)) AS total_stock,
                   COUNT(*) AS rows
            FROM read_csv_auto('{{path}}')
            GROUP BY 1
            ORDER BY total_stock DESC
            LIMIT 20
            """
            return sql

        # Final fallback: frequency of first column
        first_col = cols[0]
        sql = f"""
        SELECT "{first_col}" AS value,
               COUNT(*) AS cnt
        FROM read_csv_auto('{{path}}')
        GROUP BY 1
        ORDER BY cnt DESC
        LIMIT 20
        """
        return sql
