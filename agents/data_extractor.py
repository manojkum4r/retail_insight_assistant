# agents/data_extractor.py

import pandas as pd
import duckdb
from typing import Dict, Any

class DataExtractionAgent:
    def __init__(self, executor=None):
        self.executor = executor

    def execute_plan(self, plan: Dict[str, Any], data_path: str) -> Dict[str, Any]:
        """
        Build SQL dynamically from the plan:
        - intent: top_n, aggregate_stats, missing_count, abc_analysis, etc.
        - metrics: stock, count, revenue, etc.
        - dimensions: category, color, size, sku, etc.
        - filters (optional)
        - limit (optional)
        """
        sql = self._build_sql_from_plan(plan, data_path)

        try:
            table = duckdb.sql(sql).df()
            return {
                "sql": sql,
                "rows_scanned": len(table),
                "table": table,
                "table_preview": table.head(10)
            }
        except Exception as e:
            return {
                "sql": sql,
                "rows_scanned": 0,
                "table": None,
                "error": str(e)
            }

    # ------------------------------------------------------------------
    # SQL builder that respects plan["dimensions"], plan["metrics"], plan["filters"]
    # ------------------------------------------------------------------
    def _build_sql_from_plan(self, plan, data_path):

        dims = plan.get("dimensions") or []
        metrics = plan.get("metrics") or []
        filters = plan.get("filters") or {}
        limit = plan.get("limit", 20)

        dims = [d for d in dims if d]  # clean None
        metrics = [m for m in metrics if m]

        # If no dimensions, fallback to all rows
        if not dims:
            dims = []

        # Build SELECT clause
        select_parts = []
        group_parts = []

        for d in dims:
            select_parts.append(f'"{d}"')
            group_parts.append(f'"{d}"')

        # Metrics
        for m in metrics:
            if m.lower() in ("stock", "qty", "quantity"):
                select_parts.append(f"SUM(CAST(\"{m}\" AS DOUBLE)) AS total_stock")
            elif m.lower() in ("revenue", "amount", "sales"):
                select_parts.append(f"SUM(CAST(\"{m}\" AS DOUBLE)) AS total_revenue")
            elif m.lower() in ("count",):
                select_parts.append(f"COUNT(*) AS count_value")
            else:
                # Generic numeric fallback
                select_parts.append(f"SUM(CAST(\"{m}\" AS DOUBLE)) AS {m}")

        if not select_parts:
            select_clause = "*"
        else:
            select_clause = ", ".join(select_parts)

        # FROM clause
        from_clause = f"read_csv_auto('{data_path}')"

        # WHERE clause
        filter_clauses = []
        for col, rule in filters.items():
            op = rule.get("op")
            val = rule.get("value")
            if isinstance(val, str):
                filter_clauses.append(f'"{col}" {op} \'{val}\'')
            else:
                filter_clauses.append(f'"{col}" {op} {val}')
        where_clause = " AND ".join(filter_clauses) if filter_clauses else "1=1"

        # GROUP BY
        if group_parts:
            group_clause = "GROUP BY " + ", ".join(group_parts)
        else:
            group_clause = ""

        # ORDER
        if "order" in plan and plan["order"] == "asc":
            order_clause = "ORDER BY 2 ASC"
        else:
            order_clause = "ORDER BY 2 DESC"

        # LIMIT
        limit_clause = f"LIMIT {limit}"

        # Final SQL
        sql = f"""
            SELECT {select_clause}
            FROM {from_clause}
            WHERE {where_clause}
            {group_clause}
            {order_clause}
            {limit_clause}
        """

        return sql.strip()
