# core/duckdb_executor.py
"""
DuckDBExecutor: lightweight wrapper around duckdb to run ad-hoc SQL against CSV/Parquet.
"""
import duckdb
import pandas as pd
from typing import Any

class DuckDBExecutor:
    def __init__(self, database: str = ":memory:"):
        self.conn = duckdb.connect(database=database)
        # Allow parallel/optimized reads if required
        # No special setup needed

    def query(self, sql: str) -> pd.DataFrame:
        """
        Run SQL and return a pandas.DataFrame.
        Uses duckdb's read_csv_auto('{path}') convention in SQL.
        """
        try:
            df = self.conn.execute(sql).df()
            return df
        except Exception as e:
            # Re-raise with SQL included to help debug
            raise RuntimeError(f"DuckDB query failed: {e}\nSQL: {sql}")
