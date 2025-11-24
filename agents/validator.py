# agents/validator.py
"""
Validation Agent

- Accepts the `extraction` output from DataExtractionAgent and the `plan` from LanguageToQueryAgent.
- Produces a structured answer dict: { "key_insight": str, "top_rows_table": [...], "confidence": str, "notes": ... }
- Uses the returned dataframe (extraction["table"]) for deterministic insights. Falls back to LLM for phrasing only when necessary.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import math
import json
import os

from core.llm_client import LLMClient

class ValidationAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(provider="mock")

    def validate(self, extraction: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters:
        - extraction: dict returned by DataExtractionAgent, expected keys:
            - 'sql' (str)
            - 'table' (pd.DataFrame) or 'table_preview' (pd.DataFrame)
            - 'rows_scanned' (int)
        - plan: the structured plan from LanguageToQueryAgent (intent, metrics, dimensions, etc.)

        Returns:
        - dict with keys: answer_text (dict or str), confidence (str), notes (optional)
        """
        # Basic validation
        df = None
        if extraction is None:
            return self._make_answer_text(None, plan, confidence="low", notes="No extraction provided")

        # prefer full table if present
        if isinstance(extraction.get("table"), pd.DataFrame):
            df = extraction.get("table")
        elif isinstance(extraction.get("table_preview"), pd.DataFrame):
            df = extraction.get("table_preview")
        else:
            # try to build a dataframe from raw records if available
            raw = extraction.get("raw")
            if isinstance(raw, list) and len(raw) > 0:
                try:
                    df = pd.DataFrame(raw)
                except Exception:
                    df = None

        # If no dataframe is available, fallback to LLM summarization (low confidence)
        if df is None or df.empty:
            # If SQL and rows_scanned are present but dataframe is empty, note that
            notes = "Query returned no rows" if (extraction.get("rows_scanned") in (0, None) or (isinstance(extraction.get("rows_scanned"), int) and extraction.get("rows_scanned") == 0)) else "No table available"
            return self._make_answer_text(None, plan, confidence="low", notes=notes)

        # Normalize column names (strip whitespace)
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

        # Determine main intent
        intent = plan.get("intent", "").lower() if plan else ""

        # Build insight according to intent type
        try:
            if intent in ("top_n", "top", "topk", "topk_n") or (plan.get("metrics") and ("stock" in str(plan.get("metrics")).lower() or "count" in str(plan.get("metrics")).lower() or "revenue" in str(plan.get("metrics")).lower())):
                answer = self._insight_from_topn(df, plan)
                confidence = "high" if answer.get("top_rows_table") else "medium"
                return {"answer_text": answer, "confidence": confidence, "notes": None}
            elif intent in ("aggregate_stats", "metric_query", "aggregate", "summary"):
                answer = self._insight_from_aggregates(df, plan)
                confidence = "high"
                return {"answer_text": answer, "confidence": confidence, "notes": None}
            elif intent in ("missing_count",):
                answer = self._insight_from_missing(df, plan)
                return {"answer_text": answer, "confidence": "high", "notes": None}
            elif intent in ("abc_analysis",):
                answer = self._insight_from_abc(df, plan)
                return {"answer_text": answer, "confidence": "high", "notes": None}
            else:
                # Generic: return top rows and short summary generated from data
                answer = self._generic_insight(df, plan)
                return {"answer_text": answer, "confidence": "medium", "notes": None}
        except Exception as e:
            # If any error, fallback to textual LLM-based answer (low confidence)
            text = self._llm_summarize(df, plan, error=str(e))
            return {"answer_text": text, "confidence": "low", "notes": f"Validation failed: {e}"}

    def _insight_from_topn(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a top-n style insight based on df and plan.
        Expect df to already contain aggregated rows (dimension + metric).
        """
        # Determine ordering and limit from plan
        order = plan.get("order", "desc").lower()
        limit = plan.get("limit", 20) or 20

        # Find dimension and metric columns heuristically
        dims = [d for d in plan.get("dimensions") if d] if plan.get("dimensions") else []
        metrics = [m for m in plan.get("metrics") if m] if plan.get("metrics") else []

        # If dimensions not provided, try to infer from df columns: prefer category, sku, color, size
        if not dims:
            for cand in ["Category", "category", "SKU", "sku", "SKU Code", "product_id", "product id", "color", "size"]:
                if cand in df.columns:
                    dims = [cand]
                    break

        # If no metrics provided, infer numeric column (largest numeric)
        if not metrics:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # choose the first numeric column
                metrics = [numeric_cols[0]]
            else:
                # fallback to count
                metrics = [None]

        # Build table: sort by first metric if present; otherwise by count
        metric_col = metrics[0] if metrics and metrics[0] else None
        dim_col = dims[0] if dims else None

        out_rows = []
        # If metric column exists in df, sort by that
        if metric_col and metric_col in df.columns:
            ascending = (order == "asc")
            sorted_df = df.sort_values(by=metric_col, ascending=ascending).head(limit)
            # Format rows
            for _, row in sorted_df.iterrows():
                rowd = {col: self._safe_convert(row.get(col)) for col in sorted_df.columns}
                out_rows.append(rowd)
            # Compose key insight
            if dim_col and dim_col in sorted_df.columns:
                top_val = sorted_df.iloc[0][dim_col]
                top_metric_val = sorted_df.iloc[0][metric_col]
                key_insight = f"Top {dim_col} by {metric_col}: {top_val} ({self._fmt_number(top_metric_val)})"
            else:
                key_insight = f"Top results by {metric_col}"
            return {"key_insight": key_insight, "top_rows_table": out_rows}
        else:
            # If metric not present, use value counts of dim_col
            if dim_col and dim_col in df.columns:
                vc = df[dim_col].value_counts().head(limit)
                for idx, cnt in vc.items():
                    out_rows.append({dim_col: idx, "count": int(cnt)})
                key_insight = f"Top {dim_col} by frequency: {', '.join([str(r[dim_col]) for r in out_rows[:3]])}"
                return {"key_insight": key_insight, "top_rows_table": out_rows}
            else:
                # fallback to returning first N rows
                preview = df.head(limit)
                for _, row in preview.iterrows():
                    out_rows.append({col: self._safe_convert(row.get(col)) for col in preview.columns})
                return {"key_insight": "Top rows preview", "top_rows_table": out_rows}

    def _insight_from_aggregates(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce aggregate stats (min, max, mean, median) for numeric columns or requested metric
        """
        metrics = plan.get("metrics") or []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = None
        if metrics:
            # prefer the first metric if it exists in df
            for m in metrics:
                if m and m in df.columns:
                    target_col = m
                    break
        if not target_col and numeric_cols:
            target_col = numeric_cols[0]
        if not target_col:
            # nothing numeric to aggregate
            return {"key_insight": "No numeric column available for aggregation", "top_rows_table": []}
        series = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if series.empty:
            return {"key_insight": f"No numeric data available in column {target_col}", "top_rows_table": []}
        stats = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
            "count": int(series.count())
        }
        key_insight = f"{target_col} stats — min: {self._fmt_number(stats['min'])}, median: {self._fmt_number(stats['median'])}, max: {self._fmt_number(stats['max'])}"
        return {"key_insight": key_insight, "aggregates": stats, "top_rows_table": []}

    def _insight_from_missing(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Count missing values for specified dimensions
        """
        dims = plan.get("dimensions") or []
        if not dims:
            # default: check all object columns
            dims = df.select_dtypes(include=['object']).columns.tolist()
        missing = {}
        for d in dims:
            if d in df.columns:
                missing[d] = int(df[d].isnull().sum())
        summary = ", ".join([f"{k}: {v}" for k, v in missing.items()])
        key_insight = f"Missing counts — {summary}"
        return {"key_insight": key_insight, "top_rows_table": [], "missing_counts": missing}

    def _insight_from_abc(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a simple ABC classification by stock: A=top 70% stock by value, B=next 20%, C=remaining 10%
        Assumes df has 'sku' and 'stock' or similar numeric column.
        """
        # find sku and stock columns
        sku_col = None
        stock_col = None
        for c in df.columns:
            if "sku" in str(c).lower() or "product" in str(c).lower():
                sku_col = c
            if "stock" in str(c).lower() or "qty" in str(c).lower() or "quantity" in str(c).lower():
                stock_col = c
        if not sku_col or not stock_col:
            return {"key_insight": "Required columns for ABC analysis not found", "top_rows_table": []}
        tmp = df[[sku_col, stock_col]].copy()
        tmp[stock_col] = pd.to_numeric(tmp[stock_col], errors="coerce").fillna(0)
        tmp = tmp.groupby(sku_col).agg(total_stock=(stock_col, "sum")).reset_index().sort_values("total_stock", ascending=False)
        tmp["cum_pct"] = tmp["total_stock"].cumsum() / tmp["total_stock"].sum()
        def abc(x):
            if x <= 0.7:
                return "A"
            if x <= 0.9:
                return "B"
            return "C"
        tmp["class"] = tmp["cum_pct"].apply(abc)
        # counts
        counts = tmp["class"].value_counts().to_dict()
        key_insight = f"ABC classification done. Counts: {counts}"
        top_rows = tmp.head(50).to_dict(orient="records")
        return {"key_insight": key_insight, "top_rows_table": top_rows, "abc_counts": counts}

    def _generic_insight(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic fallback: produce a short textual summary and top rows.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        summary = []
        if numeric_cols:
            c = numeric_cols[0]
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if not s.empty:
                summary.append(f"{c}: min={self._fmt_number(float(s.min()))}, median={self._fmt_number(float(s.median()))}, max={self._fmt_number(float(s.max()))}")
        if text_cols:
            t = text_cols[0]
            top = df[t].value_counts().head(3).to_dict()
            summary.append(f"Top {t}s: {', '.join([str(k) for k in top.keys()])}")
        preview = df.head(10).to_dict(orient="records")
        key_insight = " | ".join(summary) if summary else "Preview of top rows"
        return {"key_insight": key_insight, "top_rows_table": preview}

    def _llm_summarize(self, df: pd.DataFrame, plan: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to produce a short summary when deterministic summarization is not possible.
        """
        # Build a short prompt (avoid sending full df)
        prompt = "You are an analytics assistant. Produce a concise summary (one paragraph) describing the provided query results. If there is an error, mention it.\n\n"
        prompt += f"Query plan: {json.dumps(plan)}\n"
        if error:
            prompt += f"\nError encountered: {error}\n"
        prompt += "\nReturn a JSON object with keys: key_insight, top_rows_table (list of up to 5 rows), confidence.\n\nReturn JSON now."
        try:
            resp_text = self.llm.generate_text(prompt, max_tokens=256)
            # attempt to parse JSON
            start = resp_text.find("{")
            end = resp_text.rfind("}")
            if start != -1 and end != -1:
                blob = resp_text[start:end+1]
                parsed = json.loads(blob)
                return parsed
        except Exception:
            pass
        # fallback simple summary
        preview = df.head(5).to_dict(orient="records")
        return {"key_insight": "Unable to generate LLM summary. Returning preview.", "top_rows_table": preview, "confidence": "low"}

    # utilities
    def _safe_convert(self, v):
        if pd.isnull(v):
            return None
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        if isinstance(v, (np.int64, np.float64)):
            return float(v)
        if isinstance(v, (list, dict)):
            try:
                return json.dumps(v)
            except Exception:
                return str(v)
        return v

    def _fmt_number(self, v):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "N/A"
            if abs(v) >= 1_000_000:
                return f"{v:,.0f}"
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            return str(round(float(v), 2))
        except Exception:
            return str(v)

    def _make_answer_text(self, answer_obj: Optional[Dict[str, Any]], plan: Dict[str, Any], confidence: str = "low", notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Helper to construct the standard response shape if answer_obj is None or already formed.
        """
        if answer_obj is None:
            answer_obj = {"key_insight": "No data available for this query", "top_rows_table": []}
        return {"answer_text": answer_obj, "confidence": confidence, "notes": notes}
