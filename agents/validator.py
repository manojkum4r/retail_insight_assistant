# agents/validator.py
"""
Validation Agent — natural-language answer generation

This validator:
- Uses the DataExtractionAgent's returned dataframe (extraction["table"] or table_preview)
- Produces a natural-language `key_insight` for top_n and aggregate queries (one-line human-friendly)
- Returns structured output:
    { "answer_text": { "key_insight": str, "top_rows_table": [...], ... }, "confidence": str, "notes": ... }

Replace your existing agents/validator.py with this file and restart the app.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import math
import json

from core.llm_client import LLMClient

class ValidationAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(provider="mock")

    def validate(self, extraction: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer full table, then preview, then raw records
        df = None
        if extraction is None:
            return {"answer_text": {"key_insight": "No data returned from query."}, "confidence": "low", "notes": "No extraction object"}

        if isinstance(extraction.get("table"), pd.DataFrame):
            df = extraction.get("table")
        elif isinstance(extraction.get("table_preview"), pd.DataFrame):
            df = extraction.get("table_preview")
        else:
            raw = extraction.get("raw")
            if isinstance(raw, list) and raw:
                try:
                    df = pd.DataFrame(raw)
                except Exception:
                    df = None

        if df is None or df.empty:
            notes = "Query returned no rows" if extraction.get("rows_scanned") == 0 else "No table available"
            return {"answer_text": {"key_insight": "No results for the given query."}, "confidence": "low", "notes": notes}

        # Normalize column labels
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

        intent = (plan.get("intent") or "").lower() if plan else ""
        try:
            # If top_n-like intent or metrics indicate top-n
            if intent in ("top_n", "top", "topk", "topk_n") or self._looks_like_topn(plan):
                ans = self._nl_insight_topn(df, plan)
                return {"answer_text": ans, "confidence": "high", "notes": None}
            # If aggregate/statistics
            if intent in ("aggregate_stats", "aggregate", "metric_query", "summary"):
                ans = self._nl_insight_aggregate(df, plan)
                return {"answer_text": ans, "confidence": "high", "notes": None}
            if intent in ("missing_count",):
                ans = self._nl_insight_missing(df, plan)
                return {"answer_text": ans, "confidence": "high", "notes": None}
            if intent in ("abc_analysis",):
                ans = self._nl_insight_abc(df, plan)
                return {"answer_text": ans, "confidence": "high", "notes": None}

            # Fallback: generic natural-language summary + preview
            ans = self._generic_nl(df, plan)
            return {"answer_text": ans, "confidence": "medium", "notes": None}
        except Exception as e:
            # LLM fallback to produce a textual summary if deterministic logic fails
            short = self._llm_summarize(df, plan, error=str(e))
            return {"answer_text": short, "confidence": "low", "notes": f"Validation error: {e}"}

    # -----------------------
    # Heuristics
    # -----------------------
    def _looks_like_topn(self, plan: Dict[str, Any]) -> bool:
        if not plan:
            return False
        metrics = plan.get("metrics", []) or []
        dims = plan.get("dimensions", []) or []
        # top-n often has a single dimension and a metric like count/stock/revenue
        if len(dims) >= 1 and metrics:
            return True
        # explicit intent key
        intent = (plan.get("intent") or "").lower()
        return intent.startswith("top") or intent == "top_n"

    # -----------------------
    # Natural-language top-n insight
    # -----------------------
    def _nl_insight_topn(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        # Determine dimension and metric columns
        dims = [d for d in (plan.get("dimensions") or []) if d]
        metrics = [m for m in (plan.get("metrics") or []) if m]
        dim_col = None
        metric_col = None

        # Prefer explicit columns if present in df
        if dims:
            for d in dims:
                if d in df.columns:
                    dim_col = d
                    break
        if not dim_col:
            # Pick a likely categorical column
            for cand in ["color", "Color", "category", "Category", "SKU", "sku", "SKU Code", "product_id"]:
                if cand in df.columns:
                    dim_col = cand
                    break

        # Find metric col by looking for common aliases
        possible_metrics = ["count", "total_stock", "total_revenue", "count_value", "total", "value"]
        if metrics:
            for m in metrics:
                if m in df.columns:
                    metric_col = m
                    break
                # sometimes the SQL aliases use different names; check common aliases
                for pm in possible_metrics:
                    if pm in df.columns:
                        metric_col = pm
                        break
                if metric_col:
                    break
        else:
            # find numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                metric_col = numeric_cols[0]

        # If metric absent but df contains counts by dim, use value_counts
        top_rows = []
        key_insight = ""
        try:
            if metric_col and metric_col in df.columns and dim_col and dim_col in df.columns:
                sorted_df = df.sort_values(by=metric_col, ascending=False).head(10)
                total_metric = float(df[metric_col].sum()) if df[metric_col].dtype.kind in 'fi' else None

                # build top 3 summary sentence
                topk = sorted_df.head(3)
                parts = []
                for _, r in topk.iterrows():
                    v = self._fmt_number(r[metric_col])
                    parts.append(f"{r[dim_col]} ({v})")
                if total_metric:
                    top_pct = sum([float(r[metric_col]) for _, r in topk.iterrows()]) / total_metric
                    pct_text = f", representing {round(top_pct*100,1)}% of total"
                else:
                    pct_text = ""
                key_insight = f"Top {dim_col} by {metric_col}: " + ", ".join(parts) + pct_text + "."
                # build table
                for _, r in sorted_df.iterrows():
                    rowd = {c: self._safe_convert(r.get(c)) for c in sorted_df.columns}
                    top_rows.append(rowd)
                return {"key_insight": key_insight, "top_rows_table": top_rows}
            elif dim_col and dim_col in df.columns:
                # Use value_counts for the dimension
                vc = df[dim_col].value_counts().head(10)
                total = int(vc.sum())
                parts = []
                for idx, cnt in vc.items():
                    parts.append(f"{idx} ({cnt})")
                    top_rows.append({dim_col: idx, "count": int(cnt)})
                key_insight = f"Top {dim_col} by frequency: " + ", ".join(parts[:3]) + f". Total counted: {total}."
                return {"key_insight": key_insight, "top_rows_table": top_rows}
            else:
                # fallback preview
                preview = df.head(5).to_dict(orient="records")
                return {"key_insight": "Preview of top rows for the query.", "top_rows_table": preview}
        except Exception as e:
            # fallback
            preview = df.head(5).to_dict(orient="records")
            return {"key_insight": "Preview of top rows for the query.", "top_rows_table": preview, "notes": str(e)}

    # -----------------------
    # Natural-language aggregates insight
    # -----------------------
    def _nl_insight_aggregate(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        # pick a numeric column (prefer requested metric)
        metrics = plan.get("metrics") or []
        target = None
        if metrics:
            for m in metrics:
                if m in df.columns:
                    target = m
                    break
        if not target:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target = numeric_cols[0]
        if not target:
            return {"key_insight": "No numeric column available to summarize.", "top_rows_table": []}
        series = pd.to_numeric(df[target], errors="coerce").dropna()
        if series.empty:
            return {"key_insight": f"No numeric data available in column {target}.", "top_rows_table": []}
        stats = {
            "min": float(series.min()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "max": float(series.max()),
            "count": int(series.count())
        }
        key_insight = f"{target}: min {self._fmt_number(stats['min'])}, median {self._fmt_number(stats['median'])}, mean {self._fmt_number(stats['mean'])}, max {self._fmt_number(stats['max'])}."
        return {"key_insight": key_insight, "aggregates": stats, "top_rows_table": []}

    # -----------------------
    # Missing values insight
    # -----------------------
    def _nl_insight_missing(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        dims = plan.get("dimensions") or []
        if not dims:
            dims = df.select_dtypes(include=['object']).columns.tolist()
        missing = {}
        for d in dims:
            if d in df.columns:
                missing[d] = int(df[d].isnull().sum())
        summary = ", ".join([f"{k}: {v}" for k, v in missing.items()])
        key_insight = f"Missing counts — {summary}."
        return {"key_insight": key_insight, "missing_counts": missing, "top_rows_table": []}

    # -----------------------
    # ABC analysis insight
    # -----------------------
    def _nl_insight_abc(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        # Find sku and stock-like column
        sku_col = None
        stock_col = None
        for c in df.columns:
            lc = str(c).lower()
            if "sku" in lc or "product" in lc:
                sku_col = c
            if "stock" in lc or "qty" in lc or "quantity" in lc:
                stock_col = c
        if not sku_col or not stock_col:
            return {"key_insight": "Required columns for ABC analysis not found.", "top_rows_table": []}
        tmp = df[[sku_col, stock_col]].copy()
        tmp[stock_col] = pd.to_numeric(tmp[stock_col], errors="coerce").fillna(0)
        tmp = tmp.groupby(sku_col).agg(total_stock=(stock_col, "sum")).reset_index().sort_values("total_stock", ascending=False)
        tmp["cum_pct"] = tmp["total_stock"].cumsum() / tmp["total_stock"].sum()
        def cls(x):
            if x <= 0.7:
                return "A"
            if x <= 0.9:
                return "B"
            return "C"
        tmp["class"] = tmp["cum_pct"].apply(cls)
        counts = tmp["class"].value_counts().to_dict()
        key_insight = f"ABC classification complete. Counts: {counts}."
        top_rows = tmp.head(50).to_dict(orient="records")
        return {"key_insight": key_insight, "top_rows_table": top_rows, "abc_counts": counts}

    # -----------------------
    # Generic natural-language fallback
    # -----------------------
    def _generic_nl(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        summary_parts = []
        if numeric_cols:
            c = numeric_cols[0]
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if not s.empty:
                summary_parts.append(f"{c}: median {self._fmt_number(float(s.median()))}, max {self._fmt_number(float(s.max()))}")
        if text_cols:
            t = text_cols[0]
            top = df[t].value_counts().head(3).to_dict()
            summary_parts.append(f"Top {t}s: {', '.join([str(k) for k in top.keys()])}")
        key_insight = " ; ".join(summary_parts) if summary_parts else "Preview of query results."
        preview = df.head(5).to_dict(orient="records")
        return {"key_insight": key_insight, "top_rows_table": preview}

    # -----------------------
    # LLM fallback summary (rare)
    # -----------------------
    def _llm_summarize(self, df: pd.DataFrame, plan: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
        prompt = "You are an analytics assistant. Provide a concise one-line insight describing the query results. Return JSON with keys: key_insight, top_rows_table (max 5 rows), confidence."
        try:
            resp_text = self.llm.generate_text(prompt, max_tokens=200)
            start = resp_text.find("{")
            end = resp_text.rfind("}")
            if start != -1 and end != -1:
                blob = resp_text[start:end+1]
                parsed = json.loads(blob)
                return parsed
        except Exception:
            pass
        # fallback preview
        preview = df.head(5).to_dict(orient="records")
        return {"key_insight": "Preview of top rows", "top_rows_table": preview, "confidence": "low"}

    # -----------------------
    # Utilities
    # -----------------------
    def _safe_convert(self, v):
        try:
            if pd.isnull(v):
                return None
        except Exception:
            pass
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
            if isinstance(v, (int, float)) and abs(v) >= 1000:
                return f"{v:,.0f}"
            if isinstance(v, float):
                return str(round(v, 2))
            return str(v)
        except Exception:
            return str(v)
