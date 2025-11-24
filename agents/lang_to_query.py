# agents/lang_to_query.py
"""
Language-to-Query Resolution Agent (deterministic mapping for default questions + LLM fallback)

- If question matches one of the known default dropdown questions, return a deterministic plan.
- Otherwise ask the LLM to produce a JSON plan and parse it.
- Ensures the returned plan contains at least 'intent', 'metrics', 'dimensions', 'filters', 'time_window', and optionally 'sql_template' or 'sql'.
"""

import json
import re
from typing import Dict, Any, Optional

from core.prompt_templates import TEMPLATE_LANG_TO_QUERY
from core.llm_client import LLMClient

# Default question -> structured plan mapping (these correspond to the dropdown defaults)
# Plans are intentionally simple: they indicate intent, metrics, dimensions and filters.
# DataExtractionAgent will use these to build safe SQL.
DEFAULT_QUESTION_MAP = {
    "Which category has the highest stock?": {
        "intent": "top_n",
        "metrics": ["stock"],
        "dimensions": ["category"],
        "filters": {}
    },
    "List top 10 SKUs by stock quantity.": {
        "intent": "top_n",
        "metrics": ["stock"],
        "dimensions": ["product_id"],
        "filters": {},
        "limit": 10
    },
    "How many unique SKUs are in inventory?": {
        "intent": "count_unique",
        "metrics": ["count"],
        "dimensions": ["product_id"],
        "filters": {}
    },
    "Which categories have zero stock items?": {
        "intent": "filter",
        "metrics": ["stock"],
        "dimensions": ["category"],
        "filters": {"stock": {"op": "eq", "value": 0}}
    },
    "Show stock distribution summary (min, max, median, mean).": {
        "intent": "aggregate_stats",
        "metrics": ["stock"],
        "dimensions": [],
        "filters": {}
    },
    "Which sizes have the most SKUs?": {
        "intent": "top_n",
        "metrics": ["count"],
        "dimensions": ["size"],
        "filters": {}
    },
    "Which colors are most common in inventory?": {
        "intent": "top_n",
        "metrics": ["count"],
        "dimensions": ["color"],
        "filters": {}
    },
    "Which SKUs have the lowest stock (bottom 20)?": {
        "intent": "top_n",
        "metrics": ["stock"],
        "dimensions": ["product_id"],
        "filters": {},
        "order": "asc",
        "limit": 20
    },
    "How many items have missing category or color?": {
        "intent": "missing_count",
        "metrics": ["missing"],
        "dimensions": ["category", "color"],
        "filters": {}
    },
    "Provide an ABC analysis suggestion based on stock quantity.": {
        "intent": "abc_analysis",
        "metrics": ["stock"],
        "dimensions": ["product_id"],
        "filters": {}
    }
}


class LanguageToQueryAgent:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(provider="mock")

    def parse(self, question: str, schema_map: Dict[str, str], conversation: list = None) -> Dict[str, Any]:
        """
        Returns a plan dict with keys:
        - intent
        - metrics
        - dimensions
        - filters
        - time_window
        - optionally: sql or sql_template

        If the question exactly matches a default known question, return its deterministic plan.
        Otherwise, call the LLM to generate a plan.
        """
        question_clean = question.strip()

        # 1) If question matches a default dropdown question, return deterministic plan
        if question_clean in DEFAULT_QUESTION_MAP:
            plan = dict(DEFAULT_QUESTION_MAP[question_clean])  # shallow copy
            # Normalize dimensions to actual schema column names (if available)
            plan = self._map_plan_columns_to_schema(plan, schema_map)
            # Ensure mandatory fields
            plan.setdefault("intent", "unknown")
            plan.setdefault("metrics", [])
            plan.setdefault("dimensions", [])
            plan.setdefault("filters", {})
            plan.setdefault("time_window", {})
            # No sql yet — DataExtractionAgent will build safe SQL from this plan
            return plan

        # 2) Fallback to LLM-based plan generation
        try:
            prompt = TEMPLATE_LANG_TO_QUERY.format(question=question, schema=json.dumps(schema_map))
            response = self.llm.generate_text(prompt)
            json_blob = self._extract_json(response)
            parsed = json.loads(json_blob)
            plan = {}
            plan["intent"] = parsed.get("intent", "unknown")
            plan["metrics"] = parsed.get("metrics", [])
            plan["dimensions"] = parsed.get("dimensions", [])
            plan["filters"] = parsed.get("filters", {})
            plan["time_window"] = parsed.get("time_window", {})
            # Accept sql_template or sql if LLM provided it
            if parsed.get("sql_template"):
                plan["sql"] = parsed.get("sql_template")
                plan["sql_template"] = parsed.get("sql_template")
            elif parsed.get("sql"):
                plan["sql"] = parsed.get("sql")
            # Normalize columns
            plan = self._map_plan_columns_to_schema(plan, schema_map)
            return plan
        except Exception:
            # If LLM parsing fails, return a conservative fallback plan using heuristics
            return self._fallback_plan(question, schema_map)

    def _extract_json(self, text: str) -> str:
        # extract first JSON object from text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON found in LLM response")
        return text[start:end+1]

    def _fallback_plan(self, question: str, schema_map: Dict[str, str]) -> Dict[str, Any]:
        # Rule-based fallback (simple heuristics)
        q = question.lower()
        plan = {"intent": "unknown", "metrics": [], "dimensions": [], "filters": {}, "time_window": {}}
        if "revenue" in q or "sales" in q:
            plan["intent"] = "metric_query"
            plan["metrics"] = ["revenue"]
            plan["dimensions"] = [schema_map.get("category") or schema_map.get("product_id") or schema_map.get("product_name")]
        elif "stock" in q or "inventory" in q:
            plan["intent"] = "metric_query"
            plan["metrics"] = ["stock"]
            plan["dimensions"] = [schema_map.get("category") or schema_map.get("product_id")]
        else:
            plan["intent"] = "top_n"
            plan["metrics"] = ["revenue"]
            plan["dimensions"] = [schema_map.get("category") or schema_map.get("product_id")]
        # Map to actual schema column names
        plan = self._map_plan_columns_to_schema(plan, schema_map)
        return plan

    def _map_plan_columns_to_schema(self, plan: Dict[str, Any], schema_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Replace any logical names in plan['dimensions'] and plan['metrics'] with actual column names
        based on schema_map. If schema_map doesn't have the mapping, keep the original name.
        """
        # map metrics if needed (e.g., 'stock' -> actual column)
        mapped_metrics = []
        for m in plan.get("metrics", []):
            if not m:
                continue
            key = m.lower()
            replacement = None
            # Common metric logical names -> schema keys
            if key in ("stock", "units", "quantity", "qty"):
                replacement = schema_map.get("units_sold") or schema_map.get("units") or schema_map.get("inventory") or schema_map.get("stock")
            if key in ("revenue", "sales", "amount"):
                replacement = schema_map.get("revenue") or schema_map.get("amount")
            if key in ("count", "unique", "distinct"):
                replacement = None  # count doesn't map to a single column
            # fallback: if we got a direct mapping in schema_map
            if not replacement and schema_map.get(m):
                replacement = schema_map.get(m)
            mapped_metrics.append(replacement or m)
        plan["metrics"] = mapped_metrics

        # map dimensions
        mapped_dims = []
        for d in plan.get("dimensions", []):
            if not d:
                continue
            # Try to map common logical dimension names to schema_map
            key = d.lower()
            replacement = None
            if key in ("category", "product_category"):
                replacement = schema_map.get("category")
            if key in ("product_id", "sku", "sku code", "sku_code"):
                replacement = schema_map.get("product_id") or schema_map.get("product_id")  # preserve
            if key in ("size",):
                replacement = schema_map.get("size")
            if key in ("color",):
                replacement = schema_map.get("color")
            # fallback direct mapping
            if not replacement and schema_map.get(d):
                replacement = schema_map.get(d)
            mapped_dims.append(replacement or d)
        plan["dimensions"] = mapped_dims
        return plan
