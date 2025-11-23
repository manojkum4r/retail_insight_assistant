# agents/lang_to_query.py
"""
Language-to-Query Resolution Agent (fixed)
- Calls LLM for a strict JSON plan.
- If LLM fails or returns incomplete plan, create a robust fallback plan that includes a safe SQL_template.
"""
import json
import re
from typing import Dict, Any

from core.prompt_templates import TEMPLATE_LANG_TO_QUERY
from core.llm_client import LLMClient
from agents.data_extractor import DataExtractionAgent  # to build a safe SQL if needed

class LanguageToQueryAgent:
    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient(provider="mock")
        self._data_builder = DataExtractionAgent()

    def parse(self, question: str, schema_map: Dict[str, str], conversation: list = None) -> Dict[str, Any]:
        """
        Returns a plan dict with keys:
        - intent
        - metrics
        - dimensions
        - filters
        - time_window
        - sql (SQL string to execute) OR sql_template (preferred)
        Ensures 'sql' or 'sql_template' is ALWAYS present by building a safe SQL if LLM does not provide one.
        """
        plan = {"intent": "unknown", "metrics": [], "dimensions": [], "filters": {}, "time_window": {}, "sql": None, "sql_template": None}
        # Try LLM-based plan generation
        try:
            prompt = TEMPLATE_LANG_TO_QUERY.format(question=question, schema=json.dumps(schema_map))
            response = self.llm.generate_text(prompt)
            # Extract JSON object if present
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_blob = response[start:end+1]
                parsed = json.loads(json_blob)
                # normalize parsed plan
                for k in ("intent", "metrics", "dimensions", "filters", "time_window", "sql", "sql_template"):
                    if k in parsed:
                        plan[k] = parsed[k]
                # If sql_template is given as a string, accept it
                if plan.get("sql_template") and isinstance(plan.get("sql_template"), str):
                    plan["sql"] = plan["sql_template"]
        except Exception:
            # swallow LLM errors — will fallback below
            pass

        # If LLM did not produce any usable SQL, create a safe sql_template using schema_map
        if not plan.get("sql"):
            # Build a plan heuristically from the question
            q = question.lower()
            if "revenue" in q or "sales" in q or "amount" in q or "gross" in q:
                plan["intent"] = plan.get("intent","metric_query")
                plan["metrics"] = plan.get("metrics") or ["revenue"]
                # prefer category if schema_map has it, else sku
                dimension = schema_map.get("category") or schema_map.get("product_id") or schema_map.get("product_name")
                if dimension:
                    plan["dimensions"] = [dimension]
                else:
                    plan["dimensions"] = []
            elif "stock" in q or "inventory" in q or "stock level" in q:
                plan["intent"] = "inventory_query"
                plan["metrics"] = ["stock"]
                plan["dimensions"] = [schema_map.get("category") or schema_map.get("product_id") or next((v for v in schema_map.values() if v), None)]
            else:
                plan["intent"] = "top_n"
                plan["metrics"] = plan.get("metrics") or ["revenue"]
                plan["dimensions"] = plan.get("dimensions") or [schema_map.get("category") or schema_map.get("product_id") or next((v for v in schema_map.values() if v), None)]

            # now ask the DataExtractionAgent to build a safe SQL for this plan
            safe_sql = self._data_builder._build_sql(plan, schema_map.get("__source_path__") or "/mnt/data/Sale Report.csv")
            plan["sql"] = safe_sql
            plan["sql_template"] = safe_sql

        # Final normalization: ensure SQL placeholders exist (we will replace {path} in extraction)
        if plan.get("sql") and "{path}" not in plan.get("sql"):
            # If the sql includes read_csv_auto with a hard-coded path, that's OK.
            # But to make behavior predictable, if there's no {path}, we can wrap the SQL to use the file path.
            # We'll keep it as-is.
            pass

        return plan
