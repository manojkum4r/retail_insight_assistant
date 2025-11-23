# agents/validator.py
"""
Validation Agent
Performs sanity checks, compares deterministic numbers, and generates the final answer text.
"""
import pandas as pd
from core.llm_client import LLMClient

class ValidationAgent:
    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient(provider="mock")

    def validate(self, extraction: dict, plan: dict) -> dict:
        """
        Returns dict with:
        - answer_text: final LLM crafted answer (or deterministic fallback)
        - confidence: "high" | "medium" | "low"
        - notes
        """
        df = extraction.get("table")
        rows = extraction.get("rows_scanned", 0)

        if df is None:
            return {"answer_text": "No data returned for this query.", "confidence": "low", "notes": "no table returned"}

        if rows == 0 or df.empty:
            text = "I could not find any matching data for the requested query. Please refine the filters or check the dataset."
            return {"answer_text": text, "confidence": "low", "notes": "no data matched"}

        # Basic numeric sanity checks - if revenue column present
        confidence = "high"
        notes = []
        rev_cols = [c for c in df.columns if "revenue" in c.lower() or "amount" in c.lower()]
        if rev_cols:
            for rc in rev_cols:
                try:
                    if (pd.to_numeric(df[rc], errors="coerce") < 0).any():
                        notes.append("Negative revenue values detected.")
                        confidence = "medium"
                except Exception:
                    notes.append("Could not coerce revenue to numeric.")
                    confidence = "medium"

        # Prepare sample data for LLM prompt
        sample = df.head(5).to_dict(orient="records")
        # Compose answer using LLM for friendly language, but include deterministic numbers
        prompt = f"""
You are an analytics assistant. The user asked: {plan.get('intent')}. 
Here are the top result rows (JSON): {sample}.
Provide a concise 2-3 sentence answer describing the key insight and list the top 3 rows as a short table. 
Also include a one-line confidence based on data checks: {notes if notes else 'no issues detected'}.
"""
        try:
            answer = self.llm.generate_text(prompt)
        except Exception:
            # Fallback deterministic answer
            # If there's a revenue column, show top row
            top_text = ""
            if rev_cols:
                rc = rev_cols[0]
                top = df.sort_values(by=rc, ascending=False).head(1)
                top_text = f"Top: {top.iloc[0].to_dict()}"
            answer = f"Result summary. {top_text}"

        return {"answer_text": answer, "confidence": confidence, "notes": notes}
