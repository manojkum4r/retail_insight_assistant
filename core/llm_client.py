# core/llm_client.py
"""
Minimal LLM client abstraction updated for openai>=1.0.0 compatibility.

- Uses the Chat Completions API when available (openai.ChatCompletion.create).
- Falls back to the older Completion API if Chat is not available (older openai versions).
- Also supports a 'mock' provider for offline demos.

Usage:
    llm = LLMClient(provider="openai", api_key=os.getenv("OPENAI_API_KEY"))
    text = llm.generate_text("Summarize these metrics...")
"""
import os
import json
from typing import Optional

class LLMClient:
    def __init__(self, provider: str = "mock", api_key: Optional[str] = None, model: str = None):
        self.provider = (provider or "mock").lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", None)
        self.model = model or os.getenv("LLM_MODEL", None)
        self._openai = None
        if self.provider == "openai":
            try:
                import openai as _openai
                # set API key if provided
                if self.api_key:
                    # support both env var and direct assignment
                    _openai.api_key = self.api_key
                self._openai = _openai
            except Exception as e:
                # If import fails, fall back to mock
                print("OpenAI package import failed or not available:", e)
                self.provider = "mock"

        # Default model choices (safe, user can override by env LLM_MODEL)
        if self.provider == "openai" and not self.model:
            # use a chat-capable model name by default; user can override
            # Note: change this to a model you have access to (gpt-4o, gpt-4o-mini, gpt-4o-mini-2, gpt-3.5-turbo, etc.)
            self.model = "gpt-3.5-turbo"

    def generate_text(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """
        Generate text for a given prompt. Uses Chat API if present (openai.ChatCompletion).
        Returns the text string (trimmed).
        """
        if self.provider == "openai" and self._openai is not None:
            # Prefer Chat API (openai.ChatCompletion.create)
            try:
                # Newer OpenAI SDKs: ChatCompletion API
                # We pass the prompt as a single user message
                resp = self._openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful analytics assistant. Be concise and return JSON only when instructed."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1
                )
                # New Chat API responses contain choices[0].message.content
                text = None
                if resp and "choices" in resp and len(resp.choices) > 0:
                    choice = resp.choices[0]
                    # Chat style
                    msg = getattr(choice, "message", None) or choice.get("message") if isinstance(choice, dict) else None
                    if msg:
                        text = msg.get("content") if isinstance(msg, dict) else msg.content
                    else:
                        # older shapes may have text directly
                        text = choice.get("text") if isinstance(choice, dict) else getattr(choice, "text", None)
                if text is None:
                    # last resort: try stringifying top-level text
                    text = str(resp)
                return text.strip()
            except AttributeError:
                # If ChatCompletion not present in this openai version, fall through to older Completion API
                pass
            except Exception as e:
                # Bubble up more helpful error
                raise RuntimeError(f"OpenAI ChatCompletion call failed: {e}")

            # Fallback to older Completion API if ChatCompletion isn't available
            try:
                resp = self._openai.Completion.create(
                    engine=getattr(self, "model", "text-davinci-003"),
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1
                )
                # Older completions return text in choices[0].text
                text = resp.choices[0].text.strip()
                return text
            except Exception as e:
                raise RuntimeError(f"OpenAI Completion fallback failed: {e}")

        # Mock deterministic behaviour for offline/demo
        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        # Heuristic mock responses for different prompt types for UI/demo purposes.
        # This is intentionally simple — replace with real OpenAI when available.
        try:
            low = prompt.lower()
            if "return strict json" in low or "produce the json plan" in low or "produce the json plan now" in low:
                return json.dumps({
                    "intent": "top_n",
                    "metrics": ["revenue"],
                    "dimensions": ["category"],
                    "filters": {},
                    "time_window": {},
                    "sql_template": "SELECT category, SUM(revenue) as revenue FROM read_csv_auto('{path}') GROUP BY category ORDER BY revenue DESC LIMIT 10"
                })
            if "write a concise" in low or "executive summary" in low:
                return "Sales rose during the analyzed period. Top categories contributing to growth were X and Y. Please see the table for numeric details."
            if "produce a concise answer" in low or "provide a concise" in low:
                return "Top category is X with Y revenue. See table for details."
        except Exception:
            pass
        return "Mock LLM response. To use a real LLM, set OPENAI_API_KEY and choose provider 'openai' in the UI."
