# app.py
"""
Streamlit UI for Retail Insights Assistant (Full - with dropdown of default questions)
Run: streamlit run app.py
"""
import os
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

# Agents and core helpers (ensure these files exist as in the repo)
from core.duckdb_executor import DuckDBExecutor
from agents.lang_to_query import LanguageToQueryAgent
from agents.data_extractor import DataExtractionAgent
from agents.validator import ValidationAgent
from core.llm_client import LLMClient
from core.vector_store import SimpleVectorStore
from utils.schema_mapper import infer_schema_and_map
from core.summary_utils import generate_detailed_inventory_summary

# Config: default file (uploaded file in workspace)
# Developer note: this is the uploaded file path from the workspace
DEFAULT_CSV = "./data/Sale Report.csv"

st.set_page_config(layout="wide", page_title="Retail Insights Assistant")

st.title("Retail Insights Assistant — GenAI + Scalable Data System (MVP)")

# Sidebar config
st.sidebar.header("Configuration")
llm_provider = st.sidebar.selectbox("LLM Provider", ["openai", "mock"])
openai_api_key = st.sidebar.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
use_vector_store = st.sidebar.checkbox("Enable vector memory (in-memory)", value=False)

# Initialize LLM client
llm = LLMClient(provider=llm_provider, api_key=openai_api_key)

# Upload / dataset selection
st.header("1. Upload dataset or choose demo")
uploaded_file = st.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xlsx", "json"])
use_demo = st.checkbox("Use demo dataset (Sale Report.csv)", value=True)

if uploaded_file is None and use_demo:
    data_path = DEFAULT_CSV
    st.info(f"Using demo file: `{DEFAULT_CSV}`")
else:
    if uploaded_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tmp.write(uploaded_file.getvalue())
        tmp.close()
        data_path = tmp.name
        st.success(f"Uploaded file saved to `{data_path}`")
    else:
        st.stop()

# Preview for schema inference
try:
    if str(data_path).lower().endswith((".xls", ".xlsx")):
        df_preview = pd.read_excel(data_path, nrows=5)
    elif str(data_path).lower().endswith(".json"):
        # read small json safely
        df_preview = pd.read_json(data_path, lines=True) if os.path.getsize(data_path) < 10_000_000 else pd.read_json(data_path)
        df_preview = df_preview.head(5)
    else:
        df_preview = pd.read_csv(data_path, nrows=5)
except Exception as e:
    st.error(f"Unable to preview file: {e}")
    st.stop()

st.header("2. Schema inference & preview")
st.write("Preview (first 5 rows):")
st.dataframe(df_preview)

# Infer schema mapping and attach source path for agents
mapped_schema = infer_schema_and_map(data_path)
mapped_schema["__source_path__"] = data_path

st.write("Inferred schema mapping (edit if any field is incorrect):")
schema_json = st.text_area("Schema JSON", value=json.dumps(mapped_schema, indent=2), height=160)
try:
    schema_map = json.loads(schema_json)
except Exception:
    st.error("Invalid JSON for schema mapping. Using inferred mapping.")
    schema_map = mapped_schema

# Initialize modules
executor = DuckDBExecutor()
lang_agent = LanguageToQueryAgent(llm=llm)
data_agent = DataExtractionAgent(executor=executor)
validator = ValidationAgent(llm=llm)
vector_store = SimpleVectorStore() if use_vector_store else None

# Predefined default questions (10)
DEFAULT_QUESTIONS = [
    "Which category has the highest stock?",
    "List top 10 SKUs by stock quantity.",
    "How many unique SKUs are in inventory?",
    "Which categories have zero stock items?",
    "Show stock distribution summary (min, max, median, mean).",
    "Which sizes have the most SKUs?",
    "Which colors are most common in inventory?",
    "Which SKUs have the lowest stock (bottom 20)?",
    "How many items have missing category or color?",
    "Provide an ABC analysis suggestion based on stock quantity."
]

st.header("3. Modes")
mode = st.radio("Choose mode", ["Summarization Mode", "Conversational Q&A Mode"])

# ---------------------------
# Summarization Mode (detailed)
# ---------------------------
if mode == "Summarization Mode":
    st.subheader("Generate executive summary")
    if st.button("Generate Summary"):
        with st.spinner("Running detailed summarization pipeline..."):
            try:
                charts_dir = os.path.join(tempfile.gettempdir(), "retail_summary_charts")
                report = generate_detailed_inventory_summary(data_path, top_n=20, charts_dir=charts_dir)

                # Textual summary
                st.markdown("### Detailed Summary")
                st.write(report.get("summary_text"))

                # Key metrics
                st.markdown("### Key metrics")
                st.json(report.get("metrics"))

                # Missingness
                st.markdown("### Missing value counts")
                missing_df = pd.DataFrame.from_dict(report.get("missing_counts"), orient="index", columns=["missing_count"])
                st.dataframe(missing_df)

                # Stock statistics
                st.markdown("### Stock statistics")
                st.json(report.get("stock_stats"))

                # Top categories
                top_categories = report.get("top_categories")
                if isinstance(top_categories, pd.DataFrame) and not top_categories.empty:
                    st.markdown("### Top categories (by total stock)")
                    st.dataframe(top_categories.head(20))
                    if report.get("chart_paths", {}).get("top_categories"):
                        st.image(report["chart_paths"]["top_categories"], caption="Top categories by total stock")

                # Top SKUs
                top_skus = report.get("top_skus_by_stock")
                if isinstance(top_skus, pd.DataFrame) and not top_skus.empty:
                    st.markdown("### Top SKUs (by total stock)")
                    st.dataframe(top_skus.head(20))
                    if report.get("chart_paths", {}).get("top_skus"):
                        st.image(report["chart_paths"]["top_skus"], caption="Top SKUs by total stock")

                # Size distribution
                size_dist = report.get("size_distribution")
                if isinstance(size_dist, pd.DataFrame) and not size_dist.empty:
                    st.markdown("### Size distribution")
                    st.dataframe(size_dist)

                # Color distribution
                color_dist = report.get("color_distribution")
                if isinstance(color_dist, pd.DataFrame) and not color_dist.empty:
                    st.markdown("### Top colors")
                    st.dataframe(color_dist)

                # Stock histogram
                if report.get("chart_paths", {}).get("stock_hist"):
                    st.markdown("### Stock distribution (histogram)")
                    st.image(report["chart_paths"]["stock_hist"], caption="Stock distribution (histogram)")

                # raw preview
                st.markdown("### Data preview (first 50 rows)")
                st.dataframe(report.get("raw_df_preview"))

                # provenance
                st.markdown("**Provenance:** data read from")
                st.code(data_path)
            except Exception as exc:
                st.error(f"Detailed summarization failed: {exc}")

# ---------------------------
# Conversational Q&A Mode
# ---------------------------
elif mode == "Conversational Q&A Mode":
    st.subheader("Chat with your data")

    # Dropdown of default questions
    selected_q = st.selectbox("Select a question (or type your own below)", ["-- choose a default question --"] + DEFAULT_QUESTIONS)
    st.write("Or type your own question (typed text takes precedence over the dropdown):")
    user_input = st.text_input("Type a custom question", value="")

    # final question resolution: typed > selected
    if user_input and user_input.strip():
        question = user_input.strip()
    elif selected_q and selected_q != "-- choose a default question --":
        question = selected_q
    else:
        question = ""

    if st.button("Ask") and question:
        # store history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"role": "user", "content": question})
        st.write("**Question:**", question)

        # ensure schema_map has source path
        schema_map["__source_path__"] = data_path

        # 1) Language -> Query
        try:
            plan = lang_agent.parse(question, schema_map, conversation=st.session_state.history)
            st.write("**Generated query plan:**")
            st.json(plan)
        except Exception as e:
            st.error(f"Language-to-query agent failed: {e}")
            plan = {"intent": "unknown", "metrics": [], "dimensions": [], "filters": {}, "time_window": {}, "sql": None}

        # 2) Data extraction
        try:
            extraction = data_agent.execute_plan(plan, data_path)
            st.write("**Query results (top rows):**")
            if extraction.get("table_preview") is not None:
                st.dataframe(extraction.get("table_preview").head(10))
            else:
                st.write("No table preview available.")
        except Exception as e:
            st.error(f"Data extraction failed: {e}")
            extraction = {"sql": None, "rows_scanned": 0, "table": None}

        # 3) Validation & friendlier rendering
        try:
            validation = validator.validate(extraction, plan)
            st.markdown("**Answer (insight)**")

            answer_text = validation.get("answer_text", "")
            confidence = validation.get("confidence", "unknown")
            notes = validation.get("notes", None)

            parsed = None
            # normalize answer_text to dict if possible
            try:
                if isinstance(answer_text, dict):
                    parsed = answer_text
                else:
                    parsed = json.loads(answer_text)
            except Exception:
                parsed = None

            # Friendly rendering if structured
            if parsed and isinstance(parsed, dict):
                # display key insight first
                key_insight = parsed.get("key_insight") or parsed.get("summary") or parsed.get("insight") or parsed.get("message")
                if key_insight:
                    st.markdown(f"**Insight:** {key_insight}")
                else:
                    # fallback: print any top-level string fields
                    for k, v in parsed.items():
                        if isinstance(v, (str, int, float)) and k.lower() not in ("confidence", "notes"):
                            st.write(f"**{k}:** {v}")

                # render top_rows_table if present
                top_rows = parsed.get("top_rows_table") or parsed.get("top_rows") or parsed.get("table") or None
                if top_rows and isinstance(top_rows, list) and len(top_rows) > 0:
                    try:
                        tbl = pd.DataFrame(top_rows)
                        st.markdown("**Top results**")
                        st.dataframe(tbl)
                    except Exception:
                        st.write(top_rows)

                # confidence from parsed or validator
                conf_from_parsed = parsed.get("confidence")
                if conf_from_parsed:
                    st.markdown(f"**Confidence:** {conf_from_parsed}")
                else:
                    st.markdown(f"**Confidence:** {confidence}")

                if notes:
                    st.markdown("**Notes:**")
                    st.write(notes)
            else:
                # plain text answer
                st.write(answer_text)
                st.markdown(f"**Confidence:** {confidence}")
                if notes:
                    st.markdown("**Notes:**")
                    st.write(notes)

            # Show provenance (SQL executed) and rows scanned
            st.markdown("**Provenance**")
            executed_sql = extraction.get("sql")
            if executed_sql:
                # show SQL in a code block but not as raw JSON
                st.code(executed_sql)
            st.json({"rows_scanned": extraction.get("rows_scanned", 0)})
        except Exception as e:
            st.error(f"Validation/answering failed: {e}")

st.sidebar.header("Developer")
st.sidebar.markdown("Files & references:")
st.sidebar.write("- Demo file: `./data/Sale Report.csv`")
st.sidebar.write("- Other uploaded files available under `./data/`")
