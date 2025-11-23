This project implements an intelligent **Retail Insights Assistant** capable of ingesting multi-source retail datasets, generating automated business summaries, and answering ad-hoc analytical questions through natural-language queries.

It demonstrates the use of **LLM-based agents**, **DuckDB**, **Streamlit**, and a unified multi-dataset data model, with an architecture designed to scale beyond 100GB+ of historical retail data.

---

## 🌟 Features

The system operates in two primary modes:

### 1. Summarization Mode (Auto-Analytics)
* Generates detailed **inventory or sales summaries**.
* Computes key **metrics**, **distributions**, and category/SKU-level insights.
* **Auto-detects dataset type** upon upload.
* Renders dynamic **charts**, **tables**, and key statistics.

### 2. Conversational Q&A Mode (Natural Language Analytics)
* Answers **natural-language queries** (e.g., "What was the total revenue last month?").
* Automatic **intent recognition** and LLM-generated **query plan**.
* **DuckDB** executes the resulting SQL with full **provenance**.
* LLM validates, formats, and generates the final, polished insight.
* Supports a dropdown of **default questions** and **free-form text queries**.

---

## 🤖 Multi-Agent Architecture

The core functionality is powered by three specialized, cooperative LLM-based agents:

| Agent | Role | Key Functions |
| :--- | :--- | :--- |
| **Language-to-Query Agent** | **Planner** | Converts user questions into structured **JSON plans** (metrics, dimensions, filters). Produces SQL templates using schema-mapping. |
| **Data Extraction Agent** | **Executor** | Builds **safe SQL** for DuckDB. Auto-repairs missing/incorrect columns. Executes the query and returns dataframes. Prevents binder errors. |
| **Validation Agent** | **Summarizer** | Runs **data sanity checks** (negative/missing values). Generates the final, polished insight with a **confidence score** and explanation. |



---

## 🗄️ Dataset Coverage and Canonical Model

The system unifies multiple diverse retail CSVs into a coherent, queryable data model using DuckDB.

### Unified Datasets

| File | Role |
| :--- | :--- |
| `Amazon Sale Report.csv` | Transactional sales (revenue, qty, dates) |
| `International sale Report.csv` | Transactional sales (revenue, qty) |
| `Sale Report.csv` | Inventory (stock, category, size, color) |
| `May-2022.csv` | SKU pricing master |
| `P L March 2021.csv` | Product master / pricing |
| `Expense IIGF.csv` | Expense ledger |
| `Cloud Warehouse Comparison.csv` | Metadata / reference |

### Canonical Data Model

The raw data is normalized into four core virtual tables:

* `sales_transactions`: `date`, `sku`, `category`, `qty`, `revenue`, `source`
* `inventory`: `sku`, `stock`, `category`, `size`, `color`
* `product_master`: `sku`, `mrp`, `tp`, `category`
* `expenses`: `date`, `amount`, `type`

---

## 🛠️ Project Structure

retail_insights_assistant/ │ ├── app.py # Streamlit UI interface │ ├── agents/ │ ├── lang_to_query.py # LLM agent for query planning (JSON) │ ├── data_extractor.py # SQL builder + DuckDB execution interface │ └── validator.py # LLM agent for validation and summarization │ ├── core/ │ ├── duckdb_executor.py │ ├── llm_client.py │ ├── summary_utils.py # Detailed summary generation logic │ ├── vector_store.py │ └── prompt_templates.py # All LLM prompts │ └── utils/ └── schema_mapper.py # Auto-detection and mapping of column names


---

## ⚙️ Installation & Setup

### Prerequisites
* **Python 3.9+**
* `pip`
* OpenAI API key (Optional; a mock LLM works offline)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
2. Set API Key (Optional)
Create a file named .env in the project root and add your OpenAI key:

Ini, TOML

OPENAI_API_KEY=your_key_here
3. Run the Application
Bash

streamlit run app.py
Access
Open your web browser and navigate to: http://localhost:8501

🚀 Usage Guide
A) Summarization Mode
Upload any supported retail CSV file (sales, inventory, pricing, or expense).

The system will auto-detect the file type.

The output includes:

Key metrics

Missing value analysis

Category and SKU distributions

Charts (bar charts, histograms, etc.)

B) Conversational Q&A Mode
Use the default dropdown questions or type your own free-form query.

Default Questions Included (Inventory Focus):

Which category has the highest stock?

List top 10 SKUs by stock quantity.

Provide an ABC analysis suggestion based on stock quantity.

Which categories have zero stock items?

Show stock distribution summary (min, max, median, mean).

Sample Output: | Element | Description | | :--- | :--- | | Key Insight | Category: Kurta had the highest revenue. | | Detailed Table | Tabular results matching the query. | | Confidence | High | | SQL Provenance | The exact SQL query executed by DuckDB. |

⚖️ Scaling to 100GB+ Architecture
The system is designed with a roadmap for big data scalability by leveraging best-in-class data engineering practices:

Data Engineering: Use PySpark/Databricks for ingestion, converting raw CSVs to partitioned Parquet.

Storage: S3/ADLS/GCS Data Lake utilizing Delta Lake or Iceberg for ACID transactions.

Indexing & Retrieval: Metadata indexing, Vector search (for semantic filters/RAG), and Pre-aggregated summary tables.

Compute Engine: DuckDB / Trino with Pushdown filters and Parquet partition skipping for massive data reduction.

Performance & Cost Optimization
Hybrid LLM Strategy: Small model for parsing, large model for summarization.

Caching: Prompt and SQL result caching (e.g., Redis).

Optimization: Column pruning, projection pushdown, and minimized token usage via structured prompts.

📢 Additional Project: Voice Assistant
A separate, complementary repository demonstrating multimodal, real-time AI agent capability:

Voice Assistant Repository: https://github.com/manojkum4r/voice_assistant

Capabilities include Real-time STT, TTS, LLM, Telephony/WebRTC connectors, and multi-agent pipelines in an event-driven architecture.

📄 License
This project is released for assessment and demonstration purposes only. All data used is sample or anonymized retail data.