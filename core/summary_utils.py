# core/summary_utils.py
"""
Detailed inventory & transactional summary utilities.

Functions:
- generate_detailed_inventory_summary(path, top_n=20, charts_dir="/tmp/retail_summary_charts")
  -> returns a dict with textual summary, metrics, dataframes (pandas) and saved chart file paths.
"""

import os
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _safe_read_csv(path):
    # Read CSV robustly (small sample then full)
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception:
        # Try common alternatives
        try:
            df = pd.read_csv(path, encoding='latin1', low_memory=False)
            return df
        except Exception:
            raise

def generate_detailed_inventory_summary(path: str, top_n: int = 20, charts_dir: str = "/tmp/retail_summary_charts") -> Dict[str, Any]:
    """
    Generate a detailed inventory summary for a file that looks like Sale Report.csv.
    Returns:
      {
        "summary_text": str,
        "metrics": dict,
        "missing_table": pd.DataFrame,
        "top_categories": pd.DataFrame,
        "top_skus_by_stock": pd.DataFrame,
        "stock_stats": dict,
        "size_distribution": pd.DataFrame,
        "color_distribution": pd.DataFrame,
        "chart_paths": { "top_categories": path, "stock_hist": path, "top_skus": path }
      }
    """
    charts_dir = _ensure_dir(charts_dir)
    df = _safe_read_csv(path)

    # Normalize columns: strip whitespace in headers
    df.columns = [c.strip() for c in df.columns]

    # Identify common column names (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(*candidates):
        for cand in candidates:
            key = cand.lower()
            if key in cols_lower:
                return cols_lower[key]
        return None

    sku_col = pick("sku code", "sku", "product_id", "product id", "article code")
    stock_col = pick("stock", "qty", "quantity", "pcs", "available")
    category_col = pick("category", "product_category")
    size_col = pick("size")
    color_col = pick("color")

    # Basic metrics
    num_rows = len(df)
    unique_skus = df[sku_col].nunique() if sku_col and sku_col in df.columns else None
    columns = df.columns.tolist()

    # Missingness
    missing_counts = df.isnull().sum().to_dict()

    # Top categories by item count (if category col present)
    if category_col and category_col in df.columns:
        top_categories = (df.groupby(category_col)
                            .agg(item_count=(sku_col if sku_col and sku_col in df.columns else df.columns[0], "count"),
                                 total_stock=(stock_col, lambda x: pd.to_numeric(x, errors="coerce").sum()))
                            .fillna(0)
                            .sort_values(by="total_stock", ascending=False)
                            .reset_index()
                         )
        # Clean columns in case they didn't exist
        if "total_stock" not in top_categories.columns:
            top_categories["total_stock"] = 0
    else:
        top_categories = pd.DataFrame(columns=[category_col or "category", "item_count", "total_stock"])

    # Stock statistics
    stock_series = None
    if stock_col and stock_col in df.columns:
        # coerce to numeric
        stock_series = pd.to_numeric(df[stock_col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors="coerce")
        stock_stats = {
            "min": float(np.nanmin(stock_series)) if not stock_series.isna().all() else None,
            "max": float(np.nanmax(stock_series)) if not stock_series.isna().all() else None,
            "mean": float(np.nanmean(stock_series)) if not stock_series.isna().all() else None,
            "median": float(np.nanmedian(stock_series)) if not stock_series.isna().all() else None,
            "q1": float(np.nanpercentile(stock_series.dropna(), 25)) if not stock_series.isna().all() else None,
            "q3": float(np.nanpercentile(stock_series.dropna(), 75)) if not stock_series.isna().all() else None,
            "zero_stock_count": int((stock_series == 0).sum()) if not stock_series.isna().all() else 0,
            "na_count": int(stock_series.isna().sum())
        }
    else:
        stock_stats = {}

    # Top SKUs by stock (if sku_col)
    if sku_col and sku_col in df.columns and stock_series is not None:
        tmp = df[[sku_col, stock_col]].copy()
        tmp["stock_num"] = pd.to_numeric(tmp[stock_col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors="coerce")
        top_skus = (tmp.groupby(sku_col)
                      .agg(total_stock=("stock_num", "sum"), rows=(sku_col, "count"))
                      .reset_index()
                      .sort_values(by="total_stock", ascending=False)
                      .head(top_n)
                   )
    else:
        top_skus = pd.DataFrame(columns=[sku_col or "sku", "total_stock", "rows"])

    # Size distribution
    if size_col and size_col in df.columns:
        size_dist = (df[size_col].astype(str).fillna("Unknown")
                      .value_counts()
                      .rename_axis("size")
                      .reset_index(name="count"))
    else:
        size_dist = pd.DataFrame()

    # Color distribution
    if color_col and color_col in df.columns:
        color_dist = (df[color_col].astype(str).fillna("Unknown")
                       .value_counts()
                       .rename_axis("color")
                       .reset_index(name="count")
                      ).head(top_n)
    else:
        color_dist = pd.DataFrame()

    # Charts
    chart_paths = {}
    # Top categories bar chart
    try:
        if not top_categories.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top_categories[category_col].astype(str).head(top_n)[::-1], top_categories["total_stock"].head(top_n)[::-1])
            ax.set_xlabel("Total Stock")
            ax.set_title("Top Categories by Total Stock")
            p = os.path.join(charts_dir, "top_categories.png")
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            plt.close(fig)
            chart_paths["top_categories"] = p
    except Exception:
        pass

    # Stock histogram
    try:
        if stock_series is not None and not stock_series.dropna().empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(stock_series.dropna(), bins=40)
            ax.set_xlabel("Stock")
            ax.set_title("Stock Distribution (histogram)")
            p = os.path.join(charts_dir, "stock_hist.png")
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            plt.close(fig)
            chart_paths["stock_hist"] = p
    except Exception:
        pass

    # Top SKUs bar chart
    try:
        if not top_skus.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(top_skus[sku_col].astype(str)[::-1], top_skus["total_stock"][::-1])
            ax.set_xlabel("Total Stock")
            ax.set_title("Top SKUs by Total Stock")
            p = os.path.join(charts_dir, "top_skus.png")
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            plt.close(fig)
            chart_paths["top_skus"] = p
    except Exception:
        pass

    # Build summary text
    summary_lines = []
    summary_lines.append(f"The inventory dataset contains {num_rows:,} rows and {unique_skus:,} unique SKUs." if unique_skus is not None else f"The inventory dataset contains {num_rows:,} rows.")
    if stock_stats:
        summary_lines.append(f"Stock values range from {stock_stats.get('min')} to {stock_stats.get('max')}, median {stock_stats.get('median')}, mean {stock_stats.get('mean')}.")
        summary_lines.append(f"There are {stock_stats.get('zero_stock_count', 0)} SKUs with zero stock and {stock_stats.get('na_count', 0)} rows with missing stock.")
    if not top_categories.empty:
        top_cats = top_categories.head(5)[[category_col, "total_stock"]].to_dict(orient="records")
        summary_lines.append(f"Top categories by stock (top 5): {', '.join([str(r[category_col]) + ' (' + str(int(r['total_stock'])) + ')' for r in top_cats])}.")
    if not top_skus.empty:
        top_skus_list = top_skus.head(5).to_dict(orient="records")
        summary_lines.append(f"Top SKUs by stock (top 5): {', '.join([str(r[sku_col]) + ' (' + str(int(r['total_stock'])) + ')' for r in top_skus_list])}.")

    summary_text = " ".join(summary_lines)

    # Pack results
    result = {
        "summary_text": summary_text,
        "metrics": {
            "num_rows": num_rows,
            "unique_skus": int(unique_skus) if unique_skus is not None else None,
            "columns": columns
        },
        "missing_counts": missing_counts,
        "top_categories": top_categories,
        "top_skus_by_stock": top_skus,
        "stock_stats": stock_stats,
        "size_distribution": size_dist,
        "color_distribution": color_dist,
        "chart_paths": chart_paths,
        "raw_df_preview": df.head(50)
    }
    return result
