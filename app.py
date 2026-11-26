import os

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page config & paths
# ---------------------------------------------------------
st.set_page_config(
    page_title="Criminological Research Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV_PATH = os.path.join(BASE_DIR, "data", "cleaned_serial_killers_dataset.csv")


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("📊 Criminological Research Dashboard")
st.caption("Wikipedia & Kaggle research dataset, for academic study and objective data analysis.")

if not os.path.exists(CLEAN_CSV_PATH):
    st.error(
        f"Dataset not found at: {CLEAN_CSV_PATH}\n\n"
        "Make sure `data/cleaned_serial_killers_dataset.csv` exists in your repository."
    )
    df = pd.DataFrame()
else:
    df = load_data(CLEAN_CSV_PATH)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_overview, tab_explore, tab_downloads = st.tabs(
    ["📋 Overview", "🔍 Explore", "📥 Downloads"]
)

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------
with tab_overview:
    if df.empty:
        st.info("No data loaded.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

        st.markdown("#### Data preview")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("#### Summary statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

# ---------------------------------------------------------
# Explore
# ---------------------------------------------------------
with tab_explore:
    if df.empty:
        st.info("No data loaded.")
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [
            c for c in df.select_dtypes(exclude="number").columns
            if df[c].nunique() <= 50
        ]

        e1, e2 = st.columns(2)

        with e1:
            st.markdown("#### Category counts")
            if cat_cols:
                cat = st.selectbox("Choose a categorical column", cat_cols, key="cat_select")
                counts = df[cat].value_counts().head(20)
                st.bar_chart(counts)
            else:
                st.info("No suitable categorical columns found.")

        with e2:
            st.markdown("#### Numeric distribution")
            if numeric_cols:
                num = st.selectbox("Choose a numeric column", numeric_cols, key="num_select")
                series = df[num].dropna()
                if series.nunique() > 1:
                    hist = pd.cut(series, bins=15).value_counts().sort_index()
                    hist.index = hist.index.astype(str)
                    st.bar_chart(hist)
                else:
                    st.info("Not enough variation to plot.")
            else:
                st.info("No numeric columns found.")

# ---------------------------------------------------------
# Downloads
# ---------------------------------------------------------
with tab_downloads:
    st.markdown("### 📥 Downloads")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("#### 📄 Reports & Documentation")

        # Markdown report
        report_path = os.path.join(BASE_DIR, "REPORT.md")
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button(
                    "📝 Download Markdown Report (REPORT.md)",
                    data=f.read(),
                    file_name="REPORT.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_report",
                )
        else:
            st.caption("REPORT.md not found.")

    with d_col2:
        st.markdown("#### 💻 Code & Clean Datasets")

        # Cleaned CSV
        if os.path.exists(CLEAN_CSV_PATH):
            with open(CLEAN_CSV_PATH, "rb") as f:
                st.download_button(
                    "💾 Download Cleaned Dataset (.csv)",
                    data=f.read(),
                    file_name="cleaned_serial_killers_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_csv",
                )

        # Jupyter Notebook
        nb_path = os.path.join(BASE_DIR, "serial_killers_analysis_notebook.ipynb")
        if os.path.exists(nb_path):
            with open(nb_path, "rb") as f:
                st.download_button(
                    "📓 Download Annotated Jupyter Notebook (.ipynb)",
                    data=f.read(),
                    file_name="serial_killers_analysis_notebook.ipynb",
                    mime="application/x-ipynb+json",
                    use_container_width=True,
                    key="dl_notebook",
                )

        # Project README
        readme_path = os.path.join(BASE_DIR, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "rb") as f:
                st.download_button(
                    "📖 Download Project Documentation (README.md)",
                    data=f.read(),
                    file_name="README.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_readme",
                )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><small style='color:#64748B;'>"
    "Criminological Research Dashboard | Wikipedia & Kaggle Research Dataset | "
    "Designed for academic study and objective data science demonstration."
    "</small></center>",
    unsafe_allow_html=True,
)
