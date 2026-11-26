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


def find_file(base_dir: str, filename: str):
    """Search the repo for a file, case-insensitively. Returns the path or None."""
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.lower() == filename.lower():
                return os.path.join(root, name)
    return None


CSV_NAME = "cleaned_serial_killers_dataset.csv"
CLEAN_CSV_PATH = find_file(BASE_DIR, CSV_NAME)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("📊 Criminological Research Dashboard")
st.caption("Wikipedia & Kaggle research dataset, for academic study and objective data analysis.")

df = pd.DataFrame()

if CLEAN_CSV_PATH:
    df = load_data(CLEAN_CSV_PATH)
else:
    st.warning(
        f"`{CSV_NAME}` was not found in the repository. "
        "Commit it to your GitHub repo (e.g. in a `data/` folder), or upload it below."
    )
    uploaded = st.file_uploader("Upload the cleaned CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)

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
                st.bar_chart(df[cat].value_counts().head(20))
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

        report_path = find_file(BASE_DIR, "REPORT.md")
        if report_path:
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
            st.caption("REPORT.md not found in the repository.")

        readme_path = find_file(BASE_DIR, "README.md")
        if readme_path:
            with open(readme_path, "rb") as f:
                st.download_button(
                    "📖 Download Project Documentation (README.md)",
                    data=f.read(),
                    file_name="README.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_readme",
                )

    with d_col2:
        st.markdown("#### 💻 Code & Clean Datasets")

        if not df.empty:
            st.download_button(
                "💾 Download Cleaned Dataset (.csv)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=CSV_NAME,
                mime="text/csv",
                use_container_width=True,
                key="dl_csv",
            )

        nb_path = find_file(BASE_DIR, "serial_killers_analysis_notebook.ipynb")
        if nb_path:
            with open(nb_path, "rb") as f:
                st.download_button(
                    "📓 Download Annotated Jupyter Notebook (.ipynb)",
                    data=f.read(),
                    file_name="serial_killers_analysis_notebook.ipynb",
                    mime="application/x-ipynb+json",
                    use_container_width=True,
                    key="dl_notebook",
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
