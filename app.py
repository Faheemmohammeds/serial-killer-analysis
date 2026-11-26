import os
import re

import numpy as np
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
CSV_NAME = "cleaned_serial_killers_dataset.csv"


def find_file(base_dir: str, filename: str):
    """Search the repo for a file, case-insensitively. Returns the path or None."""
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.lower() == filename.lower():
                return os.path.join(root, name)
    return None


CLEAN_CSV_PATH = find_file(BASE_DIR, CSV_NAME)


# ---------------------------------------------------------
# Data cleaning helpers
# ---------------------------------------------------------
def clean_gender(x) -> str:
    """Map free-text gender descriptions to Male / Female / Both / Unknown."""
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().lower()
    has_female = "female" in s or bool(re.search(r"\b(f|woman|women)\b", s))
    # remove 'female' first so 'male' is not matched inside it
    s_no_female = s.replace("female", "")
    has_male = "male" in s_no_female or bool(re.search(r"\b(m|man|men)\b", s_no_female))
    if has_female and has_male:
        return "Both"
    if has_female:
        return "Female"
    if has_male:
        return "Male"
    return "Unknown"


def to_year(x):
    """Turn 1985, 1985.0, '1985', 'c. 1985', '1985-1990' into an integer year (first one found)."""
    if pd.isna(x):
        return pd.NA
    m = re.search(r"(1[0-9]{3}|20[0-9]{2})", str(x))
    return int(m.group(1)) if m else pd.NA


def is_year_col(name: str) -> bool:
    return "year" in name.lower()


def tidy_numeric(s: pd.Series) -> pd.Series:
    """Convert to numbers; if every value is whole, store as integers (5.0 -> 5)."""
    s = pd.to_numeric(s, errors="coerce")
    non_null = s.dropna()
    if len(non_null) and (non_null % 1 == 0).all():
        return s.round().astype("Int64")
    return s


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 1) Clean text columns (trim spaces, blanks -> missing)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
            df[c] = df[c].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})

    # 2) Fix year columns -> integer years, no decimals
    for c in df.columns:
        if is_year_col(c):
            df[c] = pd.array([to_year(v) for v in df[c]], dtype="Int64")

    # 3) Fix other numeric columns (floats that are really whole numbers, numbers stored as text)
    for c in df.columns:
        if is_year_col(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = tidy_numeric(df[c])
        else:
            converted = pd.to_numeric(df[c], errors="coerce")
            non_null = df[c].notna().sum()
            if non_null and converted.notna().sum() / non_null >= 0.9:
                df[c] = tidy_numeric(converted)

    # 4) Rebuild a proper gender column from gender_profile (keeps all categories)
    profile_col = next((c for c in df.columns if c.lower() == "gender_profile"), None)
    if profile_col:
        df["gender_clean"] = df[profile_col].apply(clean_gender)

    return df


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return prepare_data(pd.read_csv(path))


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
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
        df = prepare_data(pd.read_csv(uploaded))

year_cols = [c for c in df.columns if is_year_col(c)]
numeric_cols = [
    c for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c]) and c not in year_cols
]
cat_cols = [
    c for c in df.columns
    if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 50
]

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

        if numeric_cols:
            st.markdown("#### Numeric summary")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

        if year_cols:
            st.markdown("#### Year columns (earliest, latest, median)")
            rows = []
            for c in year_cols:
                s = df[c].dropna().astype(int)
                if len(s):
                    rows.append(
                        {"column": c, "earliest": int(s.min()), "latest": int(s.max()),
                         "median": int(s.median()), "records with a year": len(s)}
                    )
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("column"), use_container_width=True)

        with st.expander("🛠 Data quality check"):
            st.write("Loaded from:", CLEAN_CSV_PATH or "uploaded file")
            st.markdown("**Column types**")
            st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
            for col in df.columns:
                if "gender" in col.lower():
                    st.markdown(f"**Value counts: {col}**")
                    st.dataframe(df[col].value_counts(dropna=False).rename("count"),
                                 use_container_width=True)

# ---------------------------------------------------------
# Explore
# ---------------------------------------------------------
with tab_explore:
    if df.empty:
        st.info("No data loaded.")
    else:
        e1, e2 = st.columns(2)

        with e1:
            st.markdown("#### Category counts")
            if cat_cols:
                default = cat_cols.index("gender_clean") if "gender_clean" in cat_cols else 0
                cat = st.selectbox("Choose a categorical column", cat_cols,
                                   index=default, key="cat_select")
                st.bar_chart(df[cat].value_counts(dropna=False).head(20))
            else:
                st.info("No suitable categorical columns found.")

        with e2:
            st.markdown("#### Numeric distribution")
            if numeric_cols:
                num = st.selectbox("Choose a numeric column", numeric_cols, key="num_select")
                series = df[num].dropna().astype(float)
                if series.nunique() > 1:
                    hist = pd.cut(series, bins=15).value_counts().sort_index()
                    hist.index = hist.index.astype(str)
                    st.bar_chart(hist)
                else:
                    st.info("Not enough variation to plot.")
            else:
                st.info("No numeric columns found.")

        if year_cols:
            st.markdown("#### Cases by decade")
            ycol = st.selectbox("Choose a year column", year_cols, key="year_select")
            years = df[ycol].dropna().astype(int)
            if len(years):
                by_decade = (years // 10 * 10).value_counts().sort_index()
                by_decade.index = [f"{d}s" for d in by_decade.index]
                st.bar_chart(by_decade)
            else:
                st.info("No valid years in this column.")

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
