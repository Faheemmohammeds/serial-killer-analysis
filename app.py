"""
app.py
------
Streamlit Web Application for Global Serial Killer Profiles Analysis.
Interactive dashboard for exploration, filtering, profiling, and report downloads.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global Serial Killer Profiles | Criminology Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    cleaned_path = 'data/cleaned_serial_killers_dataset.csv'
    raw_path = 'data/00_serial_killers_dataset.csv'
    
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        st.error("Dataset not found. Please ensure data files exist in the 'data/' directory.")
        return pd.DataFrame(), pd.DataFrame()
        
    df_raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
    return df, df_raw

df, df_raw = load_dataset()

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/law.png", width=64)
    st.title("Filter Dataset")
    st.markdown("Customize cohorts and geographical boundaries.")

    # Search Name or Alias
    search_query = st.text_input("🔍 Search Killer Name or Alias", "").strip()

    # Country filter
    all_countries = sorted(df['country_primary'].dropna().unique().tolist()) if not df.empty else []
    selected_countries = st.multiselect("🌍 Primary Country", all_countries, default=[])

    # Decade filter
    if not df.empty and 'active_decade' in df.columns:
        decades_list = [d for d in df['active_decade'].dropna().unique() if d != 'Unknown']
        decades_list = sorted(decades_list, key=lambda x: int(x.replace('s', '')))
        if 'Unknown' in df['active_decade'].values:
            decades_list.append('Unknown')
    else:
        decades_list = []
    selected_decades = st.multiselect("📅 Active Decade", decades_list, default=[])

    # Victim Category filter
    all_categories = ['Low (5-14)', 'High (30+)', 'Medium (15-29)', 'Unrecorded / Disputed', 'Very Low (<5)']
    selected_categories = st.multiselect("🎯 Victim Count Category", all_categories, default=[])

    # Special Demographic & Occupational Cohorts
    st.markdown("---")
    st.markdown("### Specialized Cohorts")
    
    gender_options = ['All', 'Male', 'Female', 'Group / Pair', 'Unidentified / Unknown']
    selected_gender = st.selectbox("Demographic / Operating Mode", gender_options, index=0)
    
    filter_medical = st.checkbox("🩺 Medical Professional Only", value=False)
    filter_le = st.checkbox("🛡️ Law Enforcement Only", value=False)
    filter_solo = st.checkbox("👤 Solo Operatives Only", value=False)

    st.markdown("---")
    st.caption("Data Source: Wikipedia (CC BY-SA 4.0) via Kaggle")

# ---------------------------------------------------------
# Apply Filtering Logic
# ---------------------------------------------------------
filtered_df = df.copy() if not df.empty else pd.DataFrame()

if not filtered_df.empty:
    if search_query:
        mask = (
            filtered_df['name'].astype(str).str.contains(search_query, case=False, na=False) |
            filtered_df['aliases'].astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    if selected_countries:
        filtered_df = filtered_df[filtered_df['country_primary'].isin(selected_countries)]

    if selected_decades:
        filtered_df = filtered_df[filtered_df['active_decade'].isin(selected_decades)]

    if selected_categories:
        filtered_df = filtered_df[filtered_df['victim_count_category'].isin(selected_categories)]

    if selected_gender != 'All':
        if selected_gender == 'Male':
            filtered_df = filtered_df[filtered_df['gender_profile'].str.contains('Male', na=False)]
        else:
            filtered_df = filtered_df[filtered_df['gender_profile'] == selected_gender]

    if filter_medical:
        filtered_df = filtered_df[filtered_df['medical_professional'] == True]

    if filter_le:
        filtered_df = filtered_df[filtered_df['law_enforcement'] == True]

    if filter_solo:
        filtered_df = filtered_df[filtered_df['operated_alone'] == True]

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown('<div class="main-title">Global Serial Killer Profiles Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Empirical Criminological Data Analysis, Systematic Cleaning & Profile Investigation (1900–2023)</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Top High-Level Metrics
# ---------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

total_cases = len(filtered_df)
proven_series = filtered_df['victims_proven'].dropna()
median_victims = f"{proven_series.median():.0f}" if not proven_series.empty else "N/A"
mean_victims = f"{proven_series.mean():.1f}" if not proven_series.empty else "N/A"
top_country = filtered_df['country_primary'].value_counts().index[0] if not filtered_df.empty else "N/A"
top_country_pct = f"({(filtered_df['country_primary'].value_counts().iloc[0] / total_cases * 100):.1f}%)" if total_cases > 0 else ""
med_cases = int(filtered_df['medical_professional'].sum()) if 'medical_professional' in filtered_df.columns else 0
le_cases = int(filtered_df['law_enforcement'].sum()) if 'law_enforcement' in filtered_df.columns else 0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_cases}</div>
        <div class="metric-label">Filtered Profiles</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{median_victims} <span style="font-size:1rem;color:#64748B;">(avg: {mean_victims})</span></div>
        <div class="metric-label">Proven Victims (Median)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.4rem;">{top_country} <span style="font-size:0.9rem;color:#64748B;">{top_country_pct}</span></div>
        <div class="metric-label">Top Jurisdiction</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{med_cases} <span style="font-size:1rem;color:#64748B;">({med_cases/total_cases*100:.1f}%)</span></div>
        <div class="metric-label">Medical Workers</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{le_cases} <span style="font-size:1rem;color:#64748B;">({le_cases/total_cases*100:.1f}%)</span></div>
        <div class="metric-label">Law Enforcement</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Dashboard",
    "📈 Interactive Analytics",
    "🔍 Profile Investigator",
    "🛠️ Data Cleaning Audit",
    "📑 Downloads & Reports"
])

# =========================================================
# TAB 1: EXECUTIVE DASHBOARD
# =========================================================
with tab1:
    st.markdown("### Dataset Overview & Empirical Findings")
    st.info(
        "💡 **Key Criminological Reality**: Despite pop-culture perceptions of prolific triple-digit killers, "
        "**50% of serial killers in this global dataset have 8 or fewer proven victims**. "
        "Cases dramatically surged between 1970 and 1999 (52.6% of all profiles) before declining in the 21st century "
        "due to widespread DNA databases, automated fingerprinting (AFIS), and digital surveillance."
    )
    
    col_l, col_r = st.columns([1.5, 1])
    
    with col_l:
        st.markdown("#### Primary Data Table")
        display_cols = [
            'name', 'country_primary', 'active_decade', 'victims_proven',
            'victim_count_max', 'victim_count_category', 'gender_profile', 'operated_alone'
        ]
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(
            filtered_df[available_cols],
            use_container_width=True,
            height=400
        )
        
        # CSV Download Button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_serial_killers_dataset.csv",
            mime="text/csv"
        )
        
    with col_r:
        st.markdown("#### Summary Composition")
        if not filtered_df.empty and 'victim_count_category' in filtered_df.columns:
            cat_counts = filtered_df['victim_count_category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            fig_pie = px.pie(
                cat_counts,
                names='Category',
                values='Count',
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Blues_r,
                title="Victim Count Tier Share"
            )
            fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

# =========================================================
# TAB 2: INTERACTIVE ANALYTICS
# =========================================================
with tab2:
    st.markdown("### Interactive Statistical Visualizations")
    
    analysis_view = st.radio(
        "Select Analytical Angle:",
        ["Decade Surge (Temporal)", "Geographic Concentration", "Victimology & Outliers", "Offender Cohorts & Backgrounds", "Spree Longevity"],
        horizontal=True
    )
    
    if analysis_view == "Decade Surge (Temporal)":
        st.markdown("#### Documented Serial Killer Cases by Active Decade (1900s–2020s)")
        if 'active_decade' in filtered_df.columns:
            dec_counts = filtered_df['active_decade'].value_counts()
            decs = [d for d in dec_counts.index if d != 'Unknown']
            decs_sorted = sorted(decs, key=lambda x: int(x.replace('s', '')))
            counts = [dec_counts[d] for d in decs_sorted]
            
            fig_dec = go.Figure(data=[
                go.Bar(
                    x=decs_sorted,
                    y=counts,
                    marker=dict(color=counts, colorscale='Blues', showscale=True),
                    text=counts,
                    textposition='outside'
                )
            ])
            fig_dec.update_layout(
                xaxis_title="Active Decade (Earliest Documented Murders)",
                yaxis_title="Count of Killer Profiles",
                height=480,
                margin=dict(t=30, b=30, l=30, r=30)
            )
            st.plotly_chart(fig_dec, use_container_width=True)
            st.caption("Criminological Driver: Mid-century highway expansion, hitchhiking, and decentralized policing created an optimal predator environment. Post-2000 forensic DNA profiling closed this window.")
            
    elif analysis_view == "Geographic Concentration":
        st.markdown("#### Top 15 Countries by Recorded Offender Profiles")
        if 'country_primary' in filtered_df.columns:
            top_c = filtered_df['country_primary'].value_counts().head(15).reset_index()
            top_c.columns = ['Country', 'Profiles']
            top_c = top_c.sort_values(by='Profiles', ascending=True)
            
            fig_geo = px.bar(
                top_c,
                x='Profiles',
                y='Country',
                orientation='h',
                color='Profiles',
                color_continuous_scale='Crest_r',
                text='Profiles',
                title="Recorded Offender Volume by Sovereign State"
            )
            fig_geo.update_layout(height=480, margin=dict(t=40, b=30, l=30, r=30))
            st.plotly_chart(fig_geo, use_container_width=True)
            st.warning("⚠️ **Documentation Bias**: The US and UK account for over 45% of entries due to open records (FOIA) and English Wikipedia focus, not an inherent eight-fold violent crime rate.")
            
    elif analysis_view == "Victimology & Outliers":
        st.markdown("#### Distribution of Proven Victim Counts & High-End Outliers")
        proven = filtered_df['victims_proven'].dropna()
        if not proven.empty:
            col_v1, col_v2 = st.columns([1.5, 1])
            with col_v1:
                fig_hist = px.histogram(
                    filtered_df,
                    x='victims_proven',
                    nbins=40,
                    color_discrete_sequence=['#2B5C8F'],
                    title="Proven Victims Distribution (Right Skewed)"
                )
                fig_hist.add_vline(x=proven.median(), line_dash="dash", line_color="#D95F02", annotation_text=f"Median: {proven.median():.0f}")
                fig_hist.add_vline(x=proven.mean(), line_dash="dot", line_color="#7570B3", annotation_text=f"Mean: {proven.mean():.1f}")
                fig_hist.update_layout(height=420)
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col_v2:
                st.markdown("##### Extreme Outliers (Distorting the Mean)")
                outliers = filtered_df.nlargest(5, 'victims_proven')[['name', 'victims_proven', 'country_primary']]
                for idx, r in outliers.iterrows():
                    st.markdown(f"• **{r['name']}** ({r['country_primary']}): **{int(r['victims_proven'])}** proven victims")
                st.info("The vast majority (56.3%) of killers fall in the 5–14 victim tier. Rare triple-digit outliers skew the arithmetic mean.")

    elif analysis_view == "Offender Cohorts & Backgrounds":
        st.markdown("#### Special Offender Demographics & Institutional Infiltration")
        c1, c2 = st.columns(2)
        with c1:
            if 'gender_profile' in filtered_df.columns:
                g_df = filtered_df['gender_profile'].value_counts().reset_index()
                g_df.columns = ['Demographic Group', 'Count']
                fig_g = px.bar(g_df, x='Count', y='Demographic Group', orientation='h', color='Count', color_continuous_scale='Blues_r', title="Offender Classification")
                st.plotly_chart(fig_g, use_container_width=True)
        with c2:
            cohort_data = pd.DataFrame({
                'Cohort': ['Non-Institutional', 'Medical Professional ("Angels of Death")', 'Law Enforcement Background'],
                'Count': [
                    len(filtered_df) - med_cases - le_cases,
                    med_cases,
                    le_cases
                ]
            })
            fig_inst = px.pie(cohort_data, names='Cohort', values='Count', color_discrete_sequence=['#94A3B8', '#DC2626', '#EAB308'], hole=0.4, title="Institutional Infiltration Breakdown")
            st.plotly_chart(fig_inst, use_container_width=True)
            
    elif analysis_view == "Spree Longevity":
        st.markdown("#### Criminal Spree Longevity (Duration in Years Active)")
        if 'years_active' in filtered_df.columns:
            dur = filtered_df['years_active'].dropna()
            fig_dur = px.histogram(filtered_df, x='years_active', nbins=30, color_discrete_sequence=['#386CB0'], title="Years from First Known Crime to Cessation/Arrest")
            fig_dur.add_vline(x=dur.median(), line_dash="dash", line_color="#D95F02", annotation_text=f"Median: {dur.median():.0f} yrs")
            fig_dur.update_layout(height=420)
            st.plotly_chart(fig_dur, use_container_width=True)
            st.caption("50% of killers are stopped within 4 years; a long-tail 25% evades justice for over 12 years (up to 30–50+ years, e.g., Samuel Little).")

# =========================================================
# TAB 3: PROFILE INVESTIGATOR
# =========================================================
with tab3:
    st.markdown("### Serial Killer Case Profile Investigator")
    if not filtered_df.empty:
        killer_names = filtered_df['name'].dropna().tolist()
        selected_killer_name = st.selectbox("Select a Profile to Inspect:", killer_names)
        
        killer_row = filtered_df[filtered_df['name'] == selected_killer_name].iloc[0]
        
        col_k1, col_k2 = st.columns([1, 2])
        
        with col_k1:
            st.markdown(f"### {killer_row['name']}")
            if pd.notna(killer_row.get('aliases')) and str(killer_row['aliases']).strip():
                st.markdown(f"**Aliases**: *{killer_row['aliases']}*")
                
            st.markdown(f"**Primary Country**: {killer_row.get('country_primary', 'Unknown')}")
            st.markdown(f"**Active Span**: {killer_row.get('active_span', 'Unknown')} ({killer_row.get('active_decade', 'N/A')})")
            st.markdown(f"**Proven Victims**: `{killer_row.get('victims_proven', 'Unrecorded')}`")
            st.markdown(f"**Possible Victims**: `{killer_row.get('victims_possible', 'Unrecorded')}`")
            st.markdown(f"**Confessed Victims**: `{killer_row.get('victims_confessed', 'Unrecorded')}`")
            st.markdown(f"**Severity Tier**: `{killer_row.get('victim_count_category', 'N/A')}`")
            st.markdown(f"**Medical Background**: {'Yes (Healthcare Worker)' if killer_row.get('medical_professional') else 'No'}")
            st.markdown(f"**Law Enforcement**: {'Yes (Police/Security)' if killer_row.get('law_enforcement') else 'No'}")
            
            if pd.notna(killer_row.get('wikipedia_url')):
                st.link_button("🌐 Open Wikipedia Article", killer_row['wikipedia_url'])
                
        with col_k2:
            st.markdown("#### Case Summary")
            summary_text = killer_row.get('summary', 'No summary available.')
            st.write(summary_text)
            
            if pd.notna(killer_row.get('victim_details')) and str(killer_row['victim_details']).strip():
                st.markdown("#### Victimology Details")
                st.write(str(killer_row['victim_details'])[:600] + "...")
                
            if pd.notna(killer_row.get('modus_operandi')) and str(killer_row['modus_operandi']).strip():
                st.markdown("#### Modus Operandi")
                st.write(str(killer_row['modus_operandi']))
    else:
        st.warning("No profiles match the active filter criteria.")

# =========================================================
# TAB 4: DATA CLEANING AUDIT
# =========================================================
with tab4:
    st.markdown("### Data Quality Diagnosis & Cleaning Audit")
    st.markdown(
        "A rigorous inspection of the raw Kaggle dataset (`00_serial_killers_dataset.csv`, 494 rows, 33 columns) "
        "identified four major structural anomalies that required algorithmic remediation."
    )
    
    st.markdown("#### 1. The Victim Count Zeroing Bug")
    st.error(
        "**Defect**: The raw scraper only extracted counts from infobox `Victims` tags. "
        "When Wikipedia used `Convictions` or `Killed`, `victims_proven` was left as `NaN`, defaulting `victim_count_max = 0`. "
        "Consequently, prolific killers like **Gary Ridgway** (49 proven murders), **William Unek** (58), and **Pedro Rodrigues Filho** (71) "
        "were erroneously placed into the `'Very Low (<5)'` category. "
        "**Remediation**: Cross-referenced infobox tags and text summaries, restoring verified counts for 27 key offenders and establishing an `'Unrecorded / Disputed'` category."
    )
    
    st.markdown("#### 2. Before vs. After Remediation Matrix")
    audit_df = pd.DataFrame({
        "Metric / Parameter": [
            "Total Master Rows",
            "Missing Primary Country",
            "Harmonized Sovereign Nations",
            "Proven Victims Captured",
            "Zero-Count Misclassifications",
            "100% Null Columns ('cities')"
        ],
        "Raw Scraped Dataset": [
            "494 Profiles (2 duplicates)",
            "71 (14.6% missing)",
            "83 Fragmented Labels",
            "414 Profiles",
            "80 Prolific Killers Mislabeled <5",
            "494 / 494 Missing (100%)"
        ],
        "Cleaned Dataset": [
            "492 Unique Individuals",
            "0 (100% recovered via categories)",
            "67 Standardized Sovereign States",
            "439 Profiles (+25 restored)",
            "0 Profiles (Re-indexed accurately)",
            "Dropped from Clean Dataset"
        ],
        "Pipeline Action": [
            "Deduplicated by [name, url]",
            "Regex demonym classification",
            "Consolidated US, UK, USSR, Germany",
            "Parsed 'Convictions' & 'Killed'",
            "Recomputed max and 5-tier binning",
            "Pruned uninformative feature"
        ]
    })
    st.table(audit_df)

# =========================================================
# TAB 5: DOWNLOADS & REPORTS
# =========================================================
with tab5:
    st.markdown("### Project Deliverables & Research Artifacts")
    st.markdown("Download all verified code sets, clean data files, research reports, and executive presentations.")
    
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        st.markdown("#### 📦 Research Reports & Presentations")
        
        # PDF Report
        pdf_path = "serial_killers_analysis_report.pdf" if os.path.exists("serial_killers_analysis_report.pdf") else "REPORT.pdf"
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Download 5-Page PDF Research Report",
                    data=f.read(),
                    file_name="serial_killers_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # PPTX Presentation
        pptx_path = "serial_killers_analysis.pptx"
        if os.path.exists(pptx_path):
            with open(pptx_path, "rb") as f:
                st.download_button(
                    "📊 Download 10-Slide PowerPoint Presentation (.pptx)",
                    data=f.read(),
                    file_name="serial_killers_analysis.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True
                )
                
        # Markdown Report
        report_md_path = "REPORT.md"
        if os.path.exists(report_md_path):
            with open(report_md_path, "rb") as f:
                st.download_button(
                    "📝 Download Markdown Report (REPORT.md)",
                    data=f.read(),
                    file_name="REPORT.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    with d_col2:
        st.markdown("#### 💻 Code & Clean Datasets")
        
        # Cleaned CSV
        clean_csv_path = "data/cleaned_serial_killers_dataset.csv"
        if os.path.exists(clean_csv_path):
            with open(clean_csv_path, "rb") as f:
                st.download_button(
                    "💾 Download Cleaned Dataset (.csv)",
                    data=f.read(),
                    file_name="cleaned_serial_killers_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        # Jupyter Notebook
        nb_path = "serial_killers_analysis_notebook.ipynb"
        if os.path.exists(nb_path):
            with open(nb_path, "rb") as f:
                st.download_button(
                    "📓 Download Annotated Jupyter Notebook (.ipynb)",
                    data=f.read(),
                    file_name="serial_killers_analysis_notebook.ipynb",
                    mime="application/x-ipynb+json",
                    use_container_width=True
                )
                
        # Project README
        readme_path = "README.md"
        if os.path.exists(readme_path):
            with open(readme_path, "rb") as f:
                st.download_button(
                    "📖 Download Project Documentation (README.md)",
                    data=f.read(),
                    file_name="README.md",
                    mime="text/markdown",
                    use_container_width=True
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
    unsafe_allow_html=True
)
