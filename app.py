"📝 Download Markdown Report (REPORT.md)",
data=f.read(),
file_name="REPORT.md",
mime="text/markdown",
use_container_width=True
                
    with d_col2:
        st.markdown("#### 💻 Code & Clean Datasets")
        
        # Cleaned CSV
        clean_csv_path = os.path.join(BASE_DIR, "data", "cleaned_serial_killers_dataset.csv")
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
        nb_path = os.path.join(BASE_DIR, "serial_killers_analysis_notebook.ipynb")
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
        readme_path = os.path.join(BASE_DIR, "README.md")
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
