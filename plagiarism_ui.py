import streamlit as st
from plagiarism_detector import PlagiarismDetector, Config
import pandas as pd
import numpy as np

# Set page layout
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🧩",
    layout="wide",
)

# Title and description
st.title("🧠 Advanced Plagiarism Detector")
st.markdown("""
This tool analyzes your text for potential plagiarism using **TF-IDF**, **SBERT**, **N-gram**, and **LCS similarity metrics**.  
Optionally, it can search **ArXiv**, **Google Scholar**, and **the Web** for similar content (if API key is configured).
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")

use_web_search = st.sidebar.checkbox("Use Web Search (requires API key)", value=False)
show_details = st.sidebar.checkbox("Show detailed similarity breakdown", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Aviral | Powered by Sentence-BERT + TF-IDF")

# Text input
st.subheader("✍️ Enter your text below:")
user_text = st.text_area("Paste or type your text here:", height=250, placeholder="Enter at least 50 characters...")

# Detect button
if st.button("🚀 Run Plagiarism Check"):
    if not user_text or len(user_text.strip()) < 50:
        st.warning("⚠️ Please enter at least 50 characters.")
    else:
        with st.spinner("Analyzing text for plagiarism... Please wait ⏳"):
            config = Config()
            detector = PlagiarismDetector(config)
            results = detector.detect_plagiarism(user_text, use_web_search=use_web_search)

        st.success("✅ Analysis Complete!")

        # Display overall score and verdict
        score = results['overall_score'] * 100
        verdict = results['verdict']

        st.markdown("## 📊 Overall Results")
        st.metric(label="Plagiarism Score", value=f"{score:.1f}%")
        st.markdown(f"**Verdict:** {verdict}")

        # Progress bar visualization
        st.progress(min(score / 100, 1.0))

        # Display color-coded risk
        if score >= 60:
            st.error("🔴 Critical - Likely Plagiarized")
        elif score >= 40:
            st.warning("🟠 High Risk - Review Required")
        elif score >= 20:
            st.info("🟡 Moderate Risk - Some Similarities")
        else:
            st.success("🟢 Low Risk - Appears Original")

        st.markdown("---")

        # Display match results
        if results['matches']:
            st.subheader("🔍 Top Matches Found")

            match_data = []
            for match in results['matches']:
                match_data.append({
                    "Title": match["title"],
                    "Source": match["source"],
                    "Similarity (%)": round(match["similarity"] * 100, 2),
                    "URL": match["url"],
                })

            df = pd.DataFrame(match_data)
            st.dataframe(df, use_container_width=True)

            # Expandable detailed breakdowns
            if show_details:
                for i, match in enumerate(results['matches'], start=1):
                    with st.expander(f"📘 [{i}] {match['title']} ({match['similarity'] * 100:.1f}%)"):
                        st.markdown(f"**Source:** {match['source']}")
                        if 'authors' in match:
                            st.markdown(f"**Authors:** {match['authors']} ({match.get('year', 'N/A')})")
                        st.markdown(f"**URL:** [{match['url']}]({match['url']})")
                        st.markdown(f"**Snippet:** {match['snippet'][:400]}...")

                        details = match['details']
                        metrics_df = pd.DataFrame({
                            "Metric": ["TF-IDF", "N-gram", "LCS"],
                            "Score (%)": [
                                details["tfidf"] * 100,
                                details["ngram"] * 100,
                                details["lcs"] * 100,
                            ]
                        })

                        st.bar_chart(metrics_df.set_index("Metric"))

        else:
            st.info("✅ No significant matches found. Text appears original!")

# Footer
st.markdown("---")
st.caption("⚙️ This plagiarism detector uses NLP similarity algorithms (TF-IDF, LCS, N-gram) \
and optional online sources (ArXiv, Google Scholar, Web Search).")
