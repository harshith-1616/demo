import re
import string
import time

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------
# Preprocessing — MUST match the function used when the model was trained.
# ----------------------------------------------------------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1116; }
    .stApp { background: linear-gradient(180deg, #0f1116 0%, #14161d 100%); }

    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1c1f2b 0%, #23273a 100%);
        border: 1px solid #2c2f42;
        margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 2.2rem; margin-bottom: 0.2rem; }
    .hero p { color: #9aa0b4; font-size: 1.02rem; margin: 0; }

    .verdict-real {
        background: linear-gradient(135deg, #16362a 0%, #1c4a37 100%);
        border: 1px solid #2f8f63;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #3a1b1b 0%, #4a2323 100%);
        border: 1px solid #b33d3d;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .verdict-label { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .verdict-sub { color: #b8bccb; font-size: 0.95rem; margin-top: 0.3rem; }

    .metric-card {
        background: #1a1c26;
        border: 1px solid #2a2d3d;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }

    .example-card {
        background: #1a1c26;
        border: 1px solid #2a2d3d;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #c4c8d6;
        margin-bottom: 0.5rem;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
MODEL_PATH = "news.pkl"

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


def get_label_names(model, raw_label):
    """Map whatever label format the model uses (0/1, 'FAKE'/'REAL', etc.)
    to a clean human-readable string."""
    s = str(raw_label).strip().lower()
    if s in ("1", "real", "true", "reliable"):
        return "Real"
    if s in ("0", "fake", "false", "unreliable"):
        return "Fake"
    return str(raw_label).title()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ About")
    st.write(
        "This app uses a trained TF-IDF + classifier pipeline to estimate "
        "whether a news article's text is **real** or **fake**."
    )
    st.markdown("---")
    st.markdown("### 🧪 Model status")

    model = None
    model_error = None
    try:
        model = load_model(MODEL_PATH)
        st.success("Model loaded ✅")
        if hasattr(model, "named_steps"):
            st.caption("Pipeline steps: " + " → ".join(model.named_steps.keys()))
    except Exception as e:
        model_error = e
        st.error("Model failed to load ❌")
        st.caption(f"`{type(e).__name__}`: {e}")

    st.markdown("---")
    st.markdown("### 📝 Tips")
    st.caption(
        "Paste a full paragraph or article body for the most reliable result — "
        "very short snippets give the model less signal to work with."
    )

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📰 Fake News Detector</h1>
    <p>Paste an article or headline below and let the model estimate its credibility.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning(
        "The model couldn't be loaded, so predictions are disabled below. "
        "Re-export `news.pkl` (with `joblib.dump`) and re-upload it, then rerun the app."
    )

# ----------------------------------------------------------------------------
# Main interaction area
# ----------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🔍 Check an article", "📄 Batch check (CSV)"])

EXAMPLES = {
    "Sample — plausible headline": (
        "Local council approves new funding for public transportation "
        "improvements after months of community consultation."
    ),
    "Sample — sensational claim": (
        "SHOCKING: Scientists confirm the moon is actually hollow and "
        "governments have hidden this secret for decades, insider reveals."
    ),
}

with tab_single:
    col_input, col_examples = st.columns([2.3, 1])

    with col_examples:
        st.markdown("**Try an example**")
        chosen_example = None
        for label, text in EXAMPLES.items():
            if st.button(label, use_container_width=True):
                chosen_example = text

    with col_input:
        default_text = chosen_example if chosen_example else st.session_state.get("article_text", "")
        article_text = st.text_area(
            "Article text",
            value=default_text,
            height=220,
            placeholder="Paste the article text or headline here...",
            key="article_text",
        )

        analyze_clicked = st.button("Analyze", type="primary", disabled=(model is None))

    if analyze_clicked:
        if not article_text.strip():
            st.info("Enter some text above first.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.3)
                try:
                    pred = model.predict([article_text])[0]
                    label = get_label_names(model, pred)

                    proba = None
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba([article_text])[0]
                        classes = getattr(model, "classes_", None)
                        if classes is not None:
                            idx = list(classes).index(pred)
                            proba = probs[idx]

                    st.markdown("### Result")
                    verdict_class = "verdict-real" if label == "Real" else "verdict-fake"
                    icon = "✅" if label == "Real" else "🚩"
                    conf_text = f"{proba*100:.1f}% confidence" if proba is not None else "Confidence unavailable"

                    st.markdown(f"""
                    <div class="{verdict_class}">
                        <p class="verdict-label">{icon} Predicted: {label}</p>
                        <p class="verdict-sub">{conf_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if proba is not None:
                        st.progress(min(max(float(proba), 0.0), 1.0))

                    with st.expander("See cleaned text used by the model"):
                        st.code(wordopt(article_text) or "(empty after cleaning)")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

with tab_batch:
    st.write("Upload a CSV with a column containing article text to check many at once.")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head(), use_container_width=True)

            text_col = st.selectbox("Which column has the article text?", df.columns)

            if st.button("Run batch analysis", type="primary", disabled=(model is None)):
                with st.spinner(f"Analyzing {len(df)} rows..."):
                    preds = model.predict(df[text_col].astype(str).tolist())
                    labels = [get_label_names(model, p) for p in preds]
                    result_df = df.copy()
                    result_df["prediction"] = labels

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(df[text_col].astype(str).tolist())
                        classes = list(getattr(model, "classes_", []))
                        confidences = [
                            probs[i][classes.index(p)] if classes else np.nan
                            for i, p in enumerate(preds)
                        ]
                        result_df["confidence"] = [f"{c*100:.1f}%" for c in confidences]

                st.success("Done.")
                st.dataframe(result_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <b>Real predicted</b><br>
                        <span style="font-size:1.6rem;">{labels.count('Real')}</span>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <b>Fake predicted</b><br>
                        <span style="font-size:1.6rem;">{labels.count('Fake')}</span>
                    </div>""", unsafe_allow_html=True)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_out,
                    file_name="news_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Couldn't process the CSV: {e}")

st.markdown("---")
st.caption(
    "This tool provides an automated estimate only and is not a substitute for "
    "independent fact-checking."
)
