from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR.parent / "models" / "sentiment_model.joblib"


st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str):
    model = load_model()
    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0]
    confidence = float(max(probability) * 100)
    label = "Positive" if int(prediction) == 1 else "Negative"
    return label, confidence


def sentiment_style(label: str) -> str:
    color = "#1f8f4c" if label == "Positive" else "#c92a2a"
    return (
        f"<div style='padding: 1rem; border-radius: 14px; background: {color}15; "
        f"border: 1px solid {color}40; color: {color}; text-align: center;'>"
        f"<h2 style='margin: 0;'>{label}</h2></div>"
    )


st.title("Sentiment Analyzer")
st.write("Enter a movie review or any text and get a simple positive/negative prediction.")

text = st.text_area("Your text", height=180, placeholder="Type a review here...")

if st.button("Analyze sentiment", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        if len(text.split()) < 10:
            st.warning("⚠️ For best results, enter at least a few sentences. Short inputs may be unreliable.")
        
        try:
            label, confidence = predict_sentiment(text)
            st.markdown(sentiment_style(label), unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.1f}%")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to analyze text: {exc}")
