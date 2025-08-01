import streamlit as st
from transformers import pipeline
import re

# App Title & Config
st.set_page_config(page_title="🧠 Smart AI Text Summarizer", layout="centered")
st.title("📚 Smart AI Text Summarizer")
st.write("Paste any long text and get a meaningful, focused summary on key themes like importance, purpose, and mindfulness 🔍")

# User input
text = st.text_area("📝 Paste your long text below:", height=300)

# Word limit slider
word_limit = st.slider("✂ Desired summary length (approx. word count):", 30, 200, 80)

# Load summarizer (cached)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def clean_text(text):
    """Remove extra whitespaces and tidy up text."""
    return re.sub(r'\s+', ' ', text.strip())

def extract_key_points(summary_text):
    """Split summary into bullet points for clarity."""
    sentences = re.split(r'(?<=[.!?]) +', summary_text)
    # Filter out too-short sentences
    points = [f"• {s.strip()}" for s in sentences if len(s.strip()) > 30]
    return points

# Generate summary button
if st.button("🚀 Generate Focused Summary"):
    if not text.strip():
        st.warning("⚠️ Please paste some content to summarize.")
    else:
        with st.spinner("✨ Thinking deeply and summarizing..."):
            cleaned = clean_text(text)
            # Dynamic length calculation
            max_len = int(word_limit * 1.3)
            min_len = int(word_limit * 0.6)

            # Generate summary
            result = summarizer(
                cleaned,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            summary_text = result[0]['summary_text']

            # Display summary
            st.subheader("🧠 Summary Output:")
            st.success(summary_text)

            # Display bullet points
            key_points = extract_key_points(summary_text)
            if key_points:
                st.subheader("📌 Key Points:")
                for point in key_points:
                    st.markdown(point)
            else:
                st.info("Couldn’t extract key points (summary might be too short).")

st.markdown("---")
st.markdown("✅ Built with [Streamlit](https://streamlit.io) & 🤗 HuggingFace Transformers")
