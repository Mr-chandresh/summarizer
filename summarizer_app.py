import streamlit as st
from transformers import pipeline
import re

# App title & config
st.set_page_config(page_title="🧠 Smart AI Text Summarizer", layout="centered")
st.title("📚 Smart AI Text Summarizer")
st.write("Paste any long text and get a focused summary on key themes like importance, mindfulness, purpose 🔍")

# User input
text = st.text_area("📝 Paste your long text below:", height=300)

# Word limit slider
word_limit = st.slider("✂ Desired summary length (approx. word count):", 30, 200, 80)

# Load summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def extract_key_points(summary_text):
    sentences = re.split(r'(?<=[.!?]) +', summary_text)
    points = [f"• {s.strip()}" for s in sentences if len(s.strip()) > 30]
    return points

def trim_summary_to_word_limit(summary_text, word_limit):
    words = summary_text.split()
    if len(words) <= word_limit:
        return summary_text
    else:
        return ' '.join(words[:word_limit]) + "..."

if st.button("🚀 Generate Focused Summary"):
    if not text.strip():
        st.warning("⚠️ Please paste some content to summarize.")
    else:
        with st.spinner("✨ Thinking deeply and summarizing..."):
            cleaned = clean_text(text)

            # Add prompt for better focus
            prompt_text = (
                "Summarize this text focusing on: importance of time, student life, mindfulness and purpose.\n\n" + cleaned
            )

            # Since max_length/min_length are tokens, approximate: 1 word ≈ 1.3 tokens
            max_tokens = int(word_limit * 1.3)
            min_tokens = int(word_limit * 0.8)

            result = summarizer(
                prompt_text,
                max_length=max_tokens,
                min_length=min_tokens,
                do_sample=False
            )
            raw_summary = result[0]['summary_text']

            # Trim to exact word count if longer
            summary_text = trim_summary_to_word_limit(raw_summary, word_limit)

            st.subheader("🧠 Summary Output:")
            st.success(summary_text)

            key_points = extract_key_points(summary_text)
            if key_points:
                st.subheader("📌 Key Points:")
                for point in key_points:
                    st.markdown(point)
            else:
                st.info("Couldn’t extract key points (summary might be too short).")

st.markdown("---")
st.markdown("✅ Built with [Streamlit](https://streamlit.io) & 🤗 HuggingFace Transformers")
