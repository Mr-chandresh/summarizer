import streamlit as st
from transformers import pipeline
import re

# App title & config
st.set_page_config(page_title="🧠 Smart AI Text Summarizer", layout="centered")
st.title("📚 Smart AI Text Summarizer")
st.write("Paste any long text and get a focused summary with your desired word limit 🔍")

# User input
text = st.text_area("📝 Paste your long text below:", height=300)

# Word limit slider
word_limit = st.slider("✂ Desired summary length (word count):", 30, 200, 100)

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

def trim_or_expand_summary(summary_text, word_limit):
    words = summary_text.split()
    if len(words) > word_limit:
        # Trim
        return ' '.join(words[:word_limit]) + "..."
    elif len(words) < word_limit:
        # Pad by repeating last sentence or text (simple approach)
        while len(words) < word_limit:
            words += words[-min(10, len(words)):]  # repeat last few words
        return ' '.join(words[:word_limit]) + "..."
    else:
        return summary_text

if st.button("🚀 Generate Focused Summary"):
    if not text.strip():
        st.warning("⚠️ Please paste some content to summarize.")
    else:
        with st.spinner("✨ Generating summary..."):
            cleaned = clean_text(text)

            # Add prompt for better focus + word limit
            prompt_text = (
                f"Summarize this text in about {word_limit} words, focusing on: importance of time, student life, mindfulness, and purpose.\n\n" + cleaned
            )

            # Since max_length/min_length are in tokens (≈1.3 tokens per word)
            max_tokens = int(word_limit * 1.5)
            min_tokens = int(word_limit * 1.1)

            try:
                result = summarizer(
                    prompt_text,
                    max_length=max_tokens,
                    min_length=min_tokens,
                    do_sample=False
                )
                raw_summary = result[0]['summary_text']

                # Adjust summary to exact word limit
                final_summary = trim_or_expand_summary(raw_summary, word_limit)

                st.subheader("🧠 Summary Output:")
                st.success(final_summary)

                key_points = extract_key_points(final_summary)
                if key_points:
                    st.subheader("📌 Key Points:")
                    for point in key_points:
                        st.markdown(point)
                else:
                    st.info("Couldn’t extract key points (summary might be too short).")

            except Exception as e:
                st.error(f"❌ Error generating summary: {e}")

st.markdown("---")
st.markdown("✅ Built with [Streamlit](https://streamlit.io) & 🤗 HuggingFace Transformers")
