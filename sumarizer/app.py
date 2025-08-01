import streamlit as st
from transformers import pipeline
import re

# App Title & Config
st.set_page_config(page_title="AI Text Summarizer", layout="centered")
st.title("📚 Smart AI Text Summarizer")
st.write("Paste any long text and get a meaningful, short and sharp summary 🔍")

# User input
text = st.text_area("📝 Paste your long text below:", height=300)

# Desired summary length
word_limit = st.slider("✂ Desired summary length (approx. words):", 30, 200, 80)

# Load summarizer model (cached)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def clean_text(text):
    # Remove extra whitespaces & special chars if needed
    return re.sub(r'\s+', ' ', text.strip())

def extract_key_points(summary_text):
    # Split into bullet points by sentence
    sentences = re.split(r'(?<=[.!?]) +', summary_text)
    return [f"• {s.strip()}" for s in sentences if len(s.strip()) > 30]

# On click
if st.button("🚀 Generate Summary"):
    if not text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        with st.spinner("Generating summary... please wait!"):
            cleaned_text = clean_text(text)
            
            # Slightly adjust lengths to guide summarizer
            max_len = int(word_limit * 1.4)
            min_len = int(word_limit * 0.6)

            try:
                result = summarizer(
                    cleaned_text, 
                    max_length=max_len, 
                    min_length=min_len, 
                    do_sample=False
                )
                summary_text = result[0]['summary_text']

                st.subheader("🧠 Summary Output:")
                st.success(summary_text)

                # Show key points if summary is long enough
                if len(summary_text.split()) > 25:
                    st.subheader("📌 Key Points:")
                    for point in extract_key_points(summary_text):
                        st.markdown(point)
                else:
                    st.info("Summary is already quite short, no key points extracted.")

            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")

st.markdown("---")
st.caption("💡 Built with [Streamlit](https://streamlit.io) & 🤗 HuggingFace Transformers")
