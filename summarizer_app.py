import streamlit as st
from transformers import pipeline
import re

# App Title
st.set_page_config(page_title="AI Text Summarizer", layout="centered")
st.title("📚 Smart AI Text Summarizer")
st.write("Paste any long text and get a meaningful, short and sharp summary 🔍")

# User input for text
text = st.text_area("📝 Paste your long text below:", height=300)

# User selects desired word count
word_limit = st.slider("✂ How many words should the summary be?", 30, 200, 80)

# Load summarizer model (cached)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def clean_text(text):
    # Remove extra whitespaces
    return re.sub(r'\s+', ' ', text.strip())

def extract_key_points(summary_text):
    # Basic splitting into bullet points
    sentences = re.split(r'(?<=[.!?]) +', summary_text)
    points = [f"• {s.strip()}" for s in sentences if len(s.strip()) > 30]
    return points

# On button click
if st.button("🚀 Generate Summary"):
    if not text.strip():
        st.warning("Please paste some content to summarize.")
    else:
        with st.spinner("Generating smart summary..."):
            cleaned_text = clean_text(text)
            max_len = int(word_limit * 1.3)
            min_len = int(word_limit * 0.6)

            result = summarizer(cleaned_text, max_length=max_len, min_length=min_len, do_sample=False)
            summary_text = result[0]['summary_text']

            st.subheader("🧠 Summary Output:")
            st.success(summary_text)

            st.subheader("📌 Key Points:")
            for point in extract_key_points(summary_text):
                st.markdown(point)

st.markdown("---")
st.markdown("💡 Built with [Streamlit](https://streamlit.io) & HuggingFace Transformers")