import streamlit as st
from transformers import pipeline
import re
import PyPDF2
import docx

# App config & title
st.set_page_config(page_title="🧠 Smart AI Text Summarizer", layout="centered")
st.title("📚 Smart AI Text Summarizer")
st.write("Upload any document or paste text and get a focused summary with your desired word limit 🔍")

# Upload section
st.subheader("📂 Upload your document (TXT, PDF, DOCX)")
uploaded_file = st.file_uploader("Choose a file to upload:", type=["txt", "pdf", "docx"])

text = ""

if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "type": uploaded_file.type, "size": uploaded_file.size}
    st.info(f"✅ Uploaded: `{file_details['filename']}` ({round(file_details['size']/1024, 2)} KB)")
    
    try:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# If no file uploaded, use text area
if not text:
    st.subheader("📝 Or paste your text below:")
    text = st.text_area("Paste your long text here:", height=300)

# Word limit & focus points
word_limit = st.slider("✂ Desired summary length (word count):", 30, 200, 100)
focus_points = st.text_input("🎯 Optional: Focus on specific themes (comma-separated)", 
                             value="importance of time, student life, mindfulness, purpose")

# Load summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

with st.spinner("🔄 Loading summarization model..."):
    summarizer = load_summarizer()

# Utilities
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def extract_key_points(summary_text):
    sentences = re.split(r'(?<=[.!?]) +', summary_text)
    points = [f"• {s.strip()}" for s in sentences if len(s.strip()) > 30]
    return points

def trim_or_expand_summary(summary_text, word_limit):
    words = summary_text.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + "..."
    elif len(words) < word_limit:
        last_sentence = re.split(r'(?<=[.!?]) +', summary_text.strip())[-1]
        while len(words) < word_limit:
            words += last_sentence.split()[:min(10, len(last_sentence.split()))]
        return ' '.join(words[:word_limit]) + "..."
    else:
        return summary_text

# Generate summary
if st.button("🚀 Generate Focused Summary", disabled=not text.strip()):
    with st.spinner("✨ Generating summary..."):
        try:
            cleaned = clean_text(text)
            prompt_text = f"Summarize this text in about {word_limit} words, focusing on: {focus_points}.\n\n{cleaned}"
            max_tokens = int(word_limit * 1.5)
            min_tokens = int(word_limit * 1.1)

            result = summarizer(
                prompt_text,
                max_length=max_tokens,
                min_length=min_tokens,
                do_sample=False,
                truncation=True
            )
            raw_summary = result[0]['summary_text']
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

            # Optional: add download button
            st.download_button(
                label="💾 Download summary as TXT",
                data=final_summary,
                file_name="summary.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Error generating summary: {e}")

st.markdown("---")
st.markdown("✅ Built with [Streamlit](https://streamlit.io) & 🤗 HuggingFace Transformers")


