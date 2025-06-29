import streamlit as st
from transformers import pipeline
from pdfminer.high_level import extract_text

st.set_page_config(page_title="GenAI Research Assistant", layout="wide")
st.title("📄 Smart Assistant for Document Analysis")
st.markdown("Upload a PDF or TXT file and interact with it!")

uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        text = extract_text("temp.pdf")
    else:
        text = uploaded_file.read().decode("utf-8")

    st.success("✅ File uploaded and text extracted.")

    # Limit for summarization by character count
    max_chars = 1000
    small_text = text[:max_chars]

    st.info(f"Document contains {len(text)} characters. Showing summary of first {max_chars} characters.")

    # Generate summary safely
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(small_text, max_length=150, min_length=40, do_sample=False)
    st.subheader("📌 Auto Summary")
    st.write(summary[0]["summary_text"])

    # Ask Anything
    st.subheader("🤔 Ask Anything from Document")
    question = st.text_input("Type your question:")
    if question:
        qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        answer = qa(question=question, context=text)
        st.write("**Answer:**", answer["answer"])

    # Challenge Me Mode
    st.subheader("🎯 Challenge Me (Auto Questions)")
    if st.button("Generate Questions"):
        generator = pipeline("text-generation", model="gpt2")
        prompt = f"Generate 3 comprehension questions from this:\n{text[:1000]}"
        questions = generator(prompt, max_length=200, num_return_sequences=1)
        st.write("**Questions:**")
        st.write(questions[0]["generated_text"].replace(prompt, "").strip())
