import streamlit as st
import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from utils import create_vectorstore, translate_text


# -----------------------------
# Load ENV
# -----------------------------
load_dotenv()


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🩺",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🩺 MediAssist AI")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

language = st.sidebar.selectbox(
    "Choose Language",
    ["English", "Hindi", "Gujarati"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Medical PDF",
    type="pdf"
)


# -----------------------------
# Upload PDF
# -----------------------------
if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF..."):

        st.session_state.db = create_vectorstore(file_path)

    st.sidebar.success("PDF Uploaded Successfully!")


# -----------------------------
# AI Symptom Checker
# -----------------------------
st.subheader("🩺 AI Symptom Checker")

symptoms = st.text_input(
    "Enter symptoms (example: fever, cough, weakness, headache, vomiting)"
)

if symptoms:

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    prompt = f"""
    You are a safe AI medical assistant.

    A user entered these symptoms:
    {symptoms}

    Common symptoms users may report include:
    fever, cough, weakness, headache, cold, sore throat,
    vomiting, nausea, stomach pain, loose motion, diarrhea,
    chest pain, body pain, fatigue, dizziness, breathlessness,
    runny nose, chills, sneezing, back pain, joint pain,
    burning urination, loss of appetite, rash, high sugar,
    thirst, frequent urination.

    Analyze the symptoms and give:

    Possible Causes:
    Precautions:
    Urgency Level:
    General Care:
    When to See Doctor:
    Disclaimer:

    Keep response short and safe.
    """

    with st.spinner("Analyzing Symptoms..."):

        response = llm.invoke(prompt)

        answer = response.content

        if language != "English":
            answer = translate_text(answer, language)

        st.info(answer)


# -----------------------------
# Chat Section
# -----------------------------
st.subheader("💬 Ask Questions")

query = st.text_input(
    "Ask question from uploaded PDF"
)


# -----------------------------
# Ask Button
# -----------------------------
if st.button("Submit"):

    if not query:
        st.warning("Please enter a question.")
        st.stop()

    if "db" not in st.session_state:
        st.error("Please upload PDF first.")
        st.stop()

    with st.spinner("Thinking..."):

        db = st.session_state.db

        retriever = db.as_retriever(
            search_kwargs={"k": 3}
        )

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.2
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa.invoke({"query": query})

        answer = result["result"]

        if language != "English":
            answer = translate_text(answer, language)

        st.success(answer)

        st.subheader("📚 Sources")

        for doc in result["source_documents"]:
            st.info(doc.metadata["source"])