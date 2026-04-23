import os
from deep_translator import GoogleTranslator

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore


# -----------------------------
# Translate Text
# -----------------------------
def translate_text(text, lang):
    lang_codes = {
        "English": "en",
        "Hindi": "hi",
        "Gujarati": "gu"
    }

    try:
        return GoogleTranslator(
            source="auto",
            target=lang_codes[lang]
        ).translate(text)

    except:
        return text


# -----------------------------
# Create Vector Store
# -----------------------------
def create_vectorstore(pdf_path):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Keep only pages with text
    clean_docs = []

    for doc in docs:
        if doc.page_content.strip():
            clean_docs.append(doc)

    if len(clean_docs) == 0:
        raise Exception("No readable text found in PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(clean_docs)

    if len(chunks) == 0:
        raise Exception("No chunks created from PDF.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = SKLearnVectorStore.from_documents(
        chunks,
        embeddings
    )

    return db