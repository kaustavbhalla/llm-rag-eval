from langchain_community.vectorstores.starrocks import Metadata
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
import torch
import pdfplumber
import streamlit as st

@st.cache_resource(show_spinner=False)
def getEmbeddingModel():
    hardware = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceBgeEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": hardware}
    )


def getContextFromFiles(uploaded_files) -> list[Document]:
    data = []
    for f in uploaded_files:
        if f.name.endswith(".pdf"):
            with pdfplumber.open(f) as pdfFile:
                text = "\n\n".join(page.extract_text() or "error extracting pdf" for page in pdfFile.pages)
        else:
            text = f.read().decode("utf-8")


        if text.strip():
            data.append(Document(page_content=text, metadata={"source": f.name}))


    return data


def chunkAndBuildVectors(docs: list[Document], chunk_size: int = 512) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = max(64, chunk_size // 8)
    )

    chunks = splitter.split_documents(docs)
    embedder = getEmbeddingModel()

    return FAISS.from_documents(chunks, embedder)


def retrieveContext(vectorStore: FAISS, question: str, k: int = 4) -> str:
    retrieval = vectorStore.similarity_search(question, k)
    return "\n\n----\n\n".join(r.page_content for r in retrieval)


        
