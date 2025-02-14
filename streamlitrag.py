__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langchain.chains import create_retrieval_chain
from langchain_core.tools import tool

from utils.preprocess import load_data, split_data, upsert_chromadb
from utils.prebuilt_chain import history_aware_retriever, documents_retriever

import streamlit as st

db_name = "chroma" # default name for Chromadb

st.set_page_config(page_title="RAG Demo App")
st.title("Retrieval Augmented Generation With LanghChain & Chroma")

@st.cache_resource
def load_model(api_key):
    """cached llm and embedding model"""
    st.session_state.llm = ChatOpenAI(model="llama-3.1-8b-instant", temperature=0.3, api_key=api_key, base_url="https://api.groq.com/openai/v1")
    st.session_state.embedding = FastEmbedEmbeddings(
                batch_size=64,
                model_name="jinaai/jina-embeddings-v2-base-de"
                )   

def inputs():
    """Input fields for user interaction"""
    with st.sidebar:
        st.session_state.groq_api_key = st.text_input("GroQ API Key", type="password")
        os.environ["OPENAI_API_KEY"] = st.session_state.groq_api_key
        
        st.session_state.chroma_collection_name = st.text_input("Chroma Collection Name")

    st.session_state.source_docs = st.file_uploader("Unggah file PDF", type=["pdf"], accept_multiple_files=True)
    st.button("Proses Dokumen", on_click=process_data)

def process_data():
    """Main function to process data"""
    if not st.session_state.groq_api_key or not st.session_state.chroma_collection_name or not st.session_state.source_docs:
        st.error("Tolong masukan API key, Chroma collection name, dan dokumen yang diperlukan!!")
    else:
        with st.spinner("📚 Memproses dokumen..."):
            loaded_docs = load_data(st.session_state.source_docs)
            splitted_docs = split_data(loaded_docs)

            idx = [str(uuid4()) for _ in range(len(splitted_docs))]

            st.session_state.vector_store = upsert_chromadb(splitted_docs,
                                                            st.session_state.embedding,
                                                            idx,
                                                            st.session_state.chroma_collection_name,
                                                            db_name)
        msg = st.empty()
        msg.success("Dokumen berhasil diproses!")
        time.sleep(3)
        msg.empty()

# Main retriever
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query.
    
    Args:
        query: The user's query.
    """
    retrieved_docs = st.session_state.vector_store.similarity_search(query, k=6)
    keys = ["author", "creator", "page", "source", "start_index", "total_pages"]
    serialized = "\n\n".join(
        (f"Source: {[{key: doc.metadata.get(key)} for key in keys]}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def generate(query):
    """Generate a response to the user's query."""
    # Dummy retriever.
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k" : 1})

    # Create a RAG chain using the history-aware retriever and the document-retriever.
    history_retriever = history_aware_retriever(st.session_state.llm, retriever)
    question_answer_chain = documents_retriever(st.session_state.llm)

    rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)

    # Usage:
    response = rag_chain.invoke({"input": query, "chat_history" : st.session_state.messages, "context" : retrieve.invoke(query)})
    st.session_state.messages.append(query)
    st.session_state.messages.append(response["answer"])
    return response["answer"]

if __name__ == "__main__":
    os.makedirs(db_name, exist_ok=True) # This directory is used to store persistent files from Chromadb

    inputs()
    load_model(os.getenv("OPENAI_API_KEY"))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if st.session_state.messages:
        st.chat_message('human').write(st.session_state.messages[-2])
        st.chat_message('ai').write(st.session_state.messages[-1])

    query = st.chat_input("Masukkan Prompt")
    if query:
        st.chat_message("human").write(query)
        response = generate(query)
        st.chat_message("ai").write(response)
