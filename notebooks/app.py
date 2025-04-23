import streamlit as st
import pandas as pd
import torch
import time
import psutil
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
from langchain_ollama import OllamaLLM

# Load data
@st.cache_resource
def load_data():
    qa_df = pd.read_csv("../data/processed/Questions & Answers.csv")
    chunked_cases_df = pd.read_csv("chunked_law_cases.csv")
    return qa_df, chunked_cases_df

qa_df, chunked_cases_df = load_data()

# Load model and embeddings
@st.cache_resource
def load_models_and_embeddings():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_texts = chunked_cases_df["text"].tolist()
    chunk_embeddings = embedder.encode(chunk_texts, convert_to_tensor=True)
    llm = OllamaLLM(model="llama3.1")
    return embedder, chunk_embeddings, llm

embedder, chunk_embeddings, llm = load_models_and_embeddings()

# ARAD Multi-Hop Function
def multi_hop_arad(question, k=5, max_hops=2):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    similarities = cos_sim(question_embedding, chunk_embeddings)[0]
    top_k_indices = torch.topk(similarities, k).indices.tolist()
    context = " ".join(chunked_cases_df.iloc[i]["text"] for i in top_k_indices)

    full_context = context

    if max_hops > 1:
        prompt = f"Question: {question}\nRelevant Law Context: {context}\nAnswer:"
        intermediate_answer = llm.invoke(prompt)
        answer_embedding = embedder.encode(intermediate_answer, convert_to_tensor=True)
        new_similarities = cos_sim(answer_embedding, chunk_embeddings)[0]
        new_indices = torch.topk(new_similarities, k).indices.tolist()
        second_hop_context = " ".join(chunked_cases_df.iloc[i]["text"] for i in new_indices)
        full_context += "\n" + second_hop_context

    final_prompt = f"Question: {question}\n\nContext:\n{full_context}\n\nAnswer:"

    cpu_before = psutil.cpu_percent(interval=None)
    start_time = time.time()
    answer = llm.invoke(final_prompt)
    elapsed_time = time.time() - start_time
    cpu_after = psutil.cpu_percent(interval=None)

    return answer, elapsed_time, cpu_after

# Streamlit UI
st.title("🔍 Legal Query Assistant - LLaMA 3.1 (ARAD Zero-Shot)")
query = st.text_area("Type your legal question:", height=150)

if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating response..."):
            answer, time_taken, cpu = multi_hop_arad(query, k=5, max_hops=2)
            st.success(f"✅ Response generated in {time_taken:.2f} seconds (CPU usage: {cpu}%)")
            st.markdown("### 📘 Answer")
            st.write(answer)