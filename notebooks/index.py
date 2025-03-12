import streamlit as st
import ollama
from langchain.graphs import Neo4jGraph

def retrieve_relevant_chunks(query_text):
    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
    cypher_query = """
    MATCH (c:Chunk)
    WHERE c.text CONTAINS $query_text
    RETURN c.text LIMIT 5
    """
    results = graph.query(cypher_query, {"query_text": query_text})
    return [record["c.text"] for record in results]

def legal_ai_response(query):
    context = retrieve_relevant_chunks(query)
    prompt = f"Use the following legal context to answer the query: {context}\nQuery: {query}"
    response = ollama.chat(model="meta/llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Streamlit UI
st.set_page_config(page_title="Legal AI Assistant", layout="centered")
st.title("📜 Legal AI Assistant")
st.markdown("""
### 🤖 Ask your legal queries and get AI-powered responses!
Enter your legal question below, and our AI will retrieve relevant legal texts and provide an informed response.
""")

query = st.text_area("Enter your legal question:", "What are the legal principles in a contract breach?")
if st.button("Get Answer", use_container_width=True):
    with st.spinner("Analyzing legal texts..."):
        response = legal_ai_response(query)
    st.success("✅ Response generated!")
    st.write(response)

st.markdown("---")
st.markdown("🛠️ *Powered by LLaMA 3.1, NetworkX, and Graph RAG*")
