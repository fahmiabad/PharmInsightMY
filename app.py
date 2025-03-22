import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import pickle
from openai import OpenAI

# CONFIG
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("❌ OPENAI_API_KEY is missing. Please add it in Streamlit Secrets.")
    st.stop()

INDEX_PATH = "vector.index"
DOCS_METADATA_PATH = "docs_metadata.pkl"
EMAIL_DB = "emails.csv"
SIMILARITY_THRESHOLD = 0.75
K_RETRIEVE = 3

# Load FAISS index and metadata
@st.cache_resource
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_METADATA_PATH):
        st.error("❌ Required index files are missing. Please upload vector.index and docs_metadata.pkl.")
        st.stop()
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Get embedding using OpenAI SDK v1.x
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# RAG logic
def answer_query_with_rag(query, index, docs, threshold=SIMILARITY_THRESHOLD, k=K_RETRIEVE):
    query_vector = get_embedding(query).reshape(1, -1)
    D, I = index.search(query_vector, k=k)
    matched_chunks = [docs[i] for i, dist in zip(I[0], D[0]) if dist < threshold]

    if matched_chunks:
        context = "\n\n".join([c['text'] for c in matched_chunks])
        prompt = f"""
You are a clinical pharmacist. Use ONLY the context below to answer the question accurately.

Context:
{context}

Question: {query}

If the answer is not found in the context, say: "Information not available in the uploaded guidelines."
"""
        source = "guidelines"
    else:
        prompt = f"""
You are a clinical pharmacist. The uploaded guidelines do not contain relevant information for the question below.

Question: {query}

Provide an evidence-based response from your general knowledge, and clearly state that this information is *not found in the uploaded guidelines*.
"""
        source = "llm_fallback"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    answer = response.choices[0].message.content
    return {"answer": answer, "source": source, "chunks": matched_chunks}

# Email collection
def save_email(email):
    if not os.path.exists(EMAIL_DB):
        pd.DataFrame(columns=["email"]).to_csv(EMAIL_DB, index=False)
    df = pd.read_csv(EMAIL_DB)
    if email not in df["email"].values:
        new_row = pd.DataFrame([{"email": email}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(EMAIL_DB, index=False)

# UI starts
st.set_page_config("PharmInsightMY by BigPharmi", layout="centered")
st.title("🧠 PharmInsightMY by BigPharmi")
st.markdown("_Ask questions related to treatment, dosing, monitoring, and more._")

st.markdown("""
<div style='background-color: #fff3cd; padding: 10px; border-left: 6px solid #ffecb5;'>
⚠️ <strong>Disclaimer:</strong> This app is for <strong>research and educational purposes only</strong>. It should <strong>not</strong> be used for making actual clinical decisions on patients. 
For feedback or collaboration, contact: <a href='mailto:fahmibinabad@gmail.com'>fahmibinabad@gmail.com</a>
</div>
""", unsafe_allow_html=True)

# Email gate
if "email" not in st.session_state:
    email = st.text_input("Enter your email to access the tool:")
    if st.button("Continue"):
        if "@" in email:
            st.session_state.email = email
            save_email(email)
            st.experimental_rerun()
        else:
            st.error("Please enter a valid email address.")
    st.stop()

# Load index and ask questions
index, docs = load_index()

st.success(f"Welcome, {st.session_state.email} 👋")
query = st.text_input("Ask a clinical question:", placeholder="e.g., Monitoring plan for amiodarone?")

if st.button("Submit") and query:
    with st.spinner("Searching uploaded documents..."):
        result = answer_query_with_rag(query, index, docs)

    if result["source"] == "guidelines":
        st.success("✅ Answer from uploaded guidelines:")
    else:
        st.warning("⚠️ Not found in uploaded guidelines. Answer from general knowledge:")

    st.markdown(result["answer"])

    if result["chunks"]:
        with st.expander("📄 Sources used"):
            for i, chunk in enumerate(result["chunks"]):
                src = chunk.get("source", "unknown")
                page = f"(Page {chunk['page']})" if chunk.get("page") else ""
                st.markdown(f"**{src} {page}**")
                st.text(chunk["text"])
