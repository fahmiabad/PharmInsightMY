import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import pickle
from openai import OpenAI
from PyPDF2 import PdfReader  # Make sure to install PyPDF2 (pip install PyPDF2)

# ============================================
# 1. CONFIGURATION AND OPENAI CLIENT SETUP
# ============================================
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("❌ OPENAI_API_KEY is missing. Please add it in Streamlit Secrets.")
    st.stop()

INDEX_PATH = "vector.index"
DOCS_METADATA_PATH = "docs_metadata.pkl"
SIMILARITY_THRESHOLD = 0.75
K_RETRIEVE = 3

# ============================================
# 2. LOADING PRE‑LOADED DOCUMENTS (if available)
# ============================================
@st.cache_resource
def load_preloaded_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_METADATA_PATH):
        st.warning("Pre‑loaded index files not found. The app will use only uploaded documents.")
        return None, []
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# ============================================
# 3. GET EMBEDDING FUNCTION (with error handling)
# ============================================
def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"  # Updated to a known, supported model
        )
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        raise
    return np.array(response.data[0].embedding, dtype=np.float32)

# ============================================
# 4. PROCESS USER‑UPLOADED DOCUMENTS
# ============================================
@st.cache_data
def process_uploaded_documents(uploaded_files):
    """
    Reads the uploaded files, extracts their text, and creates a FAISS index.
    Handles both text files and PDF files.
    """
    user_docs = []
    embeddings = []
    for file in uploaded_files:
        file_name = file.name.lower()
        if file_name.endswith(".pdf"):
            # Parse PDF file using PyPDF2
            try:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            except Exception as e:
                st.error(f"Error reading PDF file {file.name}: {e}")
                continue
        else:
            # Assuming a text file
            try:
                text = file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading file {file.name}: {e}")
                continue
        
        doc = {"text": text, "source": file.name}
        user_docs.append(doc)
        emb = get_embedding(text)
        embeddings.append(emb)
    
    if embeddings:
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    else:
        index = None
    return index, user_docs

# ============================================
# 5. SEARCH FUNCTION ACROSS DOCUMENTS
# ============================================
def search_documents(query, pre_index, pre_docs, user_index, user_docs, threshold=SIMILARITY_THRESHOLD, k=K_RETRIEVE):
    """
    Searches both pre‑loaded and user‑uploaded document indices for relevant text.
    """
    query_vector = get_embedding(query).reshape(1, -1)
    results = []

    # Search in pre‑loaded documents (if available)
    if pre_index is not None:
        D, I = pre_index.search(query_vector, k)
        for i, dist in zip(I[0], D[0]):
            if dist < threshold:
                results.append(pre_docs[i])
                
    # Search in user‑uploaded documents (if available)
    if user_index is not None:
        D, I = user_index.search(query_vector, k)
        for i, dist in zip(I[0], D[0]):
            if dist < threshold:
                results.append(user_docs[i])
    return results

# ============================================
# 6. ANSWER GENERATION FUNCTION
# ============================================
def answer_query(query, pre_index, pre_docs, user_index, user_docs, include_explanation=True):
    results = search_documents(query, pre_index, pre_docs, user_index, user_docs)
    if results:
        context = "\n\n".join([r["text"] for r in results])
        if include_explanation:
            prompt = f"""
Read the following document excerpts carefully and analyze the content. Then, provide a direct answer with a clear recommendation and an explanation based on the excerpts.

**Direct Recommendation:**
- Begin with your clear recommendation.

**Explanation:**
- Bullet the key reasons and insights.
- Synthesize the relevant points from the excerpts.

**Document Excerpts:**
{context}

**Question:**
{query}
"""
        else:
            prompt = f"""
Read the document excerpts below and extract the most relevant information to provide a direct answer.

**Document Excerpts:**
{context}

**Question:**
{query}
"""
        source = "Retrieved Documents"
    else:
        prompt = f"""
No relevant document was found. Using general knowledge, provide a direct, evidence-based answer to the following question:

Question: {query}

Note: This answer is generated based on general knowledge and does not reference specific guidelines.
"""
        source = "LLM General Knowledge"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )
    answer = response.choices[0].message.content
    return {"answer": answer, "source": source, "results": results}

# ============================================
# 7. USER INTERFACE (UI) WITH STREAMLIT
# ============================================
st.set_page_config("PharmInsight - Customizable Document Search", layout="centered")
st.title("PharmInsight for Pharmacists")
st.markdown("Ask your questions related to clinical guidelines, policies, or any relevant topics.")

# Allow users to upload additional documents (text files and PDFs)
st.markdown("### Upload Your Documents (Optional)")
uploaded_files = st.file_uploader("Upload documents (text or PDF files)", accept_multiple_files=True, type=["txt", "pdf"])

user_index, user_docs = (None, [])
if uploaded_files:
    user_index, user_docs = process_uploaded_documents(uploaded_files)

pre_index, pre_docs = load_preloaded_index()

st.markdown("### Enter Your Query")
query = st.text_input("Type your question here", placeholder="e.g., What is the recommended monitoring plan for amiodarone?")

include_explanation = st.checkbox("Include explanation and references", value=True)

if st.button("Submit") and query:
    with st.spinner("Searching documents and generating answer..."):
        result = answer_query(query, pre_index, pre_docs, user_index, user_docs, include_explanation=include_explanation)
    
    if result["source"] == "Retrieved Documents":
        st.success("Answer based on the provided documents:")
    else:
        st.warning("No matching document found; answer generated from general knowledge:")
    
    st.markdown(result["answer"], unsafe_allow_html=True)
    
    if result["results"]:
        with st.expander("📄 Sources Used"):
            for idx, doc in enumerate(result["results"]):
                st.markdown(f"**Source: {doc['source']}**")
                st.text(doc["text"])
