import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import pickle
from openai import OpenAI

# ======================
# 1. CONFIGURATION
# ======================
# Set up your OpenAI client using your API key.
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("❌ OPENAI_API_KEY is missing. Please add it in Streamlit Secrets.")
    st.stop()

# Define file paths for pre-loaded documents (if available)
INDEX_PATH = "vector.index"
DOCS_METADATA_PATH = "docs_metadata.pkl"
SIMILARITY_THRESHOLD = 0.75
K_RETRIEVE = 3

# ======================
# 2. LOAD PRE-LOADED DOCUMENTS
# ======================
@st.cache_resource
def load_preloaded_index():
    # Try to load the pre-built FAISS index and metadata
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_METADATA_PATH):
        st.warning("Pre-loaded index files not found. The app will use only uploaded documents.")
        return None, []
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# ======================
# 3. EMBEDDING FUNCTION
# ======================
def get_embedding(text):
    # Get the text embedding from OpenAI
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# ======================
# 4. PROCESS USER-UPLOADED DOCUMENTS
# ======================
@st.cache_data
def process_uploaded_documents(uploaded_files):
    """
    This function reads the uploaded files, extracts their text, and creates a FAISS index.
    It returns both the index and a list of document metadata.
    """
    user_docs = []
    embeddings = []
    for file in uploaded_files:
        # For simplicity, assume the file is a text file. For PDFs or Word docs,
        # you would need additional processing.
        content = file.read().decode("utf-8")
        doc = {"text": content, "source": file.name}
        user_docs.append(doc)
        emb = get_embedding(content)
        embeddings.append(emb)
    if embeddings:
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    else:
        index = None
    return index, user_docs

# ======================
# 5. SEARCH FUNCTION
# ======================
def search_documents(query, pre_index, pre_docs, user_index, user_docs, threshold=SIMILARITY_THRESHOLD, k=K_RETRIEVE):
    """
    This function searches both pre-loaded and user-uploaded document indices for relevant text.
    It returns a list of matching document chunks.
    """
    query_vector = get_embedding(query).reshape(1, -1)
    results = []

    # Search in pre-loaded documents if available
    if pre_index is not None:
        D, I = pre_index.search(query_vector, k)
        for i, dist in zip(I[0], D[0]):
            if dist < threshold:
                results.append(pre_docs[i])
                
    # Search in user-uploaded documents if available
    if user_index is not None:
        D, I = user_index.search(query_vector, k)
        for i, dist in zip(I[0], D[0]):
            if dist < threshold:
                results.append(user_docs[i])
    return results

# ======================
# 6. ANSWER GENERATION FUNCTION
# ======================
def answer_query(query, pre_index, pre_docs, user_index, user_docs, include_explanation=True):
    """
    This function builds a prompt for the language model based on search results.
    If matching documents are found, it includes their text as context; otherwise, it falls back on general LLM knowledge.
    """
    results = search_documents(query, pre_index, pre_docs, user_index, user_docs)
    if results:
        context = "\n\n".join([r["text"] for r in results])
        if include_explanation:
            prompt = f"""
Based on the following reference text, provide a direct and detailed answer to the question using Markdown formatting.

**Direct Answer**
- Start with a clear recommendation

**Explanation**
- Explain the rationale using bullet points
- Include any relevant considerations

**References**
- Mention file names if applicable

Reference:
{context}

Question: {query}
"""
        else:
            prompt = f"""
Use the reference text below to provide a direct, concise answer.

Reference:
{context}

Question: {query}
"""
        source = "Retrieved Documents"
    else:
        prompt = f"""
No relevant document was found. Using general knowledge, provide a direct, evidence-based answer to the following question:

Question: {query}

Please note: This answer is generated based on general knowledge and does not reference specific guidelines.
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

# ======================
# 7. USER INTERFACE (UI)
# ======================
st.set_page_config("PharmInsight - Customizable Document Search", layout="centered")
st.title("PharmInsight for Pharmacists")
st.markdown("Ask your questions related to clinical guidelines, policies, or any relevant topics.")

# Option to upload additional documents
st.markdown("### Upload Your Documents (Optional)")
uploaded_files = st.file_uploader("Upload documents (text files)", accept_multiple_files=True, type=["txt"])
# (Note: To support PDFs or Word files, you’d need to add file parsing logic.)

# Process the user-uploaded documents if any files are provided
user_index, user_docs = (None, [])
if uploaded_files:
    user_index, user_docs = process_uploaded_documents(uploaded_files)

# Load pre-loaded documents from your local storage (if available)
pre_index, pre_docs = load_preloaded_index()

# Input area for the user query
st.markdown("### Enter Your Query")
query = st.text_input("Type your question here", placeholder="e.g., What is the recommended monitoring plan for amiodarone?")

# Option to include a detailed explanation and references
include_explanation = st.checkbox("Include explanation and references", value=True)

# When the user clicks 'Submit', process the query
if st.button("Submit") and query:
    with st.spinner("Searching documents and generating answer..."):
        result = answer_query(query, pre_index, pre_docs, user_index, user_docs, include_explanation=include_explanation)
    
    if result["source"] == "Retrieved Documents":
        st.success("Answer based on the provided documents:")
    else:
        st.warning("No matching document found; answer generated from general knowledge:")
    
    st.markdown(result["answer"], unsafe_allow_html=True)
    
    # Optionally, show the sources used for the answer
    if result["results"]:
        with st.expander("📄 Sources Used"):
            for idx, doc in enumerate(result["results"]):
                st.markdown(f"**Source: {doc['source']}**")
                st.text(doc["text"])
