import streamlit as st
import os
import io
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import openai

# --- Functions for PDF Extraction and Guideline Processing ---

def extract_text_from_pdf(file_obj):
    """
    Extracts text from a PDF file-like object.
    """
    reader = PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def load_local_guidelines(directory="pdf_guidelines"):
    """
    Loads guideline PDFs from a specified directory and converts them to text.
    Returns a dictionary mapping filename to text.
    """
    guidelines = {}
    if not os.path.exists(directory):
        st.warning(f"Directory '{directory}' not found. Please create it and add PDF guidelines.")
        return guidelines

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            with open(pdf_path, "rb") as f:
                text = extract_text_from_pdf(f)
            guidelines[filename] = text
    return guidelines

def build_faiss_index(guidelines_dict, model):
    """
    Computes embeddings for the guideline texts and builds a FAISS index.
    Returns the index, list of texts, and filenames.
    """
    texts = list(guidelines_dict.values())
    filenames = list(guidelines_dict.keys())
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, texts, filenames

def search_guidelines(query, model, index, texts, filenames, threshold=1.0):
    """
    Encodes the query, searches the FAISS index, and checks if the best match is below a threshold.
    Returns a tuple indicating if a match was found, the source filename, content, and the distance.
    """
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k=1)
    best_distance = D[0][0]
    best_index = I[0][0]
    if best_distance < threshold:
        return True, filenames[best_index], texts[best_index], best_distance
    else:
        return False, None, None, best_distance

# --- LLM Fallback Function ---

def get_llm_response(query):
    """
    Calls the OpenAI API to get a response for the query.
    The API key is retrieved securely from Streamlit secrets.
    """
    openai.api_key = st.secrets["openai"]["api_key"]

    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert pharmacy assistant providing evidence-based information."},
                {"role": "user", "content": query}
            ]
        )
        answer = completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"Error contacting LLM: {e}"

# --- Streamlit App ---

def main():
    st.title("PharmInsightMY")
    st.markdown("""
        **Heads Up, Fam:**
        This app is strictly for research purposes only.
        Do **NOT** rely on the info provided for making clinical decisions.
        For more deets or inquiries, hit up: [fahmibinabad@gmail.com](mailto:fahmibinabad@gmail.com).
    """)

    # Session State Initialization
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'texts' not in st.session_state:
        st.session_state.texts = []
    if 'filenames' not in st.session_state:
        st.session_state.filenames = []
    if 'uploaded_guidelines' not in st.session_state:
        st.session_state.uploaded_guidelines = {}

    user_email = st.text_input("Enter your email (for future updates):", value=st.session_state.user_email)
    if user_email:
        st.session_state.user_email = user_email
    st.write("Your email is:", st.session_state.user_email)

    # Initialize the embedding model (only once)
    if st.session_state.model is None:
        with st.spinner("Initializing embedding model..."):
            st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load guidelines from the local 'pdf_guidelines' folder
    local_guidelines = load_local_guidelines()

    # Option for the user to upload their own PDF guidelines
    st.header("Upload Your Own PDF Guidelines (Optional)")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                file_bytes = io.BytesIO(uploaded_file.read())
                text = extract_text_from_pdf(file_bytes)
                st.session_state.uploaded_guidelines[uploaded_file.name] = text
                st.success(f"Uploaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    # Merge local and uploaded guidelines
    all_guidelines = {**local_guidelines, **st.session_state.uploaded_guidelines}
    if not all_guidelines:
        st.error("No guidelines available. Please add PDF guidelines locally or upload via the app.")
        return

    # Build the FAISS index (only once or when guidelines change)
    if st.session_state.index is None or st.session_state.uploaded_guidelines:
        with st.spinner("Building FAISS index..."):
            st.session_state.index, st.session_state.texts, st.session_state.filenames = build_faiss_index(all_guidelines, st.session_state.model)
        st.session_state.uploaded_guidelines = {}  # Clear uploaded guidelines after building the index

    # Query Section
    st.header("Query Guidelines")
    query = st.text_input("Enter your query regarding medication or clinical information:")
    if st.button("Search"):
        if query:
            with st.spinner("Searching guidelines..."):
                found, source, content, distance = search_guidelines(query, st.session_state.model, st.session_state.index, st.session_state.texts, st.session_state.filenames, threshold=1.0)
            if found:
                st.success(f"Found a guideline match in '{source}' (distance: {distance:.2f}):")
                st.write(content)
            else:
                st.warning("No relevant guideline found. Using LLM fallback:")
                llm_response = get_llm_response(query)
                st.write(llm_response)
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()
