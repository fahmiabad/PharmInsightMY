import streamlit as st
import os
import io
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import openai  # For LLM fallback

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
    # Retrieve the API key from secrets (ensure your secrets.toml is properly set up)
    openai.api_key = st.secrets["openai"]["api_key"]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert pharmacy assistant providing evidence-based information."},
                {"role": "user", "content": query}
            ]
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"Error contacting LLM: {e}"

# --- Streamlit App ---

def main():
    # App Title and Disclaimer
    st.title("PharmInsightMY")
    st.markdown("""
    **Heads Up, Fam:**  
    This app is strictly for research purposes only.  
    Do **NOT** rely on the info provided for making clinical decisions.  
    For more deets or inquiries, hit up: [fahmibinabad@gmail.com](mailto:fahmibinabad@gmail.com).
    """)
    
    # Session State for Email Storage
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    user_email = st.text_input("Enter your email (for future updates):", value=st.session_state.user_email)
    if user_email:
        st.session_state.user_email = user_email
    st.write("Your email is:", st.session_state.user_email)
    
    # Initialize the embedding model
    st.info("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load guidelines from the local 'pdf_guidelines' folder
    local_guidelines = load_local_guidelines()
    
    # Option for the user to upload their own PDF guidelines
    st.header("Upload Your Own PDF Guidelines (Optional)")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    uploaded_guidelines = {}
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = io.BytesIO(uploaded_file.read())
            text = extract_text_from_pdf(file_bytes)
            uploaded_guidelines[uploaded_file.name] = text
    
    # Merge local and uploaded guidelines
    all_guidelines = {**local_guidelines, **uploaded_guidelines}
    if not all_guidelines:
        st.error("No guidelines available. Please add PDF guidelines locally or upload via the app.")
        return
    
    # Build the FAISS index from the combined guidelines
    index, texts, filenames = build_faiss_index(all_guidelines, model)
    
    # Query Section
    st.header("Query Guidelines")
    query = st.text_input("Enter your query regarding medication or clinical information:")
    if st.button("Search"):
        if query:
            found, source, content, distance = search_guidelines(query, model, index, texts, filenames, threshold=1.0)
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
