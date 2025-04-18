import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import pickle
import json
import datetime
import uuid
from openai import OpenAI
from PyPDF2 import PdfReader

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
K_RETRIEVE = 5

# ============================================
# 2. TEXT CHUNKING FUNCTIONS
# ============================================
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks of approximately chunk_size characters.
    
    Args:
        text (str): The document text to chunk
        chunk_size (int): Target size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    if not text or len(text) < chunk_size:
        return [text]
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # Find the end of the chunk
        end = start + chunk_size
        
        # If we're at the end of the text, just take what's left
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        # Try to find a good breaking point (newline or period followed by space)
        # Look backward from the end to find a natural breaking point
        last_period = text.rfind('. ', start + chunk_size // 2, end)
        last_newline = text.rfind('\n', start + chunk_size // 2, end)
        
        break_point = max(last_period + 1, last_newline + 1)
        
        # If no good breaking point found, break at the chunk size
        if break_point <= start:
            break_point = end
            
        # Add the chunk
        chunks.append(text[start:break_point])
        
        # Calculate the next starting point with overlap
        start = break_point - overlap
        
    return chunks

# ============================================
# 3. EMBEDDING FUNCTIONS
# ============================================
def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get embedding vector for text using OpenAI API with proper error handling.
    
    Args:
        text (str): Text to get embedding for
        model (str): Embedding model to use
        
    Returns:
        numpy.ndarray: Embedding vector
    """
    try:
        # Clean and prepare text
        text = text.replace("\n", " ").strip()
        
        # Handle empty text
        if not text:
            # Default dimensions for different models
            dimensions = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            dim = dimensions.get(model, 1536)
            return np.zeros(dim, dtype=np.float32)
            
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
        
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        # Try fallback to older model if the specified one fails
        if model != "text-embedding-ada-002":
            st.warning(f"Falling back to text-embedding-ada-002 model")
            return get_embedding(text, "text-embedding-ada-002")
        raise

# ============================================
# 4. LOADING PRE‑LOADED DOCUMENTS
# ============================================
@st.cache_resource
def load_preloaded_index():
    """
    Load pre-loaded document index if available.
    
    Returns:
        tuple: (faiss_index, document_metadata)
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_METADATA_PATH):
        # If pre-loaded files don't exist, create empty structures
        dimension = 1536  # Default embedding dimension
        index = faiss.IndexFlatIP(dimension)  # Use inner product for cosine similarity
        metadata = []
        return index, metadata
        
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        st.error(f"Error loading pre-loaded index: {e}")
        # Create empty structures as fallback
        dimension = 1536  # Default embedding dimension
        index = faiss.IndexFlatIP(dimension)  # Use inner product for cosine similarity
        metadata = []
        return index, metadata

# ============================================
# 5. CREATE PRE-LOADED DOCUMENTS (Admin use only)
# ============================================
def create_preloaded_index(file_paths, chunk_size=1000, overlap=200, embedding_model="text-embedding-ada-002"):
    """
    Create and save a pre-loaded index from a list of file paths.
    For admin use to prepare pre-loaded documents.
    
    Args:
        file_paths (list): List of file paths to process
        chunk_size (int): Size of text chunks
        overlap (int): Overlap between chunks
        embedding_model (str): Embedding model to use
        
    Returns:
        bool: Success status
    """
    all_docs = []
    all_embeddings = []
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        
        # Read file based on extension
        if filename.lower().endswith(".pdf"):
            try:
                pdf_reader = PdfReader(file_path)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            except Exception as e:
                print(f"Error reading PDF file {filename}: {e}")
                continue
        else:
            # Assuming a text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size, overlap)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            chunk_id = f"{filename}_{i+1}"
            doc = {
                "text": chunk,
                "source": filename,
                "chunk_id": chunk_id,
                "is_preloaded": True
            }
            all_docs.append(doc)
            
            try:
                emb = get_embedding(chunk, model=embedding_model)
                all_embeddings.append(emb)
            except Exception as e:
                print(f"Error creating embedding for chunk {i+1} of {filename}: {e}")
                continue
    
    # Create FAISS index
    if all_embeddings:
        all_embeddings = np.array(all_embeddings)
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(all_embeddings)
        index.add(all_embeddings)
        
        # Save the index and metadata
        try:
            faiss.write_index(index, INDEX_PATH)
            with open(DOCS_METADATA_PATH, "wb") as f:
                pickle.dump(all_docs, f)
            print(f"Successfully created pre-loaded index with {len(all_docs)} chunks")
            return True
        except Exception as e:
            print(f"Error saving pre-loaded index: {e}")
            return False
    else:
        print("No embeddings were created. Check your files and API key.")
        return False

# ============================================
# 6. PROCESS USER‑UPLOADED DOCUMENTS
# ============================================
@st.cache_data
def process_uploaded_documents(uploaded_files, chunk_size=1000, overlap=200, embedding_model="text-embedding-ada-002"):
    """
    Process user-uploaded documents and create embeddings.
    
    Args:
        uploaded_files: List of uploaded file objects
        chunk_size: Size of chunks in characters
        overlap: Overlap between chunks
        embedding_model: Embedding model to use
        
    Returns:
        tuple: (faiss_index, document_chunks)
    """
    user_docs = []
    embeddings = []
    
    for file in uploaded_files:
        file_name = file.name.lower()
        
        # Extract text based on file type
        if file_name.endswith(".pdf"):
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
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size, overlap)
        
        # Create document entries and get embeddings for each chunk
        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            chunk_id = f"{file.name}_{i+1}"
            doc = {
                "text": chunk,
                "source": file.name,
                "chunk_id": chunk_id,
                "is_preloaded": False
            }
            user_docs.append(doc)
            
            try:
                emb = get_embedding(chunk, model=embedding_model)
                embeddings.append(emb)
            except Exception as e:
                st.error(f"Error creating embedding for chunk {i+1} of {file.name}: {e}")
                continue
    
    # Create FAISS index if we have embeddings
    if embeddings:
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]
        # Using inner product for cosine similarity (with normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
    else:
        index = None
        
    return index, user_docs

# ============================================
# 7. SEARCH FUNCTION
# ============================================
def search_documents(query, pre_index, pre_docs, user_index, user_docs, k=5):
    """
    Search both pre-loaded and user-uploaded document indices for relevant chunks.
    
    Args:
        query: The user query
        pre_index: FAISS index for pre-loaded documents
        pre_docs: List of pre-loaded documents
        user_index: FAISS index for user-uploaded documents
        user_docs: List of user-uploaded documents
        k: Number of results to retrieve from each index
        
    Returns:
        list: Sorted list of most relevant document chunks
    """
    query_vector = get_embedding(query).reshape(1, -1)
    # Normalize query vector for cosine similarity
    faiss.normalize_L2(query_vector)
    results = []

    # Search in pre-loaded documents (if available)
    if pre_index is not None and len(pre_docs) > 0:
        # Use min to avoid retrieving more results than we have documents
        k_pre = min(k, len(pre_docs))
        distances, indices = pre_index.search(query_vector, k_pre)
        
        for i, score in zip(indices[0], distances[0]):
            if i < len(pre_docs):  # Safety check
                doc = pre_docs[i].copy()
                doc["score"] = float(score)  # Convert to float for JSON serialization
                results.append(doc)
                
    # Search in user-uploaded documents (if available)
    if user_index is not None and len(user_docs) > 0:
        # Use min to avoid retrieving more results than we have documents
        k_user = min(k, len(user_docs))
        distances, indices = user_index.search(query_vector, k_user)
        
        for i, score in zip(indices[0], distances[0]):
            if i < len(user_docs):  # Safety check
                doc = user_docs[i].copy()
                doc["score"] = float(score)  # Convert to float for JSON serialization
                results.append(doc)
    
    # Sort by score (higher is better for cosine similarity)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Take top k results overall
    return results[:k]

# ============================================
# 8. ANSWER GENERATION
# ============================================
def answer_query(query, pre_index, pre_docs, user_index, user_docs, 
                include_explanation=True, max_context_length=4000):
    """
    Generate an answer to the user's query using retrieved document chunks.
    
    Args:
        query: User query
        pre_index: FAISS index for pre-loaded documents
        pre_docs: List of pre-loaded documents
        user_index: FAISS index for user-uploaded documents
        user_docs: List of user-uploaded documents
        include_explanation: Whether to include explanation in the prompt
        max_context_length: Maximum length of context to send to the LLM
        
    Returns:
        dict: Answer, source information, and retrieved results
    """
    # Get the most relevant document chunks
    results = search_documents(query, pre_index, pre_docs, user_index, user_docs, k=5)
    
    if results:
        # Prepare context from retrieved chunks, respecting max context length
        context_pieces = []
        current_length = 0
        
        for result in results:
            # Add source information to each chunk
            chunk = f"Source: {result['source']}\n\n{result['text']}"
            chunk_length = len(chunk)
            
            # Check if adding this chunk would exceed max context length
            if current_length + chunk_length > max_context_length:
                # If this is the first chunk, add it even if it exceeds length (truncated)
                if not context_pieces:
                    context_pieces.append(chunk[:max_context_length])
                break
            
            context_pieces.append(chunk)
            current_length += chunk_length
            
        context = "\n\n---\n\n".join(context_pieces)
        
        # Build prompt based on whether explanation is requested
        if include_explanation:
            prompt = f"""
You are a clinical expert assistant. Answer the following question based ONLY on the provided document excerpts. 
If the document excerpts don't contain the information needed to answer the question, 
say "I don't have enough information about this in the provided documents."

**Document Excerpts:**
{context}

**Question:**
{query}

Provide your answer in this format:

**Answer:**
[A direct and concise answer to the question]

**Explanation:**
[A detailed explanation with specific information from the documents]

**Sources:**
[List the sources of information used]
"""
        else:
            prompt = f"""
You are a clinical expert assistant. Answer the following question based ONLY on the provided document excerpts.
If the document excerpts don't contain the information needed to answer the question, 
say "I don't have enough information about this in the provided documents."

**Document Excerpts:**
{context}

**Question:**
{query}

Provide your answer in this format:

**Answer:**
[A direct and concise answer to the question]
"""
        source = "Retrieved Documents"
    else:
        # No relevant documents found
        prompt = f"""
You are a clinical expert assistant. No relevant document was found for the following question:

Question: {query}

Please respond with:
"I don't have specific information about this in the provided documents. Please consider consulting official clinical guidelines or pharmacist resources for accurate information."
"""
        source = "No Relevant Documents"

    # Generate the answer using GPT
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You could also use gpt-4 for better accuracy if available
            messages=[
                {"role": "system", "content": "You are an expert clinical assistant for pharmacists."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        answer = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        answer = f"Error generating answer: {str(e)}"
    
    return {
        "answer": answer, 
        "source": source, 
        "results": results,
        "query": query
    }

# ============================================
# 9. FEEDBACK SYSTEM
# ============================================
class FeedbackManager:
    """
    Manages user feedback for question-answer pairs.
    """
    
    def __init__(self, feedback_dir="feedback_data"):
        """Initialize the feedback manager with storage directory"""
        self.feedback_dir = feedback_dir
        self.feedback_file = os.path.join(feedback_dir, "feedback_records.json")
        self.stats_file = os.path.join(feedback_dir, "feedback_stats.json")
        
        # Create directory if it doesn't exist
        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)
            
        # Initialize files if they don't exist
        if not os.path.exists(self.feedback_file):
            self._write_json(self.feedback_file, [])
            
        if not os.path.exists(self.stats_file):
            self._write_json(self.stats_file, {
                "total_questions": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "feedback_rate": 0,
                "last_updated": datetime.datetime.now().isoformat()
            })
    
    def _read_json(self, file_path):
        """Read JSON data from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            if file_path == self.feedback_file:
                return []
            else:
                return {}
    
    def _write_json(self, file_path, data):
        """Write JSON data to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False
    
    def record_question(self, question_id, query, answer):
        """Record a new question (without feedback yet)"""
        # Create new record
        timestamp = datetime.datetime.now().isoformat()
        
        new_record = {
            "question_id": question_id,
            "timestamp": timestamp,
            "query": query,
            "answer": answer,
            "feedback": None,
            "feedback_comments": None,
            "feedback_timestamp": None
        }
        
        # Load existing records
        records = self._read_json(self.feedback_file)
        records.append(new_record)
        
        # Update records file
        self._write_json(self.feedback_file, records)
        
        # Update stats
        stats = self._read_json(self.stats_file)
        stats["total_questions"] += 1
        stats["last_updated"] = timestamp
        self._write_json(self.stats_file, stats)
        
        return question_id
    
    def add_feedback(self, question_id, is_helpful, comments=None):
        """Add feedback for a specific question"""
        # Load records
        records = self._read_json(self.feedback_file)
        
        # Find the specific record
        record_found = False
        for record in records:
            if record["question_id"] == question_id:
                record["feedback"] = is_helpful
                record["feedback_comments"] = comments
                record["feedback_timestamp"] = datetime.datetime.now().isoformat()
                record_found = True
                break
        
        if not record_found:
            return False
            
        # Save updated records
        self._write_json(self.feedback_file, records)
        
        # Update statistics
        stats = self._read_json(self.stats_file)
        if is_helpful:
            stats["positive_feedback"] += 1
        else:
            stats["negative_feedback"] += 1
            
        total_feedback = stats["positive_feedback"] + stats["negative_feedback"]
        if total_feedback > 0:
            stats["feedback_rate"] = total_feedback / stats["total_questions"]
                
        stats["last_updated"] = datetime.datetime.now().isoformat()
        self._write_json(self.stats_file, stats)
        
        return True

# Function to generate a unique question ID
def generate_question_id():
    """Generate a unique ID for a question"""
    return str(uuid.uuid4())

# Add feedback UI components
def add_feedback_ui(question_id):
    """Add feedback collection UI components"""
    st.divider()
    st.markdown("### Was this answer helpful?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("👍 Yes", key=f"yes_{question_id}"):
            feedback_manager = FeedbackManager()
            feedback_manager.add_feedback(question_id, True)
            st.success("Thank you for your feedback!")
            
    with col2:
        if st.button("👎 No", key=f"no_{question_id}"):
            st.session_state["show_feedback_form"] = True
            st.session_state["current_question_id"] = question_id
            
    # Show detailed feedback form if needed
    if st.session_state.get("show_feedback_form", False) and st.session_state.get("current_question_id") == question_id:
        with st.form(key=f"feedback_form_{question_id}"):
            st.markdown("### What could be improved?")
            comments = st.text_area("Your feedback")
            
            if st.form_submit_button("Submit Feedback"):
                feedback_manager = FeedbackManager()
                feedback_manager.add_feedback(question_id, False, comments)
                
                st.success("Thank you for your detailed feedback!")
                st.session_state["show_feedback_form"] = False

# ============================================
# 10. USER INTERFACE (UI) WITH STREAMLIT
# ============================================
def main():
    st.set_page_config("PharmInsight - Document Search", layout="wide")

    # Add CSS for better UI
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .source-box {
            padding: 10px;
            border: 1px solid #f0f0f0;
            border-radius: 5px;
            margin-bottom: 10px;
            background-color: #fafafa;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("PharmInsight for Pharmacists")
    st.markdown("Ask questions related to clinical guidelines, drug information, or policies.")

    # Create a sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 500, 200, 50)
            k_retrieval = st.slider("Number of chunks to retrieve", 3, 10, 5)
            
            llm_model = st.selectbox(
                "LLM Model",
                ["gpt-3.5-turbo", "gpt-4"],
                index=0
            )
            
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        
        # Document management
        st.header("Document Management")
        include_preloaded = st.checkbox("Include pre-loaded documents", value=True)

    # Document upload section
    st.markdown("### Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Upload documents (text or PDF files)", 
        accept_multiple_files=True, 
        type=["txt", "pdf"]
    )

    # Document processing indicator
    doc_count = 0
    chunk_count = 0

    # Process user documents
    user_index, user_docs = (None, [])
    if uploaded_files:
        with st.status("Processing uploaded documents...", expanded=True) as status:
            user_index, user_docs = process_uploaded_documents(
                uploaded_files, 
                chunk_size=chunk_size, 
                overlap=chunk_overlap
            )
            doc_count = len(uploaded_files)
            chunk_count = len(user_docs)
            status.update(label=f"✅ Processed {doc_count} documents into {chunk_count} chunks", state="complete")

    # Load pre-loaded documents if requested
    pre_index, pre_docs = (None, [])
    if include_preloaded:
        with st.status("Loading pre-indexed documents...", expanded=False) as status:
            pre_index, pre_docs = load_preloaded_index()
            if pre_index is not None and len(pre_docs) > 0:
                status.update(label=f"✅ Loaded {len(pre_docs)} pre-indexed document chunks", state="complete")
            else:
                status.update(label="ℹ️ No pre-indexed documents found", state="complete")

    # Query section
    st.markdown("### Ask Your Question")
    query = st.text_input("Type your question here", placeholder="e.g., What is the recommended monitoring plan for amiodarone?")

    # Answer options
    col1, col2 = st.columns(2)
    with col1:
        include_explanation = st.checkbox("Include detailed explanation", value=True)
    with col2:
        show_sources = st.checkbox("Show source documents", value=True)

    # Check if we have documents available
    has_documents = (user_docs and len(user_docs) > 0) or (include_preloaded and pre_docs and len(pre_docs) > 0)

    # Submit button
    if st.button("Submit", type="primary", disabled=(not query or not has_documents)):
        if not has_documents:
            st.warning("⚠️ No documents available. Please upload documents or enable pre-loaded documents.")
        elif not query.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            # Generate a unique ID for this Q&A pair
            question_id = generate_question_id()
            
            with st.spinner("Searching documents and generating answer..."):
                result = answer_query(
                    query, 
                    pre_index if include_preloaded else None, 
                    pre_docs if include_preloaded else [], 
                    user_index, 
                    user_docs,
                    include_explanation=include_explanation
                )
            
            # Display answer
            if result["source"] == "Retrieved Documents":
                st.success("Answer based on the provided documents:")
            else:
                st.warning("No matching information found in the documents.")
            
            st.markdown(result["answer"], unsafe_allow_html=True)
            
            # Record this Q&A pair in the feedback system
            feedback_manager = FeedbackManager()
            feedback_manager.record_question(
                question_id, 
                query, 
                result["answer"]
            )
            
            # Add feedback UI components
            add_feedback_ui(question_id)
            
            # Display sources if requested
            if show_sources and result["results"]:
                with st.expander("📄 Sources Used", expanded=True):
                    for idx, doc in enumerate(result["results"]):
                        score = doc.get("score", 0)
                        source_type = "Pre-loaded" if doc.get("is_preloaded", False) else "User-uploaded"
                        score_percent = int(score * 100) if score <= 1 else int(score)
                        
                        st.markdown(f"**Source {idx+1}: {doc['source']} ({source_type}, Relevance: {score_percent}%)**")
                        
                        # Display a snippet of the text
                        text_snippet = doc["text"]
                        if len(text_snippet) > 500:
                            text_snippet = text_snippet[:500] + "..."
                        
                        st.markdown(f"<div class='source-box'>{text_snippet}</div>", unsafe_allow_html=True)
            
            # Show debug info for developers
            with st.expander("🔍 Debug Information", expanded=False):
                st.write({
                    "Query": result["query"],
                    "Retrieved Documents": len(result["results"]),
                    "Source": result["source"],
                    "Chunk Size": chunk_size,
                    "Chunk Overlap": chunk_overlap,
                    "LLM Model": llm_model
                })
                
                # Show the top scores
                if result["results"]:
                    scores = [{"Source": doc["source"], "Score": doc.get("score", 0), "Type": "Pre-loaded" if doc.get("is_preloaded", False) else "User-uploaded"} for doc in result["results"]]
                    st.write("Relevance Scores:")
                    st.dataframe(scores)
                    
    # Document statistics
    if doc_count > 0 or (pre_docs and len(pre_docs) > 0):
        st.sidebar.markdown("### Document Statistics")
        if doc_count > 0:
            st.sidebar.markdown(f"**User Documents:** {doc_count} documents ({chunk_count} chunks)")
        if pre_docs and len(pre_docs) > 0:
            st.sidebar.markdown(f"**Pre-loaded Documents:** {len(pre_docs)} chunks")
            
    # Help section
    with st.sidebar.expander("Help & Tips"):
        st.markdown("""
        **Tips for better results:**
        
        - Be specific in your questions
        - Upload relevant documents
        - Try different chunk sizes for different types of documents
        - Increase retrieval count (k) for broader searches
        - Decrease temperature for more factual answers
        """)
    
    # Admin section (hidden by default)
    admin_mode = st.sidebar.checkbox("Admin Mode", value=False)
    if admin_mode:
        with st.sidebar.expander("Admin Tools", expanded=True):
            st.markdown("### Feedback Statistics")
            if st.button("View Feedback Data"):
                feedback_manager = FeedbackManager()
                stats = feedback_manager._read_json(feedback_manager.stats_file)
                
                total_q = stats.get("total_questions", 0)
                pos_f = stats.get("positive_feedback", 0)
                neg_f = stats.get("negative_feedback", 0)
                
                st.write(f"Total Questions: {total_q}")
                st.write(f"Positive Feedback: {pos_f}")
                st.write(f"Negative Feedback: {neg_f}")
                
                # Calculate satisfaction rate
                if pos_f + neg_f > 0:
                    satisfaction = pos_f / (pos_f + neg_f) * 100
                    st.write(f"Satisfaction Rate: {satisfaction:.1f}%")
            
            # Tool to create pre-loaded index (not for end users)
            st.markdown("### Create Pre-loaded Index")
            st.markdown("This tool allows you to create a pre-loaded index from files on disk.")
            st.markdown("**Warning:** This will replace any existing pre-loaded index.")
            
            file_paths_input = st.text_area("Enter file paths (one per line)")
            
            create_index_clicked = st.button("Create Pre-loaded Index")
            if create_index_clicked and file_paths_input.strip():
                file_paths = [path.strip() for path in file_paths_input.split('\n') if path.strip()]
                
                if file_paths:
                    with st.spinner("Creating pre-loaded index..."):
                        success = create_preloaded_index(
                            file_paths,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap
                        )
                    
                    if success:
                        st.success("Pre-loaded index created successfully!")
                    else:
                        st.error("Failed to create pre-loaded index.")
                else:
                    st.error("No valid file paths provided.")

# Run the main application
if __name__ == "__main__":
    main()