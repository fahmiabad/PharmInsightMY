import argparse
import os
import faiss
import numpy as np
import pickle
from openai import OpenAI
from PyPDF2 import PdfReader

# ============================================
# Helper functions (same as in main app)
# ============================================
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    if not text or len(text) < chunk_size:
        return [text]
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        last_period = text.rfind('. ', start + chunk_size // 2, end)
        last_newline = text.rfind('\n', start + chunk_size // 2, end)
        break_point = max(last_period + 1, last_newline + 1)
        
        if break_point <= start:
            break_point = end
            
        chunks.append(text[start:break_point])
        start = break_point - overlap
        
    return chunks

def get_embedding(text, client, model="text-embedding-ada-002"):
    """Get embedding vector for text using OpenAI API"""
    try:
        text = text.replace("\n", " ").strip()
        if not text:
            return np.zeros(1536, dtype=np.float32)
            
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
        print(f"Error creating embedding: {e}")
        raise

# ============================================
# Main index creation function
# ============================================
def create_index(directory_path, output_path=".", chunk_size=1000, overlap=200, embedding_model="text-embedding-ada-002"):
    """
    Create a pre-loaded index from all files in a directory.
    
    Args:
        directory_path: Path to directory containing files
        output_path: Directory to save the index files
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        embedding_model: Embedding model to use
    
    Returns:
        bool: Success status
    """
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return False
    
    client = OpenAI(api_key=api_key)
    
    # Get all files in the directory
    all_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Only include .txt and .pdf files
            if file.lower().endswith((".txt", ".pdf")):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    if not all_files:
        print(f"No .txt or .pdf files found in {directory_path}")
        return False
    
    print(f"Found {len(all_files)} files to process")
    
    # Process all files
    all_docs = []
    all_embeddings = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
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
        print(f"Created {len(chunks)} chunks from {filename}")
        
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
                emb = get_embedding(chunk, client, model=embedding_model)
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
        
        # Create output paths
        index_path = os.path.join(output_path, "vector.index")
        metadata_path = os.path.join(output_path, "docs_metadata.pkl")
        
        # Save the index and metadata
        try:
            faiss.write_index(index, index_path)
            with open(metadata_path, "wb") as f:
                pickle.dump(all_docs, f)
            print(f"Successfully created pre-loaded index with {len(all_docs)} chunks")
            print(f"Files saved to: {index_path} and {metadata_path}")
            return True
        except Exception as e:
            print(f"Error saving pre-loaded index: {e}")
            return False
    else:
        print("No embeddings were created. Check your files and API key.")
        return False

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Create pre-loaded document index for PharmInsight")
    parser.add_argument("directory", help="Directory containing documents to index (.txt and .pdf files)")
    parser.add_argument("--output", "-o", default=".", help="Directory to save the index files")
    parser.add_argument("--chunk-size", "-c", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--overlap", "-v", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--model", "-m", default="text-embedding-ada-002", help="OpenAI embedding model to use")
    
    args = parser.parse_args()
    
    # Create the index
    success = create_index(
        args.directory,
        args.output,
        args.chunk_size,
        args.overlap,
        args.model
    )
    
    if success:
        print("Index creation completed successfully!")
    else:
        print("Index creation failed.")