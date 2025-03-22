import os
import fitz  # PyMuPDF
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_with_metadata(folder_path="docs/", chunk_size=500, overlap=100):
    """
    Load and chunk PDFs and DOCX files with metadata for each chunk.

    Returns:
        List of dicts: [{ 'text': str, 'source': str, 'page': int or None }]
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        # ---- Handle PDFs ----
        if filename.endswith(".pdf"):
            try:
                doc = fitz.open(filepath)
                for page_num, page in enumerate(doc, start=1):
                    raw_text = page.get_text().strip()
                    if raw_text:
                        chunks = splitter.split_text(raw_text)
                        for chunk in chunks:
                            all_chunks.append({
                                "text": chunk,
                                "source": filename,
                                "page": page_num
                            })
                doc.close()
            except Exception as e:
                print(f"Failed to process PDF: {filename} | Error: {e}")

        # ---- Handle DOCX ----
        elif filename.endswith(".docx"):
            try:
                doc = docx.Document(filepath)
                full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                chunks = splitter.split_text(full_text)
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "source": filename,
                        "page": None
                    })
            except Exception as e:
                print(f"Failed to process DOCX: {filename} | Error: {e}")

    return all_chunks
