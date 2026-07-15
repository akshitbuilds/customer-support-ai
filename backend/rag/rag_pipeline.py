# RAG Pipeline for TechMart Customer Support
# This module loads PDF documents, splits them into chunks,
# generates embeddings using sentence-transformers,
# and stores them in FAISS vector database for semantic retrieval

import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Paths
KNOWLEDGE_BASE_PATH = "knowledge_base/"
FAISS_INDEX_PATH = "backend/rag/faiss_index.pkl"

# Load embedding model
# all-MiniLM-L6-v2 converts text to 384-dimensional vectors
# It is lightweight, fast, and accurate for semantic search
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pdfs(folder_path):
    """
    Load all PDF files from knowledge base folder.
    Returns list of (filename, text) tuples.
    """
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append((filename, text))
            print(f"Loaded: {filename}")
    return documents

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split text into overlapping chunks.
    
    Why chunking? LLMs have token limits. We cannot pass entire documents.
    Smaller chunks also improve retrieval precision.
    
    Why overlap? So context is not lost at chunk boundaries.
    Example: If a sentence is split across two chunks,
    overlap ensures both chunks contain it.
    
    chunk_size=300 words is optimal for this knowledge base size.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_vector_store(documents):
    """
    Convert document chunks to embeddings and store in FAISS.
    
    Steps:
    1. Chunk each document
    2. Generate embedding for each chunk (384-dimensional vector)
    3. Store all embeddings in FAISS index
    4. Save chunk texts separately for retrieval
    """
    all_chunks = []
    chunk_metadata = []  # stores (filename, chunk_index) for each chunk

    for filename, text in documents:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({"source": filename, "chunk_id": i})

    print(f"Total chunks created: {len(all_chunks)}")

    # Generate embeddings for all chunks
    print("Generating embeddings...")
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index
    # IndexFlatL2 = exact search using L2 (Euclidean) distance
    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")

    # Save index and chunks together
    vector_store = {
        "index": index,
        "chunks": all_chunks,
        "metadata": chunk_metadata
    }

    with open(FAISS_INDEX_PATH, 'wb') as f:
        pickle.dump(vector_store, f)

    print(f"Vector store saved to {FAISS_INDEX_PATH}")
    return vector_store

def load_vector_store():
    """Load existing FAISS index from disk."""
    with open(FAISS_INDEX_PATH, 'rb') as f:
        return pickle.load(f)

def retrieve(query, vector_store, top_k=3):
    """
    Retrieve top_k most relevant chunks for a given query.
    
    Steps:
    1. Convert query to embedding (same model as documents)
    2. Search FAISS for nearest neighbours
    3. Return top_k chunks with their source documents
    
    top_k=3 means we pass 3 most relevant chunks to the LLM.
    More chunks = more context but also more tokens used.
    """
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = vector_store["index"].search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "chunk": vector_store["chunks"][idx],
                "source": vector_store["metadata"][idx]["source"],
                "distance": float(distances[0][i])
            })
    return results

# Run this to build the vector store
if __name__ == "__main__":
    print("Building TechMart knowledge base...")
    docs = load_pdfs(KNOWLEDGE_BASE_PATH)
    store = build_vector_store(docs)
    print("RAG pipeline ready!")

    # Test retrieval
    test_query = "How do I get a refund?"
    results = retrieve(test_query, store)
    print(f"\nTest query: '{test_query}'")
    print(f"Top result from: {results[0]['source']}")
    print(f"Content preview: {results[0]['chunk'][:200]}")