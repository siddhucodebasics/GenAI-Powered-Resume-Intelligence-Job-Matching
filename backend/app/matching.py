import faiss
import numpy as np

def load_faiss_index(index_path: str):
    """Load FAISS index from disk"""
    return faiss.read_index(index_path)

def search_candidates(index, query_embedding, top_k: int = 5):
    """Search top-k candidates using FAISS"""
    query = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)
    return scores[0], indices[0]