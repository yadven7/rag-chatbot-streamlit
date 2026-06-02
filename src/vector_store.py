import os
import pickle
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

class VectorStore:
    """
    Manages building, persisting, loading, and searching the vector index (FAISS).
    """
    def __init__(self, db_dir: str, chunks_dir: str):
        self.db_dir = os.path.abspath(db_dir)
        self.chunks_dir = os.path.abspath(chunks_dir)
        self.index = None
        self.chunks = []
        
    def build_and_save(self, embeddings: np.ndarray, chunks: list):
        """
        Builds a FAISS index from embeddings, maps it to raw chunks, and saves to disk.
        """
        self.chunks = chunks
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)

        # Save original chunks text
        chunks_file = os.path.join(self.chunks_dir, "chunks.pkl")
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)

        if faiss is not None:
            dimension = embeddings.shape[1]
            # L2 distance flat index
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            # Save index to file
            index_file = os.path.join(self.db_dir, "index.faiss")
            faiss.write_index(self.index, index_file)
            print(f"FAISS index built and saved with {len(chunks)} vectors.")
        else:
            print("FAISS package not found. Skipping saving physical index.")

    def load_index(self) -> bool:
        """
        Loads the persisted FAISS index and chunk mappings from disk.
        Returns True if successful, False otherwise.
        """
        chunks_file = os.path.join(self.chunks_dir, "chunks.pkl")
        index_file = os.path.join(self.db_dir, "index.faiss")

        # Load chunks
        if os.path.exists(chunks_file):
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            self.chunks = []
            return False

        # Load index
        if os.path.exists(index_file) and faiss is not None:
            try:
                self.index = faiss.read_index(index_file)
                return True
            except Exception as e:
                print(f"Error reading FAISS index: {e}")
                self.index = None
                return False
        else:
            # Fallback mock mode if faiss is missing or files not present
            self.index = None
            return len(self.chunks) > 0

    def is_loaded(self) -> bool:
        """
        Returns whether the vector database index is active and ready.
        """
        if faiss is not None:
            return self.index is not None and len(self.chunks) > 0
        return len(self.chunks) > 0

    def get_chunk_count(self) -> int:
        """
        Returns the number of indexed text segments.
        """
        return len(self.chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list:
        """
        Searches the FAISS index for the query embedding.
        Returns a list of dicts: [{'chunk_id': int, 'text': str, 'score': float}, ...]
        """
        if not self.is_loaded():
            print("VectorStore: Database is not loaded.")
            return []

        if faiss is None or self.index is None:
            # Mock retrieval mode
            print("VectorStore: Running search in mock mode.")
            return [{"chunk_id": i, "text": chunk, "score": 1.0 - (i * 0.1)} for i, chunk in enumerate(self.chunks[:top_k])]

        # Perform index lookup
        # distances represents L2 distances, indices contains index of matching vectors
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append({
                    "chunk_id": int(idx),
                    "text": self.chunks[idx],
                    # Lower L2 distance means higher similarity. We return the raw score.
                    "score": float(distances[0][i])
                })
        return results

if __name__ == "__main__":
    v_store = VectorStore("./vectordb", "./chunks")
    print(f"VectorStore initialized. Loaded status: {v_store.is_loaded()}")
