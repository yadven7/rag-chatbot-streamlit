import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class Embedder:
    """
    Handles generating dense vector embeddings from text chunks
    using pre-trained models from Sentence-Transformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Supports "all-MiniLM-L6-v2" or "BAAI/bge-small-en-v1.5"
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the SentenceTransformer model from HuggingFace.
        """
        if SentenceTransformer is not None:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        else:
            print("SentenceTransformer package not found. Model will run in mock mode.")

    def get_embedding_dimension(self) -> int:
        """
        Returns the embedding dimension of the loaded model.
        """
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        # Fallback dimensions
        if "all-MiniLM" in self.model_name:
            return 384
        if "bge-small" in self.model_name:
            return 384
        return 768

    def embed_chunks(self, chunks: list) -> np.ndarray:
        """
        Generates embeddings for a list of text chunks.
        Returns a float32 numpy array.
        """
        if not chunks:
            return np.empty((0, self.get_embedding_dimension()), dtype='float32')

        if self.model is not None:
            embeddings = self.model.encode(chunks, show_progress_bar=False)
            return np.array(embeddings).astype('float32')
        else:
            # Mock embeddings for testing without packages
            print("Embedder: Mocking embedding arrays.")
            dimension = self.get_embedding_dimension()
            return np.random.rand(len(chunks), dimension).astype('float32')

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates embedding for a single user search query.
        Returns a float32 numpy array of shape (1, dimension).
        """
        if self.model is not None:
            embedding = self.model.encode([query], show_progress_bar=False)
            return np.array(embedding).astype('float32')
        else:
            # Mock query embedding
            dimension = self.get_embedding_dimension()
            return np.random.rand(1, dimension).astype('float32')

if __name__ == "__main__":
    embedder = Embedder()
    test_chunks = ["Hello world", "RAG pipelines are fun to build."]
    vectors = embedder.embed_chunks(test_chunks)
    print(f"Embedding shape: {vectors.shape}")
