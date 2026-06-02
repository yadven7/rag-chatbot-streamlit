import os
import json
from dotenv import load_dotenv

# Import modular classes
from src.document_loader import DocumentLoader
from src.chunker import Chunker
from src.embedder import Embedder
from src.vector_store import VectorStore

load_dotenv()

# Configurations
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
CHUNKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chunks"))
VECTORDB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vectordb"))
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def run_ingestion(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    """
    Orchestrated ingestion workflow:
    1. Extract and clean text from PDFs in /data
    2. Segment cleaned text into 100-300 word sentence-aware chunks
    3. Generate embeddings for all chunks
    4. Persist FAISS index and mappings to disk
    """
    print("Starting ingestion workflow...")
    
    # 1. Load documents
    loader = DocumentLoader(data_dir=DATA_DIR)
    combined_text = loader.load_all_documents()
    print("Documents loaded and cleaned successfully.")

    # 2. Chunk text
    chunker = Chunker(min_words=100, max_words=300)
    chunks = chunker.create_chunks(combined_text)
    print(f"Divided documents into {len(chunks)} sentence-aware chunks.")
    
    if not chunks:
        raise ValueError("No text chunks generated. Ingestion aborted.")

    # 3. Generate embeddings
    embedder = Embedder(model_name=embedding_model)
    embeddings = embedder.embed_chunks(chunks)
    print(f"Generated embeddings of shape {embeddings.shape}.")

    # 4. Save to Vector Store
    vector_store = VectorStore(db_dir=VECTORDB_DIR, chunks_dir=CHUNKS_DIR)
    vector_store.build_and_save(embeddings, chunks)
    print("FAISS index saved successfully.")

    return {
        "status": "Success",
        "chunks_count": len(chunks),
        "embedding_model": embedding_model,
        "embedding_dim": embeddings.shape[1]
    }

if __name__ == "__main__":
    try:
        stats = run_ingestion()
        print("\n=== Ingestion Completed ===")
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"\n[ERROR] Ingestion Failed: {e}")
