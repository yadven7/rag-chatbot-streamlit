import os
from src.embedder import Embedder
from src.vector_store import VectorStore

VECTORDB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vectordb"))
CHUNKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chunks"))

class Retriever:
    """
    Combines the Embedder and VectorStore to retrieve relevant
    source document passages based on raw natural language queries.
    """
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = Embedder(model_name=embedding_model_name)
        self.vector_store = VectorStore(db_dir=VECTORDB_DIR, chunks_dir=CHUNKS_DIR)
        self.vector_store.load_index()

    def is_index_loaded(self) -> bool:
        """
        Returns whether the vector database index is active and ready.
        """
        return self.vector_store.is_loaded()

    def get_chunk_count(self) -> int:
        """
        Returns the number of indexed text segments.
        """
        return self.vector_store.get_chunk_count()

    def retrieve(self, query: str, top_k: int = 8) -> list:
        """
        Retrieves matching document chunks using hybrid search (semantic + keyword).
        """
        if not self.is_index_loaded():
            # Try reloading in case ingestion just finished
            self.vector_store.load_index()
            
        if not self.is_index_loaded():
            print("Retriever: No index loaded yet.")
            return []

        # 1. Determine if query has broad legal terms and adjust top_k
        import re
        q_lower = query.lower()
        broad_legal_terms = ["obligations", "duties", "restrictions", "responsibilities", "obligation", "duty", "restriction", "responsibility"]
        has_broad_term = any(term in q_lower for term in broad_legal_terms)
        
        actual_top_k = top_k
        if has_broad_term:
            actual_top_k = max(actual_top_k, 15)  # retrieve more chunks for broad terms

        # 2. Extract and expand keywords
        words = re.findall(r'\b\w+\b', q_lower)
        stopwords = {
            "the", "what", "are", "and", "for", "you", "your", "our", "this", "that", 
            "with", "from", "does", "say", "says", "about", "happen", "happens", 
            "after", "before", "here", "there", "they", "them", "their", "will", 
            "shall", "should", "would", "can", "could", "may", "has", "have", "had", 
            "been", "was", "were", "who", "whom", "whose", "which", "why", "how", "documents", "document"
        }
        base_keywords = list(set([w for w in words if len(w) > 2 and w not in stopwords]))
        
        # Suffix-based singular stemming
        keywords = list(base_keywords)
        for kw in base_keywords:
            if kw.endswith("ies") and len(kw) > 4:
                singular = kw[:-3] + "y"
                if singular not in keywords:
                    keywords.append(singular)
            elif kw.endswith("s") and not kw.endswith("ss") and len(kw) > 3:
                singular = kw[:-1]
                if singular not in keywords:
                    keywords.append(singular)
        
        # Add legal keyword expansion
        # "user obligations" should also search: you agree, you must, you may not, prohibited, comply, account, payment, use of services
        is_user_obligations = "user obligations" in q_lower or ("user" in q_lower and "obligation" in q_lower)
        if is_user_obligations:
            expansion = ["you agree", "you must", "you may not", "prohibited", "comply", "account", "payment", "use of services"]
            for term in expansion:
                if term not in keywords:
                    keywords.append(term)

        # 3. Retrieve semantic candidates
        # Fetch more semantic candidates than actual_top_k to allow hybrid merging to work
        semantic_search_k = max(30, actual_top_k * 2)
        query_vector = self.embedder.embed_query(query)
        semantic_results = self.vector_store.search(query_vector, top_k=semantic_search_k)

        # Build map of semantic results
        semantic_map = {}
        for res in semantic_results:
            cid = res.get("chunk_id")
            if cid is not None:
                semantic_map[cid] = res

        # 4. Perform keyword search over all database chunks
        all_chunks = self.vector_store.chunks
        candidate_pool = {}

        for idx, chunk_text in enumerate(all_chunks):
            chunk_lower = chunk_text.lower()
            matches = []
            kw_score = 0.0
            for kw in keywords:
                if " " in kw:
                    # Phrase match
                    count = chunk_lower.count(kw)
                    if count > 0:
                        matches.append(kw)
                        kw_score += count * 2.0
                else:
                    # Word boundary match
                    pattern = rf'\b{re.escape(kw)}\b'
                    finds = re.findall(pattern, chunk_lower)
                    if finds:
                        matches.append(kw)
                        kw_score += len(finds) * 1.0

            # Calculate semantic similarity score
            # L2 distance is returned as 'score' in FAISS search.
            # Convert L2 distance to a similarity score (higher is better, range (0, 1])
            is_semantic_match = idx in semantic_map
            if is_semantic_match:
                l2_dist = semantic_map[idx]["score"]
                sem_score = 1.0 / (1.0 + l2_dist)
            else:
                l2_dist = 999.0
                sem_score = 0.0

            # Combined score: semantic_score + keyword_score
            # We scale keyword_score slightly so it acts as a strong boost/signal but respects semantic ranking
            combined_score = sem_score + 0.1 * kw_score

            # Only consider it if it's a semantic candidate or it has keyword matches
            if is_semantic_match or kw_score > 0:
                candidate_pool[idx] = {
                    "chunk_id": idx,
                    "text": chunk_text,
                    "score": l2_dist,  # Keep the L2 distance for backward compatibility
                    "semantic_score": sem_score,
                    "keyword_score": kw_score,
                    "keyword_matches": matches,
                    "combined_score": combined_score
                }

        # 5. Sort final chunks by combined score descending and deduplicate (dictionary naturally deduplicates by chunk_id)
        sorted_candidates = sorted(candidate_pool.values(), key=lambda x: x["combined_score"], reverse=True)

        # 6. Return top actual_top_k results
        return sorted_candidates[:actual_top_k]

if __name__ == "__main__":
    retriever = Retriever()
    if retriever.is_index_loaded():
        print(f"Retriever loaded with {retriever.get_chunk_count()} chunks.")
        results = retriever.retrieve("eBay policies")
        print(f"Query results: {len(results)}")
    else:
        print("Retriever: No active index available. Run ingestion first.")
