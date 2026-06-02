from src.retriever import Retriever
from src.generator import Generator
from typing import Generator as TypingGenerator


class RAGPipeline:
    """
    Coordinates Retriever and Generator to run the RAG workflow.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.retriever = Retriever(embedding_model_name=embedding_model)
        self.generator = Generator()

    def retrieve(self, query: str, top_k: int = 8) -> list:
        """
        Retrieves matching document chunks for the user query.
        """
        return self.retriever.retrieve(query, top_k=top_k)

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: list,
        model_name: str = "Ollama llama3.2:3b"
    ) -> TypingGenerator[str, None, None]:
        """
        Streams grounded response using selected model.
        """
        return self.generator.generate_stream(
            query=query,
            retrieved_chunks=retrieved_chunks,
            model_name=model_name
        )


if __name__ == "__main__":
    pipeline = RAGPipeline()
    print(
        f"RAG Pipeline initialized. Index loaded status: "
        f"{pipeline.retriever.is_index_loaded()}"
    )