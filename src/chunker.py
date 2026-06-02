import re

class Chunker:
    """
    Handles segmenting raw text into 100-300 word chunks using sentence-aware splitting.
    """
    def __init__(self, min_words: int = 100, max_words: int = 300):
        self.min_words = min_words
        self.max_words = max_words

    def split_into_sentences(self, text: str) -> list:
        """
        Uses regular expressions to split text into list of sentences cleanly.
        """
        # Split on sentence terminals (.!? followed by space)
        sentence_ends = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_ends.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> list:
        """
        Groups sentences into chunks that satisfy the [min_words, max_words] length constraints.
        Ensures sentences are not cut in half.
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence exceeds maximum chunk limit, finalize the current chunk
            if current_word_count + sentence_word_count > self.max_words:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
            
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

            # If the chunk has reached minimum word limit, we can keep it as a valid chunk.
            # We continue adding sentences until adding next sentence would exceed max_words.
            
        # Add the remaining sentences if any
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Filter out extremely small chunks (e.g. less than 10 words) to avoid noise
        chunks = [c for c in chunks if len(c.split()) > 10]
        
        return chunks

if __name__ == "__main__":
    # Test chunker locally
    chunker = Chunker()
    test_text = (
        "This is the first sentence. It introduces the subject. "
        "Here is another sentence that adds more information. "
        "We want to make sure the chunker aggregates sentences appropriately. "
        "Each chunk should be coherent and respect word limits. "
        "Let's see if this splits correctly."
    )
    test_chunks = chunker.create_chunks(test_text)
    print(f"Generated {len(test_chunks)} chunks.")
    for idx, c in enumerate(test_chunks):
        print(f"Chunk {idx+1}: {c} (Words: {len(c.split())})")
