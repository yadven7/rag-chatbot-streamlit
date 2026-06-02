# Technical Report: Instruction-Tuned Hybrid RAG Chatbot

This report covers the design decisions, component implementations, prompt strategies, evaluation queries, and technical limitations of the hybrid Retrieval-Augmented Generation (RAG) chatbot system built for the Streamlit QA project.

---

## 1. Document Structure & Chunking Logic

### Text Cleaning
Before chunking, raw document pages undergo a cleaning phase to remove noise:
- Stripping HTML/XML tags.
- Collapsing multiple continuous whitespaces, newlines, and tabs into single spacing.
- Removing common header/footer fragments (if applicable) to avoid duplicate noise.

### Sentence-Aware Chunking Strategy
The text is chunked using a sentence-aware algorithm designed to prevent context clipping:
- **Length Constraint**: Chunks are constrained to stay between **100 and 300 words**.
- **Sentence-Aware Split**: The raw text is first segmented into complete sentences using punctuation terminals (`.!?`).
- **Greedy Grouping**: Sentences are aggregated into a single chunk. If adding the next sentence would cause the word count of the chunk to exceed the `max_words` threshold (300), the current chunk is finalized, and a new chunk begins.
- **Benefits**:
  - Sentences are never cut in half, preserving grammatical structure and semantics.
  - Ensures each chunk is cohesive enough to contain context but small enough to fit within LLM context window constraints.

---

## 2. Embedding Model & Vector Database Selection

### Embedding Model: `all-MiniLM-L6-v2`
- **Architecture**: A lightweight SentenceTransformer model pre-trained on a massive corpus (over 1B sentence pairs).
- **Dimension size**: 384 dimensions.
- **Rationale**:
  - Extremely fast and runs efficiently on CPU hosts without requiring high-end GPUs.
  - Keeps vector index memory footprints small.
  - Outperforms similar models of its size on similarity metrics.
- **Alternative Option**: `BAAI/bge-small-en-v1.5` is supported as a drop-down option in the sidebar configuration.

### Vector Store: `FAISS` (Facebook AI Similarity Search)
- **Rationale**:
  - FAISS handles fast L2 and inner product similarity lookups in high-dimensional spaces.
  - Completely local file serialization (`index.faiss` format), requiring no remote cloud services, which is perfect for self-contained, high-performance setups.
  - Features low latency lookups even when scaling to thousands of document pages.

### Hybrid Retrieval & Scoring
To improve retrieval precision for legal terms:
- We combine **dense semantic FAISS search** with **lexical exact keyword/phrase searching**.
- Query keywords are extracted, filtered for common stopwords, and expanded using suffix singularization (e.g. "refunds" -> "refund", "duties" -> "duty").
- For specific terms like "user obligations", the query keywords are expanded with legal phrases ("you agree", "you must", "prohibited", "comply", "account", "payment").
- Chunks are deduplicated by `chunk_id` and ranked by a combined score: `combined_score = semantic_score + 0.1 * keyword_score`.

---

## 3. Prompt Engineering & Grounding Constraints

The prompt template is designed to force the Gemini/Ollama model to behave like an extractive question-answering agent, prohibiting external knowledge injection:

### Prompt Template
```text
System Instructions:
You are a helpful, precise, and fact-focused AI Assistant. Your task is to answer the user's query based ONLY on the provided Context Sources. Strictly adhere to the following rules:
1. Rely only on the clear facts mentioned in the Context. Do not make up facts, external details or assumptions.
2. If the answer cannot be found in the provided Context, say: 'I apologize, but the provided documents do not contain the answer to your query.'
3. Cite which Context Source(s) you used to construct your answer (e.g., [Source 1], [Source 2]).
4. Keep your response factual, grounded, and concise.

Context Sources:
[Source 1]
{source_text_1}

[Source 2]
{source_text_2}

User Query: {query}

Grounded Answer:
```

### Grounding and Hallucination Mitigation
- **Strict Grounding Rule**: By instructing the assistant to rely "ONLY on the provided Context Sources," the model is restricted from using its pre-trained parametric knowledge.
- **Fallback Sentence**: The explicit instruction to state "I apologize, but the provided documents do not contain the answer..." prevents the model from attempting to fill context gaps with fabrications.
- **Source Citations**: Forcing citation formats (`[Source 1]`, etc.) makes it easy to verify references.

---

## 4. Evaluation Queries & Results

Below are evaluation cases tested using the eBay User Agreement PDF document as the knowledge base:

### Query 1: Dispute Notice Mailing Details
* **Query**: "What is the dispute process and what is the address to mail a notice of dispute to eBay?"
* **Retrieved Chunks**: Chunk 31 & Chunk 32 (containing Informal Dispute Resolution details).
* **LLM Answer**: "According to the documents, you must participate in an informal dispute resolution process by sending a written Notice of Dispute. The notice must be sent via email to DisputeNotice@eBay.com or by regular mail to: eBay Inc., Attn: Litigation Department, RE: Notice of Dispute, 583 W. eBay Way, Draper, UT 84020 [Source 1]."
* **Status**: **Pass** (Highly accurate, correctly cited, zero hallucination).

### Query 2: Grounded Out-of-Context Fallback
* **Query**: "What are the rules regarding selling fresh fruits on eBay?"
* **Retrieved Chunks**: Chunks containing general policy updates or feedback rules (none containing fruit selling details).
* **LLM Answer**: "I apologize, but the provided documents do not contain the answer to your query regarding the rules for selling fresh fruits on eBay."
* **Status**: **Pass** (Gracefully triggered the fallback message instead of hallucinating fruit selling policies).

### Query 3: Missing Shipping Label Consequences
* **Query**: "What happens if a seller fails to send the buyer a return shipping label?"
* **Retrieved Chunks**: Chunk 18 & Chunk 19 (Returns and cancellations sections).
* **LLM Answer**: "If a seller fails to send a return shipping label to the buyer, an eBay-generated shipping label will be used, and the seller will be responsible for its cost [Source 1]."
* **Status**: **Pass** (Correctly retrieved details from deep inside returns policy).

### Query 4: User Obligations (Hybrid Retriever Verification)
* **Query**: "What are the user obligations?"
* **Retrieved Chunks**: Chunk 20, Chunk 22, Chunk 21, Chunk 0, Chunk 8. (Auto-scaled retrieval to 15 chunks because of the broad legal word "obligations").
* **LLM Answer**: "Based on the provided documents, user obligations include agreeing to the rules and policies for buying and selling, complying with all applicable laws, satisfying payment obligations to the seller, keeping account credentials safe, and placing a valid payment method on file to authorize charges for fees or transactions [Source 1, Source 2, Source 3]."
* **Status**: **Pass** (Prioritized core obligation clauses over arbitration terms, which previously dominated purely semantic lookups).

### Query 5: Post-Termination Consequences
* **Query**: "What happens after termination?"
* **Retrieved Chunks**: Chunk 40, Chunk 39, Chunk 27.
* **LLM Answer**: "After termination of this agreement, certain provisions will survive, including the Agreement to Arbitrate, the general terms regarding business accounts, and your obligations to satisfy outstanding payment fees incurred before termination [Source 1, Source 2]."
* **Status**: **Pass** (Correctly retrieved survival clauses).

---

## 5. Technical Limitations & Mitigation Notes

1. **Parametric Bias / Pre-training Leakage**:
   - *Issue*: Large LLMs sometimes ignore prompting rules and answer questions based on general knowledge if the question matches common internet texts.
   - *Mitigation*: Set LLM generation `temperature=0.0` to force deterministic matching and penalize non-extractive completions.
2. **Document Image Layouts**:
   - *Issue*: Standard pdf readers (`pypdf`) do not extract text from tables or charts correctly, and completely miss scanned image text.
   - *Mitigation*: For production contracts containing image scans, integrate OCR loaders (e.g., PyMuPDF, OCRmyPDF) or layout-aware parsers (Unstructured).
3. **API Latency, Network Outages, and Gemini 429 Quotas**:
   - *Issue*: Requests to Google Gemini API might time out or fail with HTTP 429 (ResourceExhausted quota exceeded) error codes.
   - *Mitigation*: The chatbot streams responses directly to Streamlit. If Gemini fails with a 429 error, the generator catches the exception, yields a visual fallback warning, and automatically routes completion generation to a local Ollama instance (`llama3.2:3b`) without crashing the application.
