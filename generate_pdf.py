import os
from fpdf import FPDF

class RAGTechnicalReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(100, 110, 120)
            self.cell(0, 10, "Technical Report: Instruction-Tuned Hybrid RAG Chatbot", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(15, 18, 195, 18)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(100, 110, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def add_heading(pdf, text, level=1):
    pdf.ln(4)
    if level == 1:
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(26, 54, 93)  # Dark Blue
        pdf.cell(pdf.epw, 8, text, new_x="LMARGIN", new_y="NEXT")
        pdf.line(pdf.get_x(), pdf.get_y(), 195, pdf.get_y())
        pdf.ln(2)
    elif level == 2:
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(45, 55, 72)  # Charcoal
        pdf.cell(pdf.epw, 7, text, new_x="LMARGIN", new_y="NEXT")
    elif level == 3:
        pdf.set_font("helvetica", "BI", 10)
        pdf.set_text_color(74, 85, 104) # Grey
        pdf.cell(pdf.epw, 6, text, new_x="LMARGIN", new_y="NEXT")

def add_paragraph(pdf, text):
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(45, 55, 72)
    pdf.multi_cell(pdf.epw, 5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

def add_bullet(pdf, label, desc):
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(45, 55, 72)
    pdf.write(5, f"  * {label}: ")
    pdf.set_font("helvetica", "", 10)
    pdf.write(5, f"{desc}\n")
    pdf.ln(1)

def add_code_block(pdf, text):
    pdf.set_font("courier", "", 9)
    pdf.set_fill_color(240, 242, 245)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(pdf.epw, 4.5, text, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

def add_query_block(pdf, query, retrieved, answer, status):
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(44, 122, 123)  # Teal
    pdf.cell(pdf.epw, 6, f"Query: \"{query}\"", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "I", 9.5)
    pdf.set_text_color(74, 85, 104)
    pdf.multi_cell(pdf.epw, 5, f"Retrieved Chunks: {retrieved}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(45, 55, 72)
    pdf.multi_cell(pdf.epw, 5, f"LLM Grounded Answer: {answer}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "B", 10)
    if "Pass" in status:
        pdf.set_text_color(34, 139, 34)  # Forest Green
    else:
        pdf.set_text_color(178, 34, 34)   # Firebrick Red
    pdf.cell(pdf.epw, 6, f"Evaluation Status: {status}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

def generate_pdf():
    pdf = RAGTechnicalReport(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Title Block
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(26, 54, 93) # Dark Blue
    pdf.cell(pdf.epw, 12, "TECHNICAL REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(74, 85, 104) # Slate grey
    pdf.cell(pdf.epw, 8, "Instruction-Tuned Hybrid RAG Chatbot with Local Fallback", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    
    pdf.set_font("helvetica", "", 9.5)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(pdf.epw, 5, "Author: Yadven", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(pdf.epw, 5, "GitHub Repository: https://github.com/yadven7/rag-chatbot-streamlit", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(pdf.epw, 5, "Demo Video: https://drive.google.com/file/d/1O2rE2RlACqSimnAsqdnaIaV5InqSek_v/view?usp=drive_link", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.line(15, 53, 195, 53)
    pdf.ln(3)
    
    # Section 1
    add_heading(pdf, "1. Document Structure & Chunking Logic", level=1)
    add_heading(pdf, "Text Extraction and Cleaning", level=2)
    add_paragraph(pdf, "Before indexing, document text is extracted and cleaned to reduce parsing noise. This includes stripping HTML/XML tags, normalising tabs and newlines, and collapsing multiple continuous spaces. Common headers, page numbers, and footer metadata are filtered out to prevent indexing duplicates that add noise to the semantic matching layer.")
    
    add_heading(pdf, "Sentence-Aware Chunking Strategy", level=2)
    add_paragraph(pdf, "Rather than splitting text at arbitrary character boundaries, the Chunker segments texts on sentence terminals (.!?). Sentences are greedily aggregated into a single chunk up to a length constraint:")
    add_bullet(pdf, "Length Range", "Each chunk must stay within a strict limit of 100 to 300 words.")
    add_bullet(pdf, "Grammatical Coherence", "Sentences are never split in half, preserving complete contextual statements.")
    add_bullet(pdf, "Contextual Integrity", "Chunks are small enough to stay within LLM context window limits but large enough to contain complete, semantic facts.")

    # Section 2
    add_heading(pdf, "2. Embedding Model & Vector Database Selection", level=1)
    add_heading(pdf, "Semantic Embeddings via 'all-MiniLM-L6-v2'", level=2)
    add_paragraph(pdf, "The system embeds chunks using 'all-MiniLM-L6-v2', a 384-dimensional SentenceTransformer model trained on over 1 billion sentence pairs. This model is chosen for its efficiency, low memory foot-print, and high performance on CPU architectures. Additionally, 'BAAI/bge-small-en-v1.5' is integrated as an alternate embedding model option in the UI sidebar.")
    
    add_heading(pdf, "Vector Storage with local FAISS Indexing", level=2)
    add_paragraph(pdf, "Dense embeddings are stored in a local flat L2 index managed by FAISS (Facebook AI Similarity Search). A local file serialization (index.faiss) is used to avoid external cloud latency or hosting overheads. This approach delivers sub-millisecond retrieval speeds.")
    
    add_heading(pdf, "Hybrid Semantic-Lexical Scoring", level=2)
    add_paragraph(pdf, "To prevent semantic search failures on specific legal phrases, we implement a hybrid retriever. Query tokens are filtered for stopwords and expanded (e.g. plural suffixes mapped to singular). For query phrases like 'user obligations', keywords are expanded with compliance terms ('you agree', 'you must', 'prohibited', etc.). The retriever merges the results and ranks chunks by combined score: combined_score = semantic_score + 0.1 * keyword_score, where semantic_score = 1.0 / (1.0 + L2_distance).")

    # Section 3
    add_heading(pdf, "3. Prompt Engineering & Grounding Constraints", level=1)
    add_paragraph(pdf, "The system prompt acts as a set of guardrails to prevent LLM hallucinations. It instructs the assistant to respond only based on the supplied context chunks, to cite sources explicitly, and to emit a standard fallback message if the answer is missing:")
    
    add_code_block(pdf, "System Instructions:\n"
                        "You are a helpful, precise, and fact-focused AI Assistant. Your task is to\n"
                        "answer the user's query based ONLY on the provided Context Sources.\n"
                        "Strictly adhere to the following rules:\n"
                        "1. Rely only on the clear facts mentioned in the Context. Do not make up\n"
                        "   facts, external details or assumptions.\n"
                        "2. If the answer cannot be found in the provided Context, say: 'I apologize,\n"
                        "   but the provided documents do not contain the answer to your query.'\n"
                        "3. Cite which Context Source(s) you used to construct your answer.\n"
                        "4. Keep your response factual, grounded, and concise.")
    
    # Section 4
    add_heading(pdf, "4. Evaluation Queries & Results", level=1)
    add_paragraph(pdf, "Below are evaluation runs executed against the ingested eBay User Agreement corpus:")
    
    add_query_block(pdf, 
                    "What is the dispute process and what is the address to mail a notice of dispute to eBay?",
                    "Chunk 31 & 32 (informal resolution details)",
                    "\"According to the documents, you must participate in an informal dispute resolution process by sending a written Notice of Dispute. The notice must be sent via email to DisputeNotice@eBay.com or regular mail to: eBay Inc., Attn: Litigation Department, RE: Notice of Dispute, 583 W. eBay Way, Draper, UT 84020 [Source 1].\"",
                    "Pass (Extremely accurate, correct address retrieval, zero hallucination)")
    
    add_query_block(pdf, 
                    "What are the rules regarding selling fresh fruits on eBay?",
                    "Various unrelated policy pages",
                    "\"I apologize, but the provided documents do not contain the answer to your query regarding the rules for selling fresh fruits on eBay.\"",
                    "Pass (Correctly triggered the out-of-context fallback phrase instead of hallucinating)")

    add_query_block(pdf, 
                    "What happens if a seller fails to send the buyer a return shipping label?",
                    "Chunk 18 & 19 (Returns policies)",
                    "\"If a seller fails to send a return shipping label to the buyer, an eBay-generated shipping label will be used, and the seller will be responsible for its cost [Source 1].\"",
                    "Pass (Retrieved the correct returns policy detail)")

    add_query_block(pdf, 
                    "What are the user obligations?",
                    "Chunk 20, Chunk 22, Chunk 21, Chunk 0, Chunk 8 (Triggered top_k=15 broad term scaling)",
                    "\"User obligations include agreeing to rules for buying/selling, complying with laws, keeping account details secure, and maintaining a valid payment method on file to settle fees [Source 1, Source 2].\"",
                    "Pass (Keyword expansion successfully promoted user obligation clauses over general arbitration sections)")

    add_query_block(pdf, 
                    "What happens after termination?",
                    "Chunk 40, Chunk 39, Chunk 27 (Termination survival clauses)",
                    "\"Upon termination, certain terms survive, including the Agreement to Arbitrate and your obligations to satisfy outstanding payment fees incurred before termination [Source 1, Source 2].\"",
                    "Pass (Correctly retrieved post-agreement survival details)")

    # Section 5
    add_heading(pdf, "5. Technical Limitations, Hallucinations & LLM Fallbacks", level=1)
    
    add_heading(pdf, "Hallucination Defense", level=2)
    add_paragraph(pdf, "Setting temperature to 0.0 forces the model to choose high-probability tokens, making answers deterministic. Combined with grounding constraints, this prevents the model from generating creative content outside the text context.")
    
    add_heading(pdf, "Gemini 429 Quota Exceeded Fallback to Local Ollama", level=2)
    add_paragraph(pdf, "A common RAG issue in production is API rate limits. In our design, Google Gemini is the default cloud generator. If Gemini encounters a 429 rate limit error (ResourceExhausted) or has an invalid/missing API key, the pipeline automatically catches the exception, outputs an informative fallback warning in the UI, and routes generation to a local Ollama instance (llama3.2:3b) on localhost. This ensures the app is completely reliable and self-contained.")
    
    add_heading(pdf, "Inference Latency and CPU Limitations", level=2)
    add_paragraph(pdf, "While Ollama running llama3.2:3b locally provides a strong offline fallback, local CPU-only execution can introduce latency. To mitigate this, streaming response rendering is utilized, displaying tokens in real time to the user to keep perceived latency to a minimum.")
    
    # Save the PDF file
    pdf.output("report.pdf")
    print("report.pdf generated successfully.")

if __name__ == "__main__":
    generate_pdf()
