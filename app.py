import streamlit as st
import os
import time
from dotenv import load_dotenv

# Import pipeline orchestrator
try:
    from src.rag_pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot | Amlgo Labs",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom Styling CSS
st.markdown("""
<style>
    /* Fonts and Overall Aesthetics */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Sleek background gradient */
    .stApp {
        background: linear-gradient(180deg, #0B0E14 0%, #151922 100%);
        color: #E2E8F0;
    }
    
    /* Title container with subtle glow */
    .header-container {
        text-align: center;
        padding: 2rem 1.5rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.6rem;
        font-weight: 300;
    }
    
    /* Sidebar Design */
    section[data-testid="stSidebar"] {
        background-color: #080A0E !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Citations / Source cards styling */
    .source-box {
        background: rgba(59, 130, 246, 0.05);
        border-left: 3px solid #3B82F6;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-top: 0.8rem;
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.5;
        border-top: 1px solid rgba(255, 255, 255, 0.02);
        border-right: 1px solid rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid rgba(255, 255, 255, 0.02);
    }
    
    .source-header {
        font-weight: 600;
        color: #60A5FA;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Chat inputs custom layout */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Status Badge styling */
    .status-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-active {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-inactive {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configurations
with st.sidebar:
    st.image(
    "https://www.amlgolabs.com/wp-content/uploads/2021/04/amlgolabs-logo.png",
    width=180
    )

    st.markdown("## 🛠️ Configuration Panel")
    # Model configuration
    st.markdown("### 🧠 LLM Engine Selection")
    active_llm = st.selectbox(
        "Generative LLM Model",
        ["Ollama llama3.2:3b", "Gemini gemini-2.0-flash"],
        index=0
    )
    
    st.markdown("### 🔤 Semantic Embedding")
    embedding_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
        index=0
    )
    
    st.markdown("### 💾 Vector Database")
    vector_db_type = st.text_input("Active Index Type", value="FAISS", disabled=True)
    
    # Pipeline status
    st.markdown("### 📊 Index Status")
    
    # Instantiate RAG Pipeline with chosen embedding model
    @st.cache_resource(show_spinner=False)
    def initialize_pipeline(model_name: str) -> RAGPipeline:
        if RAGPipeline is not None:
            return RAGPipeline(embedding_model=model_name)
        return None

    pipeline = initialize_pipeline(embedding_model)
    db_loaded = pipeline is not None and pipeline.retriever.is_index_loaded()
    
    if db_loaded:
        chunk_count = pipeline.retriever.get_chunk_count()
        st.markdown(f'<div class="status-badge status-active">● Index Loaded</div>', unsafe_allow_html=True)
        st.metric(label="Indexed Chunks Count", value=chunk_count)
    else:
        st.markdown(f'<div class="status-badge status-inactive">● No Index Detected</div>', unsafe_allow_html=True)
        st.metric(label="Indexed Chunks Count", value=0)
        
        st.markdown("⚠️ Place a PDF document in `data/` and trigger build:")
        if st.button("🚀 Ingest & Build Vector DB", use_container_width=True):
            with st.spinner("Processing document corpus..."):
                try:
                    from src.ingest import run_ingestion
                    stats = run_ingestion(embedding_model=embedding_model)
                    st.success(f"Successfully processed {stats.get('chunks_count')} chunks!")
                    # Clear pipeline cache so it reloads the newly created index
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    
    st.markdown("---")
    
    # System Actions
    st.markdown("### ⚙️ Chat Utilities")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CONTENT AREA ---
st.markdown("""
<div class="header-container">
    <h1 class="main-title">Instruction-Tuned RAG Chatbot</h1>
    <div class="sub-title">Serving real-time streaming grounded answers using FAISS and Google Gemini API</div>
</div>
""", unsafe_allow_html=True)

# Initialize Session Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages from History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Retrieved Grounding Context Sources", expanded=False):
                for idx, source in enumerate(msg["sources"]):
                    chunk_id = source.get('chunk_id', 'N/A')
                    sem_score = source.get('semantic_score', 0.0)
                    kw_matches = ", ".join(source.get('keyword_matches', [])) or "None"
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">Source Context Chunk #{idx+1} (Chunk ID: {chunk_id})</div>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.2rem; margin-bottom: 0.5rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 0.3rem;">
                            <strong>Semantic Sim:</strong> {sem_score:.4f} | 
                            <strong>Keyword Matches:</strong> {kw_matches}
                        </div>
                        {source.get('text', '')}
                    </div>
                    """, unsafe_allow_html=True)

# Handle New Question Queries
if query_input := st.chat_input("Ask a question about the uploaded document corpus..."):
    # Display user input
    with st.chat_message("user"):
        st.markdown(query_input)
        
    st.session_state.messages.append({"role": "user", "content": query_input})
    
    # Display assistant placeholder
    with st.chat_message("assistant"):
        response_text_area = st.empty()
        sources_expander_area = st.empty()
        
        full_assistant_reply = ""
        retrieved_contexts = []
        
        if db_loaded and pipeline is not None:
            # 1. Retrieve
            with st.spinner("Searching document index..."):
                # Fetch top_k=8 chunks
                retrieved_contexts = pipeline.retrieve(query_input, top_k=8)
                
            # 2. Generate and Stream
            stream = pipeline.generate_stream(query_input, retrieved_contexts, model_name=active_llm)
            for chunk in stream:
                full_assistant_reply += chunk
                response_text_area.markdown(full_assistant_reply + "▌")
            response_text_area.markdown(full_assistant_reply)
        else:
            # Inform user if DB index is not generated
            warning_msg = (
                "The RAG index has not been initialized. "
                "Please add a PDF to the `/data` folder and click the "
                "**Ingest & Build Vector DB** button in the sidebar."
            )
            for word in warning_msg.split():
                full_assistant_reply += word + " "
                response_text_area.markdown(full_assistant_reply + "▌")
                time.sleep(0.05)
            response_text_area.markdown(full_assistant_reply)
            
        # Display citation chunks below answer
        if retrieved_contexts:
            with sources_expander_area.expander("📚 Retrieved Grounding Context Sources", expanded=False):
                for idx, source in enumerate(retrieved_contexts):
                    chunk_id = source.get('chunk_id', 'N/A')
                    sem_score = source.get('semantic_score', 0.0)
                    kw_matches = ", ".join(source.get('keyword_matches', [])) or "None"
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">Source Context Chunk #{idx+1} (Chunk ID: {chunk_id})</div>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.2rem; margin-bottom: 0.5rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 0.3rem;">
                            <strong>Semantic Sim:</strong> {sem_score:.4f} | 
                            <strong>Keyword Matches:</strong> {kw_matches}
                        </div>
                        {source.get('text', '')}
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Append message history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_assistant_reply,
            "sources": retrieved_contexts
        })
