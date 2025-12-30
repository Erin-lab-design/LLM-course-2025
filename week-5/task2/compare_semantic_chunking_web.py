"""
Task 2: Semantic Chunking Comparison - Web Interface
=====================================================
Interactive Streamlit app to compare Fixed-size vs Semantic chunking strategies.
"""

import sys
import os

# Add parent directory to path - must be done before other imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import torch
import ollama
from sentence_transformers import SentenceTransformer
from time import perf_counter as timer, sleep
import json
from datetime import datetime
import gc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from util import pdf_utils
from util.embedings_utils import embed_chunks
from util.nlp_utils import sentencize_jieba, split_list_overlapping, chunks_to_text_elems
from util.vector_search_utils import retrieve_relevant_resources
from util.evaluation_utils import (
    calculate_context_relevance,
    calculate_answer_relevance,
    calculate_faithfulness
)
from semantic_chunking import semantic_chunk, analyze_chunking

# Page config
st.set_page_config(
    page_title="Task 2: Semantic Chunking Comparison",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state to preserve results after download button clicks
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'fixed_results' not in st.session_state:
    st.session_state.fixed_results = None
if 'semantic_results' not in st.session_state:
    st.session_state.semantic_results = None
if 'chunking_metrics' not in st.session_state:
    st.session_state.chunking_metrics = None
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = None
if 'timestamp' not in st.session_state:
    st.session_state.timestamp = None

st.title("🧠 Task 2: Semantic Chunking Comparison")
st.markdown("""
Compare **Fixed-size** (Task 1 baseline) vs **Semantic** (Task 2) chunking strategies.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Test Configuration")

# Load queries
@st.cache_data
def load_queries():
    with open("../test_queries_professional.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

all_queries = load_queries()

# Select number of queries to test
num_queries = st.sidebar.slider(
    "Number of Test Queries",
    min_value=1,
    max_value=len(all_queries),
    value=len(all_queries),
    help=f"Select how many queries to test (max {len(all_queries)}). Aggressive GPU memory cleanup enabled."
)

selected_queries = all_queries[:num_queries]

# Semantic chunking parameters (adjusted to match baseline chunk count ~108)
st.sidebar.markdown("### 🧠 Semantic Chunking Config")
SIMILARITY_THRESHOLD = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.40,
    max_value=0.70,
    value=0.50,
    step=0.05,
    help="Lower = larger chunks, target ~108 chunks like baseline"
)
MAX_CHUNK_SIZE = st.sidebar.number_input("Max Chunk Size (sentences)", value=10, min_value=8, max_value=15)
MIN_CHUNK_SIZE = st.sidebar.number_input("Min Chunk Size (sentences)", value=7, min_value=5, max_value=10)
OVERLAP = st.sidebar.number_input("Overlap (sentences)", value=2, min_value=0, max_value=3, help="Match baseline's overlap=2")

# Show selected queries
with st.sidebar.expander("📝 Selected Queries", expanded=False):
    for i, q in enumerate(selected_queries, 1):
        st.text(f"{i}. {q[:60]}{'...' if len(q) > 60 else ''}")

# PDF and model settings
PDF_PATH = "../lawbook_sample.pdf"
OLLAMA_MODEL = "llama3"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
NUM_RESULTS = 10  # Same as command-line version
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Test Settings:**
- Document: {PDF_PATH}
- Embedding: {EMBEDDING_MODEL.split('/')[-1]}
- LLM: {OLLAMA_MODEL}
- Device: {DEVICE}
- Queries: {num_queries}
""")

# Configuration display
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 FIXED-SIZE Chunking")
    st.code("""
Strategy: Task 1 Best Config
- Segmentation: jieba (Chinese)
- Chunk Size: 10 sentences
- Overlap: 2 sentences
- Hybrid Search: Enabled
- Variable Size: No
    """, language="yaml")

with col2:
    st.subheader("🧠 SEMANTIC Chunking")
    st.code(f"""
Strategy: Task 2 Semantic
- Segmentation: jieba (Chinese)
- Similarity Threshold: {SIMILARITY_THRESHOLD}
- Min/Max Size: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} sentences
- Overlap: {OVERLAP} sentences
- Hybrid Search: Enabled
- Variable Size: Yes (adaptive)
    """, language="yaml")

def run_pipeline(strategy_name, queries, chunks, embeddings_tensor, progress_bar, status_text):
    """Run RAG pipeline with given chunks"""
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    results = []
    
    for i, query in enumerate(queries):
        status_text.text(f"🔍 {strategy_name}: Query {i+1}/{len(queries)}")
        progress_bar.progress((i) / len(queries))
        
        start = timer()
        
        # Retrieve
        scores, indices = retrieve_relevant_resources(
            query,
            embeddings_tensor,
            embedding_model,
            st=None,
            n_resources_to_return=NUM_RESULTS,
            pages_and_chunks=chunks,
            use_hybrid=True,
            print_time=False
        )
        
        # Convert scores tensor to list
        scores_list = scores.cpu().tolist() if isinstance(scores, torch.Tensor) else scores
        retrieved_chunks = [chunks[idx]['sentence_chunk'] for idx in indices]
        
        # Generate (use same prompt as compare_ui_versions_web)
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""根据以下检索到的法律文本，用中文回答问题。

检索文本：
{context}

问题：{query}

答案："""
        
        # Retry mechanism for CUDA errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.7}
                )
                answer = response['message']['content']
                break
            except Exception as e:
                if "CUDA" in str(e) or "status code: 500" in str(e):
                    if attempt < max_retries - 1:
                        status_text.text(f"⚠️ GPU memory issue, retrying... ({attempt+1}/{max_retries})")
                        # Clear GPU cache
                        torch.cuda.empty_cache()
                        gc.collect()
                        sleep(2)  # Wait for GPU to recover
                        continue
                    else:
                        st.error(f"❌ CUDA error after {max_retries} retries. Please restart Ollama service.")
                        raise
                else:
                    raise
        
        # Small delay to prevent GPU overload
        sleep(0.5)
        elapsed = timer() - start
        
        # Aggressively clear GPU memory after each query
        torch.cuda.empty_cache()
        gc.collect()
        
        # Evaluate
        cr = calculate_context_relevance(query, retrieved_chunks, scores_list)
        ar = calculate_answer_relevance(query, answer)
        f = calculate_faithfulness(answer, retrieved_chunks)
        overall = (cr + ar + f) / 3
        
        results.append({
            'query': query,
            'answer': answer,
            'context_relevance': cr,
            'answer_relevance': ar,
            'faithfulness': f,
            'overall': overall,
            'time': elapsed
        })
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ {strategy_name} completed!")
    
    # Calculate averages
    avg = {
        'context_relevance': sum(r['context_relevance'] for r in results) / len(results),
        'answer_relevance': sum(r['answer_relevance'] for r in results) / len(results),
        'faithfulness': sum(r['faithfulness'] for r in results) / len(results),
        'overall': sum(r['overall'] for r in results) / len(results),
        'time': sum(r['time'] for r in results) / len(results)
    }
    
    return {'strategy': strategy_name, 'results': results, 'averages': avg}

# Run comparison button
if st.button("🚀 Run Comparison Test", type="primary", use_container_width=True):
    
    st.markdown("---")
    st.subheader("🧪 Running Tests...")
    
    # Load PDF and preprocess (shared for both strategies)
    with st.spinner("Loading and preprocessing PDF..."):
        pages_and_texts = pdf_utils.open_and_read_pdf(PDF_PATH)
        pages_and_texts = sentencize_jieba(pages_and_texts)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    
    st.success(f"✅ Loaded {len(pages_and_texts)} pages")
    
    # ============================================================================
    # STRATEGY 1: Fixed-size chunking
    # ============================================================================
    st.markdown("### 📦 Testing Fixed-size Chunking...")
    fixed_progress = st.progress(0)
    fixed_status = st.empty()
    
    with st.spinner("Creating fixed-size chunks..."):
        # Create baseline chunks
        for item in pages_and_texts:
            item["sentence_chunks"] = split_list_overlapping(item["sentences"], 10, 2)
        fixed_chunks = chunks_to_text_elems(pages_and_texts)
        
        # Embed
        embed_chunks(fixed_chunks, embedding_model)
        fixed_embeddings = torch.tensor(
            np.array([chunk['embedding'] for chunk in fixed_chunks]), 
            dtype=torch.float32
        ).to(DEVICE)
        
        st.info(f"📦 Created {len(fixed_chunks)} fixed-size chunks")
    
    # Run queries for fixed-size
    fixed_results = run_pipeline(
        "Fixed-size", 
        selected_queries, 
        fixed_chunks,
        fixed_embeddings,
        fixed_progress, 
        fixed_status
    )
    
    # Aggressive memory cleanup between strategies
    st.info("🧹 Cleaning GPU memory before semantic chunking...")
    del fixed_embeddings
    torch.cuda.empty_cache()
    gc.collect()
    sleep(2)  # Allow GPU to fully clear
    
    # ============================================================================
    # STRATEGY 2: Semantic chunking
    # ============================================================================
    st.markdown("### 🧠 Testing Semantic Chunking...")
    semantic_progress = st.progress(0)
    semantic_status = st.empty()
    
    with st.spinner("Creating semantic chunks..."):
        # Create semantic chunks
        semantic_chunks_list = []
        all_stats = []
        
        for item in pages_and_texts:
            page_chunks, page_stats = semantic_chunk(
                item["sentences"],
                embedding_model,
                similarity_threshold=SIMILARITY_THRESHOLD,
                max_chunk_size=MAX_CHUNK_SIZE,
                min_chunk_size=MIN_CHUNK_SIZE,
                overlap=OVERLAP,
                verbose=False
            )
            
            # Convert to dict format
            for chunk_text, chunk_stat in zip(page_chunks, page_stats):
                semantic_chunks_list.append({
                    'page_number': item['page_number'],
                    'sentence_chunk': chunk_text,
                    'chunk_char_count': len(chunk_text),
                    'chunk_word_count': len(chunk_text.split()),
                    'chunk_token_count': len(chunk_text) // 4,
                    'chunk_size': chunk_stat['size'],
                    'avg_similarity': chunk_stat['avg_similarity']
                })
            
            all_stats.extend(page_stats)
        
        # Analyze chunking
        chunking_metrics = analyze_chunking(all_stats)
        
        # Embed
        embed_chunks(semantic_chunks_list, embedding_model)
        semantic_embeddings = torch.tensor(
            np.array([chunk['embedding'] for chunk in semantic_chunks_list]), 
            dtype=torch.float32
        ).to(DEVICE)
        
        st.info(f"🧠 Created {len(semantic_chunks_list)} semantic chunks | "
                f"Avg size: {chunking_metrics['avg_chunk_size']:.1f} sentences | "
                f"Similarity: {chunking_metrics['avg_within_chunk_similarity']:.3f}")
    
    # Run queries for semantic
    semantic_results = run_pipeline(
        "Semantic", 
        selected_queries, 
        semantic_chunks_list,
        semantic_embeddings,
        semantic_progress, 
        semantic_status
    )
    
    # Save results to session state
    st.session_state.results_ready = True
    st.session_state.fixed_results = fixed_results
    st.session_state.semantic_results = semantic_results
    st.session_state.chunking_metrics = chunking_metrics
    st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# RESULTS VISUALIZATION (outside button block to persist after download clicks)
# ============================================================================
if st.session_state.results_ready:
    st.markdown("---")
    st.header("📊 Results Comparison")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    fixed_avg = st.session_state.fixed_results['averages']
    semantic_avg = st.session_state.semantic_results['averages']
    
    with col1:
        st.metric(
            "Overall Score",
            f"{semantic_avg['overall']:.3f}",
            f"{semantic_avg['overall'] - fixed_avg['overall']:+.3f}",
            help="Higher is better"
        )
    
    with col2:
        st.metric(
            "Context Relevance",
            f"{semantic_avg['context_relevance']:.3f}",
            f"{semantic_avg['context_relevance'] - fixed_avg['context_relevance']:+.3f}"
        )
    
    with col3:
        st.metric(
            "Avg Time (s)",
            f"{semantic_avg['time']:.1f}",
            f"{semantic_avg['time'] - fixed_avg['time']:+.1f}",
            delta_color="inverse"
        )
    
    # Bar chart comparison
    metrics_df = pd.DataFrame({
        'Metric': ['Context\nRelevance', 'Answer\nRelevance', 'Faithfulness', 'Overall\nScore'],
        'Fixed-size': [fixed_avg['context_relevance'], fixed_avg['answer_relevance'], 
                       fixed_avg['faithfulness'], fixed_avg['overall']],
        'Semantic': [semantic_avg['context_relevance'], semantic_avg['answer_relevance'], 
                     semantic_avg['faithfulness'], semantic_avg['overall']]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Fixed-size', x=metrics_df['Metric'], y=metrics_df['Fixed-size'], 
                         marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Semantic', x=metrics_df['Metric'], y=metrics_df['Semantic'], 
                         marker_color='lightgreen'))
    
    fig.update_layout(
        title='Performance Comparison by Metric',
        yaxis_title='Score',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    
    if st.session_state.comparison_df is None:
        comparison_df = pd.DataFrame({
            'Metric': ['Context Relevance', 'Answer Relevance', 'Faithfulness', 'Overall Score'],
            'Fixed-size': [f"{fixed_avg['context_relevance']:.4f}", 
                           f"{fixed_avg['answer_relevance']:.4f}",
                           f"{fixed_avg['faithfulness']:.4f}",
                           f"{fixed_avg['overall']:.4f}"],
            'Semantic': [f"{semantic_avg['context_relevance']:.4f}",
                         f"{semantic_avg['answer_relevance']:.4f}",
                         f"{semantic_avg['faithfulness']:.4f}",
                         f"{semantic_avg['overall']:.4f}"],
            'Δ': [f"{semantic_avg['context_relevance'] - fixed_avg['context_relevance']:+.4f}",
                  f"{semantic_avg['answer_relevance'] - fixed_avg['answer_relevance']:+.4f}",
                  f"{semantic_avg['faithfulness'] - fixed_avg['faithfulness']:+.4f}",
                  f"{semantic_avg['overall'] - fixed_avg['overall']:+.4f}"],
            'Δ%': [f"{(semantic_avg['context_relevance'] - fixed_avg['context_relevance']) / fixed_avg['context_relevance'] * 100:+.1f}%",
                   f"{(semantic_avg['answer_relevance'] - fixed_avg['answer_relevance']) / fixed_avg['answer_relevance'] * 100:+.1f}%",
                   f"{(semantic_avg['faithfulness'] - fixed_avg['faithfulness']) / fixed_avg['faithfulness'] * 100:+.1f}%",
                   f"{(semantic_avg['overall'] - fixed_avg['overall']) / fixed_avg['overall'] * 100:+.1f}%"]
        })
        st.session_state.comparison_df = comparison_df
    else:
        comparison_df = st.session_state.comparison_df
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Query-by-query comparison
    with st.expander("🔍 Query-by-Query Results", expanded=False):
        for i, (q, fr, sr) in enumerate(zip(selected_queries, st.session_state.fixed_results['results'], st.session_state.semantic_results['results']), 1):
            st.markdown(f"**Query {i}:** {q}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📦 Fixed-size**")
                st.write(f"Overall: {fr['overall']:.3f}")
                st.text_area(f"Answer (Fixed) {i}", fr['answer'], height=100, key=f"fixed_{i}")
            
            with col2:
                st.markdown("**🧠 Semantic**")
                st.write(f"Overall: {sr['overall']:.3f} ({sr['overall'] - fr['overall']:+.3f})")
                st.text_area(f"Answer (Semantic) {i}", sr['answer'], height=100, key=f"semantic_{i}")
            
            st.markdown("---")
    
    # Chunking efficiency comparison
    st.subheader("📦 Chunking Efficiency")
    
    # Get fixed chunks count from first result metadata or estimate
    fixed_chunks_count = 108  # Known from baseline config
    
    efficiency_data = {
        'Strategy': ['Fixed-size', 'Semantic'],
        'Total Chunks': [fixed_chunks_count, st.session_state.chunking_metrics['num_chunks']],
        'Avg Chunk Size': ['10.0 (fixed)',
                          f"{st.session_state.chunking_metrics['avg_chunk_size']:.1f}"],
        'Size Variance': ['N/A (fixed)', f"{st.session_state.chunking_metrics['chunk_size_std']:.2f}"]
    }
    
    st.table(pd.DataFrame(efficiency_data))
    
    # Save results
    st.markdown("---")
    st.subheader("💾 Save Results")
    
    timestamp = st.session_state.timestamp
    
    # Prepare data for export
    export_data = {
        'metadata': {
            'timestamp': timestamp,
            'num_queries': num_queries,
            'pdf_path': PDF_PATH,
            'embedding_model': EMBEDDING_MODEL,
            'llm_model': OLLAMA_MODEL,
            'semantic_config': {
                'similarity_threshold': SIMILARITY_THRESHOLD,
                'max_chunk_size': MAX_CHUNK_SIZE,
                'min_chunk_size': MIN_CHUNK_SIZE
            }
        },
        'fixed_size': st.session_state.fixed_results,
        'semantic': st.session_state.semantic_results,
        'chunking_metrics': st.session_state.chunking_metrics
    }
    
    # Prepare download data
    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
    summary_csv = st.session_state.comparison_df.to_csv(index=False, encoding='utf-8-sig')
    
    # Place download buttons side by side to prevent page refresh issues
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download Detailed Results (JSON)",
            data=json_str,
            file_name=f"semantic_comparison_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=summary_csv,
            file_name=f"semantic_comparison_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.success("✅ Comparison complete! Results are ready for download.")

# Show instructions when no results yet
if not st.session_state.results_ready:
    st.info("👆 Configure parameters and click the button above to start the comparison test")
    
    st.markdown("""
    ### How it works:
    
    1. **Fixed-size Chunking** (Baseline):
       - Uses 10 sentences per chunk with 2-sentence overlap
       - Consistent chunk sizes
       - Task 1 best configuration
    
    2. **Semantic Chunking** (Task 2):
       - Analyzes sentence similarity to detect topic boundaries
       - Variable chunk sizes based on content coherence
       - Adapts to natural text structure
    
    3. **Evaluation Metrics**:
       - **Context Relevance**: How well retrieved chunks match the query
       - **Answer Relevance**: How well the answer addresses the query
       - **Faithfulness**: How grounded the answer is in retrieved text
       - **Overall Score**: Average of the three metrics
    
    ### Expected Benefits of Semantic Chunking:
    - 🎯 Better topic coherence within chunks
    - 📈 Improved retrieval accuracy
    - 🧠 More contextually appropriate chunk boundaries
    - ⚡ Potentially better overall RAG performance
    """)
