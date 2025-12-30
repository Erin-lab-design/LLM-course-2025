"""
Compare UI Versions - Web Interface
Interactive Streamlit app to compare baseline vs improved RAG configurations
"""

import sys
import os

# Add parent directory to path to import util modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import torch
import ollama
from sentence_transformers import SentenceTransformer
from time import perf_counter as timer
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from util import pdf_utils
from util.embedings_utils import embed_chunks
from util.nlp_utils import chunks_to_text_elems, sentencize_jieba
from util.vector_search_utils import retrieve_relevant_resources
from util.evaluation_utils import (
    calculate_context_relevance,
    calculate_answer_relevance,
    calculate_faithfulness
)

# Page config
st.set_page_config(
    page_title="RAG Configuration Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Configuration Comparison Tool")
st.markdown("""
Compare **Baseline** (pdf_rag_ui_ollama_baseline.py) vs **Improved** (pdf_rag_ui_ollama.py) configurations.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Test Configuration")

# Load queries
@st.cache_data
def load_queries():
    with open("test_queries_professional.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

all_queries = load_queries()

# Select number of queries to test
num_queries = st.sidebar.slider(
    "Number of Test Queries",
    min_value=1,
    max_value=len(all_queries),
    value=min(3, len(all_queries)),
    help=f"Select how many queries to test (max {len(all_queries)})"
)

selected_queries = all_queries[:num_queries]

# Show selected queries
with st.sidebar.expander("📝 Selected Queries", expanded=False):
    for i, q in enumerate(selected_queries, 1):
        st.text(f"{i}. {q[:60]}{'...' if len(q) > 60 else ''}")

# PDF and model settings
PDF_PATH = "lawbook_sample.pdf"
OLLAMA_MODEL = "llama3"

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Test Settings:**
- Document: {PDF_PATH}
- LLM: {OLLAMA_MODEL}
- Queries: {num_queries}
""")

# Configuration display
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 BASELINE Configuration")
    st.code("""
Embedding: all-mpnet-base-v2
Segmentation: spacy (English)
Chunking: 10 sentences, no overlap
Hybrid Search: Disabled
Results: 10
    """, language="yaml")

with col2:
    st.subheader("🟢 IMPROVED Configuration")
    st.code("""
Embedding: bge-large-zh-v1.5
Segmentation: jieba (Chinese)
Chunking: 10 sentences, 2 overlap
Hybrid Search: Enabled
Results: 10
    """, language="yaml")

def run_pipeline(config_name, queries, progress_bar, status_text):
    """Run RAG pipeline with specified configuration"""
    
    is_baseline = (config_name == "BASELINE")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration
    if is_baseline:
        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        use_jieba = False
        use_overlap = False
        use_hybrid = False
    else:
        embedding_model_name = "BAAI/bge-large-zh-v1.5"
        use_jieba = True
        use_overlap = True
        use_hybrid = True
    
    status_text.text(f"📄 Loading PDF...")
    pages_and_texts = pdf_utils.open_and_read_pdf(PDF_PATH)
    
    # Sentencize
    status_text.text(f"📝 Segmenting text...")
    if use_jieba:
        pages_and_texts = sentencize_jieba(pages_and_texts)
    else:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        for item in pages_and_texts:
            doc = nlp(item["text"])
            item["sentences"] = [str(sent) for sent in doc.sents]
    
    # Chunk
    status_text.text(f"📦 Creating chunks...")
    if use_overlap:
        from util.nlp_utils import split_list_overlapping
        for item in pages_and_texts:
            item["sentence_chunks"] = split_list_overlapping(item["sentences"], 10, 2)
    else:
        from util.nlp_utils import split_list
        for item in pages_and_texts:
            item["sentence_chunks"] = split_list(item["sentences"], 10)
    
    chunks = chunks_to_text_elems(pages_and_texts)
    
    # Embeddings
    status_text.text(f"🔢 Creating embeddings...")
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    embed_chunks(chunks, embedding_model)
    embeddings_tensor = torch.tensor([c['embedding'] for c in chunks], dtype=torch.float32).to(device)
    
    # Test queries
    results = []
    total = len(queries)
    
    for i, query in enumerate(queries):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"🔍 Testing query {i+1}/{total}: {query[:40]}...")
        
        start = timer()
        
        # Retrieve
        scores, indices = retrieve_relevant_resources(
            query=query,
            embeddings=embeddings_tensor,
            model=embedding_model,
            st=None,
            n_resources_to_return=10,
            print_time=False,
            pages_and_chunks=chunks if use_hybrid else None,
            use_hybrid=use_hybrid
        )
        
        # Generate
        retrieved_chunks = [chunks[idx]['sentence_chunk'] for idx in indices]
        context = "\n\n".join(retrieved_chunks)
        
        prompt = f"""根据以下检索到的法律文本，用中文回答问题。

检索文本：
{context}

问题：{query}

答案："""
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response['message']['content']
        elapsed = timer() - start
        
        # Metrics
        cr = calculate_context_relevance(query, retrieved_chunks, scores.tolist())
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
    
    status_text.text(f"✅ {config_name} completed!")
    
    # Calculate averages
    avg = {
        'context_relevance': sum(r['context_relevance'] for r in results) / len(results),
        'answer_relevance': sum(r['answer_relevance'] for r in results) / len(results),
        'faithfulness': sum(r['faithfulness'] for r in results) / len(results),
        'overall': sum(r['overall'] for r in results) / len(results),
        'time': sum(r['time'] for r in results) / len(results)
    }
    
    return {'config': config_name, 'results': results, 'averages': avg}

# Run comparison button
if st.button("🚀 Run Comparison Test", type="primary", use_container_width=True):
    
    st.markdown("---")
    st.subheader("🧪 Running Tests...")
    
    # Create progress indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔵 BASELINE")
        baseline_progress = st.progress(0)
        baseline_status = st.empty()
    
    with col2:
        st.markdown("#### 🟢 IMPROVED")
        improved_progress = st.progress(0)
        improved_status = st.empty()
    
    # Run tests
    with st.spinner("Running baseline configuration..."):
        baseline_data = run_pipeline("BASELINE", selected_queries, baseline_progress, baseline_status)
    
    with st.spinner("Running improved configuration..."):
        improved_data = run_pipeline("IMPROVED", selected_queries, improved_progress, improved_status)
    
    # Store results in session state
    st.session_state.baseline_data = baseline_data
    st.session_state.improved_data = improved_data
    st.session_state.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.success("✅ Tests completed!")
    st.rerun()

# Display results if available
if 'baseline_data' in st.session_state and 'improved_data' in st.session_state:
    
    baseline_data = st.session_state.baseline_data
    improved_data = st.session_state.improved_data
    
    st.markdown("---")
    st.header("📊 Comparison Results")
    
    # Summary metrics
    b_avg = baseline_data['averages']
    i_avg = improved_data['averages']
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Metric': ['Context Relevance', 'Answer Relevance', 'Faithfulness', 'Overall Score', 'Avg Time (s)'],
        'Baseline': [
            f"{b_avg['context_relevance']:.3f}",
            f"{b_avg['answer_relevance']:.3f}",
            f"{b_avg['faithfulness']:.3f}",
            f"{b_avg['overall']:.3f}",
            f"{b_avg['time']:.1f}"
        ],
        'Improved': [
            f"{i_avg['context_relevance']:.3f}",
            f"{i_avg['answer_relevance']:.3f}",
            f"{i_avg['faithfulness']:.3f}",
            f"{i_avg['overall']:.3f}",
            f"{i_avg['time']:.1f}"
        ],
        'Δ': [
            f"{i_avg['context_relevance'] - b_avg['context_relevance']:+.3f}",
            f"{i_avg['answer_relevance'] - b_avg['answer_relevance']:+.3f}",
            f"{i_avg['faithfulness'] - b_avg['faithfulness']:+.3f}",
            f"{i_avg['overall'] - b_avg['overall']:+.3f}",
            f"{i_avg['time'] - b_avg['time']:+.1f}"
        ],
        'Δ%': [
            f"{((i_avg['context_relevance'] - b_avg['context_relevance']) / b_avg['context_relevance'] * 100):+.1f}%",
            f"{((i_avg['answer_relevance'] - b_avg['answer_relevance']) / b_avg['answer_relevance'] * 100):+.1f}%",
            f"{((i_avg['faithfulness'] - b_avg['faithfulness']) / b_avg['faithfulness'] * 100):+.1f}%",
            f"{((i_avg['overall'] - b_avg['overall']) / b_avg['overall'] * 100):+.1f}%",
            f"{((i_avg['time'] - b_avg['time']) / b_avg['time'] * 100):+.1f}%"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Overall improvement
    overall_improvement = ((i_avg['overall'] - b_avg['overall']) / b_avg['overall'] * 100)
    
    if overall_improvement > 0:
        st.success(f"✅ **Improved version is {overall_improvement:.1f}% better overall!**")
    else:
        st.warning(f"⚠️ Improved version showed {overall_improvement:.1f}% change")
    
    # Bar chart comparison
    st.subheader("📈 Metrics Comparison")
    
    metrics_data = pd.DataFrame({
        'Metric': ['Context\nRelevance', 'Answer\nRelevance', 'Faithfulness', 'Overall\nScore'],
        'Baseline': [b_avg['context_relevance'], b_avg['answer_relevance'], 
                     b_avg['faithfulness'], b_avg['overall']],
        'Improved': [i_avg['context_relevance'], i_avg['answer_relevance'], 
                     i_avg['faithfulness'], i_avg['overall']]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Baseline',
        x=metrics_data['Metric'],
        y=metrics_data['Baseline'],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='Improved',
        x=metrics_data['Metric'],
        y=metrics_data['Improved'],
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        barmode='group',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Query-by-query results
    st.subheader("📝 Query-by-Query Results")
    
    query_results = []
    for i, (b_res, i_res) in enumerate(zip(baseline_data['results'], improved_data['results']), 1):
        query_results.append({
            'Query #': i,
            'Query': b_res['query'][:60] + '...' if len(b_res['query']) > 60 else b_res['query'],
            'Baseline Overall': f"{b_res['overall']:.3f}",
            'Improved Overall': f"{i_res['overall']:.3f}",
            'Δ': f"{i_res['overall'] - b_res['overall']:+.3f}",
            'Winner': '🟢 Improved' if i_res['overall'] > b_res['overall'] else ('🔵 Baseline' if b_res['overall'] > i_res['overall'] else '⚪ Tie')
        })
    
    query_df = pd.DataFrame(query_results)
    st.dataframe(query_df, use_container_width=True, hide_index=True)
    
    # Detailed query view
    st.subheader("🔍 Detailed Query Analysis")
    
    selected_query_idx = st.selectbox(
        "Select query to view details:",
        range(len(selected_queries)),
        format_func=lambda x: f"Query {x+1}: {selected_queries[x][:60]}..."
    )
    
    b_result = baseline_data['results'][selected_query_idx]
    i_result = improved_data['results'][selected_query_idx]
    
    st.markdown(f"**Query:** {selected_queries[selected_query_idx]}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🔵 Baseline Answer")
        st.info(b_result['answer'])
        st.markdown(f"""
        - Context Relevance: {b_result['context_relevance']:.3f}
        - Answer Relevance: {b_result['answer_relevance']:.3f}
        - Faithfulness: {b_result['faithfulness']:.3f}
        - **Overall: {b_result['overall']:.3f}**
        - Time: {b_result['time']:.1f}s
        """)
    
    with col2:
        st.markdown("##### 🟢 Improved Answer")
        st.success(i_result['answer'])
        st.markdown(f"""
        - Context Relevance: {i_result['context_relevance']:.3f}
        - Answer Relevance: {i_result['answer_relevance']:.3f}
        - Faithfulness: {i_result['faithfulness']:.3f}
        - **Overall: {i_result['overall']:.3f}**
        - Time: {i_result['time']:.1f}s
        """)
    
    # Download results
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        report = {
            'timestamp': st.session_state.test_timestamp,
            'test_info': {
                'num_queries': len(selected_queries),
                'pdf': PDF_PATH,
                'model': OLLAMA_MODEL
            },
            'baseline': baseline_data,
            'improved': improved_data
        }
        
        json_str = json.dumps(report, ensure_ascii=False, indent=2)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_str,
            file_name=f"ui_comparison_{st.session_state.test_timestamp}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV download
        csv_str = comparison_df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV Summary",
            data=csv_str,
            file_name=f"ui_comparison_{st.session_state.test_timestamp}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<small>
📚 Testing configurations from quick_ablation.py | 
🔵 Baseline = BASELINE config | 
🟢 Improved = Test 5 config
</small>
</div>
""", unsafe_allow_html=True)
