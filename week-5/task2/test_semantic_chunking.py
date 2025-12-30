"""
Task 2: Semantic Chunking Experiment
=====================================

Compare two chunking strategies:
1. BASELINE: Fixed-size (10 sentences, 2 overlap) - Task 1 Test 5
2. SEMANTIC: Variable-size based on semantic similarity

Test on same 10 professional queries as Task 1.
"""

import sys
sys.path.append('..')

import torch
import ollama
import json
import pandas as pd
from datetime import datetime
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer

from util.pdf_utils import open_and_read_pdf
from util.nlp_utils import sentencize_jieba, split_list_overlapping, chunks_to_text_elems
from util.embedings_utils import embed_chunks
from util.vector_search_utils import retrieve_relevant_resources
import numpy as np
from util.evaluation_utils import (
    calculate_context_relevance,
    calculate_answer_relevance,
    calculate_faithfulness
)
from semantic_chunking import semantic_chunk, analyze_chunking

# Configuration
PDF_PATH = "../lawbook_sample.pdf"
QUERIES_PATH = "../test_queries_professional.txt"
OLLAMA_MODEL = "llama3"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
NUM_RESULTS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Semantic chunking parameters (adjusted to match baseline chunk count ~108)
SIMILARITY_THRESHOLD = 0.50  # Lower = larger chunks
MAX_CHUNK_SIZE = 10  # Match baseline
MIN_CHUNK_SIZE = 7  # Closer to baseline's 10
OVERLAP = 2  # Match baseline's overlap

print("="*80)
print("TASK 2: SEMANTIC CHUNKING EXPERIMENT")
print("="*80)
print(f"Device: {DEVICE}")
print(f"PDF: {PDF_PATH}")
print(f"Embedding Model: {EMBEDDING_MODEL}")
print(f"LLM: {OLLAMA_MODEL}")
print(f"Num Results: {NUM_RESULTS}")
print(f"\nSemantic Chunking Config:")
print(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}")
print(f"  Max Chunk Size: {MAX_CHUNK_SIZE} sentences")
print(f"  Min Chunk Size: {MIN_CHUNK_SIZE} sentences")
print()

# Load queries
print("Loading test queries...")
with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
print(f"Loaded {len(queries)} queries\n")

# Load and preprocess PDF
print("Loading PDF...")
pages_and_texts = open_and_read_pdf(PDF_PATH)
print(f"Loaded {len(pages_and_texts)} pages\n")

# Sentencize (same for both)
print("Segmenting text with jieba...")
pages_and_texts = sentencize_jieba(pages_and_texts)
total_sentences = sum(len(p["sentences"]) for p in pages_and_texts)
print(f"Total sentences: {total_sentences}\n")

# Load embedding model
print(f"Loading embedding model ({EMBEDDING_MODEL})...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
print()

def format_rag_prompt(query: str, context_items: list[dict]) -> str:
    """Format prompt with retrieved context (same as Web Compare)"""
    # Use same format as compare_ui_versions_web_task1.py for consistency
    context = "\n\n".join([item['sentence_chunk'] for item in context_items])
    
    prompt = f"""根据以下检索到的法律文本，用中文回答问题。

检索文本：
{context}

问题：{query}

答案："""
    
    return prompt

def generate_answer_ollama(model_name: str, prompt: str) -> str:
    """Generate answer using Ollama chat API (same as Web Compare)"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7}
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def run_rag_pipeline(chunks, embeddings_tensor, query):
    """Run RAG pipeline: retrieve + generate + evaluate"""
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
    
    retrieved_chunks = [chunks[i] for i in indices]
    
    # Generate
    prompt = format_rag_prompt(query, retrieved_chunks)
    answer = generate_answer_ollama(OLLAMA_MODEL, prompt)
    
    # Evaluate
    context_relevance = calculate_context_relevance(
        query, 
        [c['sentence_chunk'] for c in retrieved_chunks],
        scores_list
    )
    answer_relevance = calculate_answer_relevance(query, answer)
    faithfulness = calculate_faithfulness(
        answer,
        [c['sentence_chunk'] for c in retrieved_chunks]
    )
    overall = (context_relevance + answer_relevance + faithfulness) / 3
    
    return {
        'answer': answer,
        'context_relevance': context_relevance,
        'answer_relevance': answer_relevance,
        'faithfulness': faithfulness,
        'overall_score': overall,
        'retrieved_chunks': [c['sentence_chunk'][:100] + '...' for c in retrieved_chunks[:3]]
    }

# ============================================================================
# EXPERIMENT 1: BASELINE (Fixed-size chunking)
# ============================================================================
print("="*80)
print("EXPERIMENT 1: BASELINE - Fixed-size Chunking")
print("="*80)
print("Config: 10 sentences, 2 overlap, hybrid search")
print()

baseline_start = timer()

# Create baseline chunks
print("Creating baseline chunks...")
for item in pages_and_texts:
    item["sentence_chunks"] = split_list_overlapping(item["sentences"], 10, 2)

baseline_chunks = chunks_to_text_elems(pages_and_texts)
print(f"Total chunks: {len(baseline_chunks)}")
print(f"Avg chunk size: {sum(len(c['sentence_chunk'].split('。')) for c in baseline_chunks) / len(baseline_chunks):.2f} sentences")

# Embed chunks
print("Creating embeddings...")
embed_chunks(baseline_chunks, embedding_model)
# Convert embeddings to tensor
baseline_embeddings = torch.tensor(
    np.array([chunk['embedding'] for chunk in baseline_chunks]), 
    dtype=torch.float32
).to(DEVICE)
print()

# Run queries
baseline_results = []
print(f"Running {len(queries)} queries...")
for i, query in enumerate(queries, 1):
    print(f"  [{i}/{len(queries)}] {query[:60]}...")
    result = run_rag_pipeline(baseline_chunks, baseline_embeddings, query)
    result['query'] = query
    result['query_num'] = i
    baseline_results.append(result)

baseline_time = timer() - baseline_start

# Calculate average scores
baseline_avg = {
    'context_relevance': sum(r['context_relevance'] for r in baseline_results) / len(baseline_results),
    'answer_relevance': sum(r['answer_relevance'] for r in baseline_results) / len(baseline_results),
    'faithfulness': sum(r['faithfulness'] for r in baseline_results) / len(baseline_results),
    'overall_score': sum(r['overall_score'] for r in baseline_results) / len(baseline_results)
}

print(f"\nBaseline Results:")
print(f"  Context Relevance:  {baseline_avg['context_relevance']:.4f}")
print(f"  Answer Relevance:   {baseline_avg['answer_relevance']:.4f}")
print(f"  Faithfulness:       {baseline_avg['faithfulness']:.4f}")
print(f"  Overall Score:      {baseline_avg['overall_score']:.4f}")
print(f"  Time: {baseline_time:.2f}s")
print()

# ============================================================================
# EXPERIMENT 2: SEMANTIC (Variable-size chunking)
# ============================================================================
print("="*80)
print("EXPERIMENT 2: SEMANTIC - Variable-size Chunking")
print("="*80)
print(f"Config: Similarity threshold {SIMILARITY_THRESHOLD}, min={MIN_CHUNK_SIZE}, max={MAX_CHUNK_SIZE} sentences")
print()

semantic_start = timer()

# Create semantic chunks
print("Creating semantic chunks...")
semantic_chunks_list = []
all_chunk_stats = []

for page_idx, item in enumerate(pages_and_texts):
    page_chunks, page_stats = semantic_chunk(
        item["sentences"],
        embedding_model,
        similarity_threshold=SIMILARITY_THRESHOLD,
        max_chunk_size=MAX_CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE,
        overlap=OVERLAP,
        verbose=False
    )
    
    # Convert to format compatible with existing pipeline
    for chunk_idx, (chunk_text, chunk_stat) in enumerate(zip(page_chunks, page_stats)):
        semantic_chunks_list.append({
            'page_number': item['page_number'],
            'sentence_chunk': chunk_text,
            'chunk_char_count': len(chunk_text),
            'chunk_word_count': len(chunk_text.split()),
            'chunk_token_count': len(chunk_text) // 4,  # Rough estimate
            'chunk_size': chunk_stat['size'],  # Number of sentences
            'avg_similarity': chunk_stat['avg_similarity']
        })
    
    all_chunk_stats.extend(page_stats)

# Analyze chunking quality
chunking_metrics = analyze_chunking(all_chunk_stats)
print(f"Total chunks: {len(semantic_chunks_list)}")
print(f"Avg chunk size: {chunking_metrics['avg_chunk_size']:.2f} sentences")
print(f"Chunk size std: {chunking_metrics['chunk_size_std']:.2f}")
print(f"Min/Max size: {chunking_metrics['min_chunk_size']}/{chunking_metrics['max_chunk_size']} sentences")
print(f"Avg within-chunk similarity: {chunking_metrics['avg_within_chunk_similarity']:.3f}")

# Embed chunks
print("Creating embeddings...")
embed_chunks(semantic_chunks_list, embedding_model)
# Convert embeddings to tensor
semantic_embeddings = torch.tensor(
    np.array([chunk['embedding'] for chunk in semantic_chunks_list]), 
    dtype=torch.float32
).to(DEVICE)
print()

# Run queries
semantic_results = []
print(f"Running {len(queries)} queries...")
for i, query in enumerate(queries, 1):
    print(f"  [{i}/{len(queries)}] {query[:60]}...")
    result = run_rag_pipeline(semantic_chunks_list, semantic_embeddings, query)
    result['query'] = query
    result['query_num'] = i
    semantic_results.append(result)

semantic_time = timer() - semantic_start

# Calculate average scores
semantic_avg = {
    'context_relevance': sum(r['context_relevance'] for r in semantic_results) / len(semantic_results),
    'answer_relevance': sum(r['answer_relevance'] for r in semantic_results) / len(semantic_results),
    'faithfulness': sum(r['faithfulness'] for r in semantic_results) / len(semantic_results),
    'overall_score': sum(r['overall_score'] for r in semantic_results) / len(semantic_results)
}

print(f"\nSemantic Results:")
print(f"  Context Relevance:  {semantic_avg['context_relevance']:.4f}")
print(f"  Answer Relevance:   {semantic_avg['answer_relevance']:.4f}")
print(f"  Faithfulness:       {semantic_avg['faithfulness']:.4f}")
print(f"  Overall Score:      {semantic_avg['overall_score']:.4f}")
print(f"  Time: {semantic_time:.2f}s")
print()

# ============================================================================
# COMPARISON
# ============================================================================
print("="*80)
print("COMPARISON: SEMANTIC vs BASELINE")
print("="*80)

comparison = {
    'context_relevance': {
        'baseline': baseline_avg['context_relevance'],
        'semantic': semantic_avg['context_relevance'],
        'diff': semantic_avg['context_relevance'] - baseline_avg['context_relevance'],
        'pct_change': (semantic_avg['context_relevance'] - baseline_avg['context_relevance']) / baseline_avg['context_relevance'] * 100
    },
    'answer_relevance': {
        'baseline': baseline_avg['answer_relevance'],
        'semantic': semantic_avg['answer_relevance'],
        'diff': semantic_avg['answer_relevance'] - baseline_avg['answer_relevance'],
        'pct_change': (semantic_avg['answer_relevance'] - baseline_avg['answer_relevance']) / baseline_avg['answer_relevance'] * 100
    },
    'faithfulness': {
        'baseline': baseline_avg['faithfulness'],
        'semantic': semantic_avg['faithfulness'],
        'diff': semantic_avg['faithfulness'] - baseline_avg['faithfulness'],
        'pct_change': (semantic_avg['faithfulness'] - baseline_avg['faithfulness']) / baseline_avg['faithfulness'] * 100
    },
    'overall_score': {
        'baseline': baseline_avg['overall_score'],
        'semantic': semantic_avg['overall_score'],
        'diff': semantic_avg['overall_score'] - baseline_avg['overall_score'],
        'pct_change': (semantic_avg['overall_score'] - baseline_avg['overall_score']) / baseline_avg['overall_score'] * 100
    }
}

for metric, values in comparison.items():
    print(f"\n{metric.replace('_', ' ').title()}:")
    print(f"  Baseline:  {values['baseline']:.4f}")
    print(f"  Semantic:  {values['semantic']:.4f}")
    print(f"  Diff:      {values['diff']:+.4f} ({values['pct_change']:+.2f}%)")

print(f"\nChunking Efficiency:")
print(f"  Baseline chunks:  {len(baseline_chunks)}")
print(f"  Semantic chunks:  {len(semantic_chunks_list)}")
print(f"  Reduction:        {len(baseline_chunks) - len(semantic_chunks_list)} chunks ({(1 - len(semantic_chunks_list)/len(baseline_chunks))*100:.1f}%)")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Detailed results
detailed_results = {
    'metadata': {
        'timestamp': timestamp,
        'pdf_path': PDF_PATH,
        'queries_path': QUERIES_PATH,
        'embedding_model': EMBEDDING_MODEL,
        'llm_model': OLLAMA_MODEL,
        'num_queries': len(queries),
        'num_results': NUM_RESULTS,
        'device': DEVICE
    },
    'baseline_config': {
        'chunk_size': 10,
        'overlap': 2,
        'hybrid_search': True,
        'total_chunks': len(baseline_chunks),
        'execution_time': baseline_time
    },
    'semantic_config': {
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'max_chunk_size': MAX_CHUNK_SIZE,
        'min_chunk_size': MIN_CHUNK_SIZE,
        'hybrid_search': True,
        'total_chunks': len(semantic_chunks_list),
        'chunking_metrics': chunking_metrics,
        'execution_time': semantic_time
    },
    'baseline_results': baseline_results,
    'semantic_results': semantic_results,
    'baseline_average': baseline_avg,
    'semantic_average': semantic_avg,
    'comparison': comparison
}

json_path = f"RESULT/semantic_chunking_detailed_{timestamp}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(detailed_results, f, ensure_ascii=False, indent=2)
print(f"✅ Detailed results saved to: {json_path}")

# Summary CSV
summary_data = []
for metric in ['context_relevance', 'answer_relevance', 'faithfulness', 'overall_score']:
    summary_data.append({
        'metric': metric,
        'baseline': comparison[metric]['baseline'],
        'semantic': comparison[metric]['semantic'],
        'diff': comparison[metric]['diff'],
        'pct_change': comparison[metric]['pct_change']
    })

summary_df = pd.DataFrame(summary_data)
csv_path = f"RESULT/semantic_chunking_summary_{timestamp}.csv"
summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ Summary CSV saved to: {csv_path}")

# Markdown report
md_content = f"""# Task 2: Semantic Chunking Results

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration

### Baseline (Task 1 Test 5)
- Chunking: 10 sentences, 2 overlap
- Hybrid Search: ✅ Enabled
- Total Chunks: {len(baseline_chunks)}

### Semantic
- Similarity Threshold: {SIMILARITY_THRESHOLD}
- Min/Max Chunk Size: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} sentences
- Hybrid Search: ✅ Enabled
- Total Chunks: {len(semantic_chunks_list)}
- Avg Chunk Size: {chunking_metrics['avg_chunk_size']:.2f} ± {chunking_metrics['chunk_size_std']:.2f} sentences
- Avg Within-Chunk Similarity: {chunking_metrics['avg_within_chunk_similarity']:.3f}

## Results

| Metric | Baseline | Semantic | Diff | Change |
|--------|----------|----------|------|--------|
| Context Relevance | {baseline_avg['context_relevance']:.4f} | {semantic_avg['context_relevance']:.4f} | {comparison['context_relevance']['diff']:+.4f} | {comparison['context_relevance']['pct_change']:+.2f}% |
| Answer Relevance | {baseline_avg['answer_relevance']:.4f} | {semantic_avg['answer_relevance']:.4f} | {comparison['answer_relevance']['diff']:+.4f} | {comparison['answer_relevance']['pct_change']:+.2f}% |
| Faithfulness | {baseline_avg['faithfulness']:.4f} | {semantic_avg['faithfulness']:.4f} | {comparison['faithfulness']['diff']:+.4f} | {comparison['faithfulness']['pct_change']:+.2f}% |
| **Overall Score** | **{baseline_avg['overall_score']:.4f}** | **{semantic_avg['overall_score']:.4f}** | **{comparison['overall_score']['diff']:+.4f}** | **{comparison['overall_score']['pct_change']:+.2f}%** |

## Chunking Efficiency

- Baseline Chunks: {len(baseline_chunks)}
- Semantic Chunks: {len(semantic_chunks_list)}
- Reduction: {len(baseline_chunks) - len(semantic_chunks_list)} chunks ({(1 - len(semantic_chunks_list)/len(baseline_chunks))*100:.1f}%)

## Execution Time

- Baseline: {baseline_time:.2f}s
- Semantic: {semantic_time:.2f}s

## Conclusion

Semantic chunking {'improved' if semantic_avg['overall_score'] > baseline_avg['overall_score'] else 'did not improve'} overall RAG performance by {comparison['overall_score']['pct_change']:+.2f}% compared to fixed-size baseline.

---
*Generated by test_semantic_chunking.py*
"""

md_path = f"RESULT/semantic_chunking_report_{timestamp}.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"✅ Markdown report saved to: {md_path}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"\nResults saved to task2/RESULT/")
print(f"  - {json_path}")
print(f"  - {csv_path}")
print(f"  - {md_path}")
