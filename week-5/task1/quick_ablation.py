"""
Quick Ablation Test - Fast validation with fewer queries and configs
Use this to quickly validate the experimental design before running full test
"""

import sys
import os
sys.path.append('..')  # Add parent directory to path

import pandas as pd
import torch
import ollama
from sentence_transformers import SentenceTransformer
from time import perf_counter as timer
import json
from datetime import datetime
import csv

from util import pdf_utils
from util.embedings_utils import embed_chunks
from util.nlp_utils import chunks_to_text_elems
from util.vector_search_utils import retrieve_relevant_resources
from util.evaluation_utils import (
    calculate_context_relevance,
    calculate_answer_relevance,
    calculate_faithfulness
)

# Load professional queries from file
with open("../test_queries_professional.txt", "r", encoding="utf-8") as f:
    QUERIES = [line.strip() for line in f if line.strip() and not line.startswith("#")]

PDF_PATH = "../lawbook_sample.pdf"  # Small sample (50 pages)
OLLAMA_MODEL = "llama3"

def sentencize_with_method(pages_and_texts, method):
    """Sentencize using specified method."""
    if method == "spacy":
        # Use English spacy (baseline - not ideal for Chinese but that's the original)
        import spacy
        nlp = spacy.load("en_core_web_sm")
        for item in pages_and_texts:
            doc = nlp(item["text"])
            item["sentences"] = [str(sent) for sent in doc.sents]
        return pages_and_texts
    elif method == "jieba":
        # Use jieba for Chinese (improvement)
        from util.nlp_utils import sentencize_jieba
        return sentencize_jieba(pages_and_texts)
    return pages_and_texts

def create_chunks_with_params(pages_with_sentences, chunk_size, overlap):
    """Create chunks with specified parameters."""
    from util.nlp_utils import split_list_overlapping, split_list
    
    for item in pages_with_sentences:
        if overlap == 0:
            item["sentence_chunks"] = split_list(item["sentences"], chunk_size)
        else:
            item["sentence_chunks"] = split_list_overlapping(item["sentences"], chunk_size, overlap)
    
    return pages_with_sentences

def test_configuration(config_name, embedding_model_name, segmentation_method, 
                      chunk_size, overlap, n_results, use_hybrid, hybrid_boost=0.3):
    """Test a specific configuration."""
    
    print(f"\n{'='*70}")
    print(f"🧪 {config_name}")
    print(f"{'='*70}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load PDF
        pages_and_texts = pdf_utils.open_and_read_pdf(PDF_PATH)
        print(f"  📄 Loaded {len(pages_and_texts)} pages")
        
        # Sentencize
        pages_with_sentences = sentencize_with_method(pages_and_texts, segmentation_method)
        
        # Chunk
        pages_with_chunks = create_chunks_with_params(pages_with_sentences, chunk_size, overlap)
        
        # Convert to text elements
        chunks = chunks_to_text_elems(pages_with_chunks)
        print(f"  📦 Created {len(chunks)} chunks")
        
        # Load embedding model
        print(f"  🔧 Loading {embedding_model_name.split('/')[-1]}...", end=" ")
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
        print("✅")
        
        # Create embeddings
        print(f"  🔢 Creating embeddings...", end=" ")
        embed_chunks(chunks, embedding_model)  # Modifies chunks in-place
        
        # Extract embeddings
        embeddings_list = [chunk['embedding'] for chunk in chunks]
        embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32).to(device)
        
        # Create dataframe for retrieval
        df = pd.DataFrame(chunks)
        print("✅")
        
        # Test all queries
        results = []
        
        for i, query in enumerate(QUERIES, 1):
            print(f"  [{i}/{len(QUERIES)}] {query[:35]}...", end=" ")
            
            start = timer()
            
            # Retrieve (model will encode query internally)
            scores, indices = retrieve_relevant_resources(
                query=query,
                embeddings=embeddings_tensor,
                model=embedding_model,
                st=None,
                n_resources_to_return=n_results,
                print_time=False,
                pages_and_chunks=chunks if use_hybrid else None,
                use_hybrid=use_hybrid
            )
            
            # Generate
            retrieved_chunks = [chunks[idx]['sentence_chunk'] for idx in indices]
            context = "\n\n".join(retrieved_chunks)
            scores_list = scores.tolist()
            
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
            
            # Calculate metrics
            cr = calculate_context_relevance(query, retrieved_chunks, scores_list)
            ar = calculate_answer_relevance(query, answer)
            f = calculate_faithfulness(answer, retrieved_chunks)
            overall = (cr + ar + f) / 3
            
            print(f"CR:{cr:.2f} AR:{ar:.2f} F:{f:.2f} O:{overall:.2f} ({elapsed:.1f}s)")
            
            # Store detailed results
            results.append({
                'query': query,
                'retrieved_chunks': retrieved_chunks,
                'retrieval_scores': scores_list,
                'answer': answer,
                'context_relevance': cr,
                'answer_relevance': ar,
                'faithfulness': f,
                'overall': overall,
                'time': elapsed
            })
        
        # Calculate averages
        avg = {
            'context_relevance': sum(r['context_relevance'] for r in results) / len(results),
            'answer_relevance': sum(r['answer_relevance'] for r in results) / len(results),
            'faithfulness': sum(r['faithfulness'] for r in results) / len(results),
            'overall': sum(r['overall'] for r in results) / len(results),
            'time': sum(r['time'] for r in results) / len(results)
        }
        
        print(f"\n  📊 AVG: CR:{avg['context_relevance']:.3f} AR:{avg['answer_relevance']:.3f} F:{avg['faithfulness']:.3f} Overall:{avg['overall']:.3f}")
        
        return {
            'success': True,
            'metrics': avg,
            'detailed_results': results  # Add detailed results
        }
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def save_results(all_results, baseline_metrics):
    """Save results to JSON and CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create RESULT directory if it doesn't exist
    os.makedirs('RESULT', exist_ok=True)
    
    # Save detailed JSON
    json_file = f"RESULT/ablation_detailed_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Detailed results saved to: {json_file}")
    
    # Save CSV summary
    csv_file = f"RESULT/ablation_summary_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Configuration', 'Embedding', 'Segmentation', 'Overlap', 'Hybrid',
            'Context_Relevance', 'Answer_Relevance', 'Faithfulness', 'Overall', 'Delta_%'
        ])
        
        for config_name, result in all_results.items():
            if result['success']:
                metrics = result['metrics']
                config = result['config']
                delta = ((metrics['overall'] - baseline_metrics['overall']) / baseline_metrics['overall'] * 100)
                
                writer.writerow([
                    config_name,
                    config['embedding'].split('/')[-1],
                    config['segmentation'],
                    config['overlap'],
                    config['use_hybrid'],
                    f"{metrics['context_relevance']:.3f}",
                    f"{metrics['answer_relevance']:.3f}",
                    f"{metrics['faithfulness']:.3f}",
                    f"{metrics['overall']:.3f}",
                    f"{delta:+.1f}%"
                ])
    
    print(f"📊 Summary CSV saved to: {csv_file}")
    return json_file, csv_file

if __name__ == "__main__":
    print("\n" + "="*70)
    print("⚡ QUICK ABLATION TEST (10 professional queries, 7 configs)")
    print("="*70)
    print(f"📄 Using: {PDF_PATH}")
    print(f"❓ Queries: {len(QUERIES)}")
    print("="*70 + "\n")
    
    # Test configurations - Baseline uses original settings, Improved tests each change
    configs = [
        # BASELINE - Original (spacy + multilingual embedding, no improvements)
        {
            'name': 'BASELINE',
            'embedding': 'sentence-transformers/all-mpnet-base-v2',
            'segmentation': 'spacy',  # Original (English spacy, not ideal for Chinese)
            'chunk_size': 10,
            'overlap': 0,
            'n_results': 10,
            'use_hybrid': False
        },
        
        # Test 1: Only change to Chinese embedding
        {
            'name': 'Test 1: Chinese Embedding',
            'embedding': 'BAAI/bge-large-zh-v1.5',
            'segmentation': 'spacy',
            'chunk_size': 10,
            'overlap': 0,
            'n_results': 10,
            'use_hybrid': False
        },
        
        # Test 2: Only change to jieba segmentation
        {
            'name': 'Test 2: Jieba Segmentation',
            'embedding': 'sentence-transformers/all-mpnet-base-v2',
            'segmentation': 'jieba',
            'chunk_size': 10,
            'overlap': 0,
            'n_results': 10,
            'use_hybrid': False
        },
        
        # Test 2.5: Jieba + Overlap (without Chinese embedding)
        {
            'name': 'Test 2.5: Jieba + Overlap',
            'embedding': 'sentence-transformers/all-mpnet-base-v2',
            'segmentation': 'jieba',
            'chunk_size': 10,
            'overlap': 2,
            'n_results': 10,
            'use_hybrid': False
        },
        
        # Test 3: Chinese embedding + jieba
        {
            'name': 'Test 3: Chinese Emb + Jieba',
            'embedding': 'BAAI/bge-large-zh-v1.5',
            'segmentation': 'jieba',
            'chunk_size': 10,
            'overlap': 0,
            'n_results': 10,
            'use_hybrid': False
        },
        
        # Test 4: Add overlap to Test 3
        {
            'name': 'Test 4: + Overlap',
            'embedding': 'BAAI/bge-large-zh-v1.5',
            'segmentation': 'jieba',
            'chunk_size': 10,
            'overlap': 2,
            'n_results': 10,
            'use_hybrid': False
        },
        
        # Test 5: ALL improvements
        {
            'name': 'Test 5: ALL IMPROVEMENTS',
            'embedding': 'BAAI/bge-large-zh-v1.5',
            'segmentation': 'jieba',
            'chunk_size': 10,
            'overlap': 2,
            'n_results': 10,
            'use_hybrid': True
        },
    ]
    
    # Run all tests
    all_results = {}
    baseline_metrics = None
    
    for i, config in enumerate(configs, 1):
        print(f"\n>>> [{i}/{len(configs)}]")
        result = test_configuration(
            config['name'],
            config['embedding'],
            config['segmentation'],
            config['chunk_size'],
            config['overlap'],
            config['n_results'],
            config['use_hybrid']
        )
        
        if result['success']:
            # Store config with result
            result['config'] = config
            all_results[config['name']] = result
            
            # Save baseline metrics for comparison
            if config['name'] == 'BASELINE':
                baseline_metrics = result['metrics']
    
    # Analysis
    print(f"\n\n{'='*70}")
    print("📊 QUICK RESULTS SUMMARY")
    print("="*70 + "\n")
    
    if 'BASELINE' in all_results and baseline_metrics:
        baseline = baseline_metrics
        
        print(f"📍 BASELINE: Overall={baseline['overall']:.3f} | CR={baseline['context_relevance']:.3f} AR={baseline['answer_relevance']:.3f} F={baseline['faithfulness']:.3f}\n")
        
        print(f"{'Configuration':<30} {'Overall':>8} {'Δ%':>8} {'CR':>6} {'AR':>6} {'F':>6} {'Status'}")
        print("-" * 70)
        
        for name, result in all_results.items():
            if name == 'BASELINE':
                continue
            
            metrics = result['metrics']
            o_diff = ((metrics['overall'] - baseline['overall']) / baseline['overall']) * 100
            cr_better = metrics['context_relevance'] > baseline['context_relevance']
            ar_better = metrics['answer_relevance'] > baseline['answer_relevance']
            f_better = metrics['faithfulness'] > baseline['faithfulness']
            o_better = metrics['overall'] > baseline['overall']
            
            all_better = cr_better and ar_better and f_better and o_better
            status = "✅ ALL BETTER" if all_better else ""
            
            print(f"{name:<30} {metrics['overall']:>8.3f} {o_diff:>7.1f}% {metrics['context_relevance']:>6.3f} {metrics['answer_relevance']:>6.3f} {metrics['faithfulness']:>6.3f} {status}")
        
        print("\n" + "="*70)
        
        # Save results to files
        json_file, csv_file = save_results(all_results, baseline_metrics)
        
        # Check if any config wins on all metrics
        winners = [name for name, result in all_results.items() 
                  if name != 'BASELINE' and
                  result['metrics']['context_relevance'] > baseline['context_relevance'] and
                  result['metrics']['answer_relevance'] > baseline['answer_relevance'] and
                  result['metrics']['faithfulness'] > baseline['faithfulness'] and
                  result['metrics']['overall'] > baseline['overall']]
        
        if winners:
            print(f"\n🎉 SUCCESS! {len(winners)} configuration(s) beat baseline on ALL metrics:")
            for w in winners:
                print(f"   ✅ {w}")
            print(f"\n💡 You can now run full ablation_study.py with all 8 configs and 5 queries")
        else:
            print(f"\n⚠️  No config beats baseline on ALL metrics yet.")
            print(f"   Try adjusting parameters or run full study for more data points")
