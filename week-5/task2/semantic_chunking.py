"""
Semantic Chunking Implementation for RAG
========================================

This module implements semantic chunking based on sentence embedding similarity.
It automatically detects topic boundaries and creates variable-size chunks.

Key Features:
- Sentence-level semantic similarity detection
- Configurable similarity threshold
- Min/max chunk size constraints
- Works with any sentence embeddings

Usage:
    from semantic_chunking import semantic_chunk
    
    chunks = semantic_chunk(
        sentences,
        model,
        similarity_threshold=0.75,
        max_chunk_size=15,
        min_chunk_size=3
    )

Author: Week 5 Task 2
Date: 2024
"""

import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_chunk(
    sentences: List[str],
    model: SentenceTransformer,
    similarity_threshold: float = 0.75,
    max_chunk_size: int = 15,
    min_chunk_size: int = 3,
    overlap: int = 0,
    verbose: bool = False
) -> Tuple[List[str], List[dict]]:
    """
    Create semantically coherent chunks based on sentence similarity.
    
    Algorithm:
    1. Encode all sentences into embeddings
    2. Calculate cosine similarity between adjacent sentences
    3. Detect topic boundaries where similarity < threshold
    4. Create chunks respecting min/max size constraints
    5. Add overlap sentences from previous chunk (if specified)
    
    Args:
        sentences: List of sentences to chunk
        model: SentenceTransformer model for encoding
        similarity_threshold: Similarity below this triggers new chunk (0.0-1.0)
        max_chunk_size: Maximum sentences per chunk
        min_chunk_size: Minimum sentences per chunk
        overlap: Number of sentences to overlap between chunks
        verbose: Print debug information
    
    Returns:
        chunks: List of text chunks (joined sentences)
        stats: List of chunk metadata (size, avg_similarity, boundary_indices)
    """
    if not sentences or len(sentences) == 0:
        return [], []
    
    if len(sentences) <= min_chunk_size:
        # Too few sentences, return as single chunk
        return [" ".join(sentences)], [{
            'size': len(sentences),
            'avg_similarity': 1.0,
            'boundary_indices': [0, len(sentences)]
        }]
    
    # Step 1: Encode all sentences
    if verbose:
        print(f"Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, show_progress_bar=verbose)
    
    # Step 2: Calculate similarities between adjacent sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
    
    if verbose:
        print(f"Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
        print(f"Similarity mean: {np.mean(similarities):.3f}")
    
    # Step 3: Detect boundaries where similarity drops below threshold
    boundary_indices = [0]  # Start of first chunk
    current_chunk_size = 1
    
    for i, sim in enumerate(similarities):
        current_chunk_size += 1
        
        # Check if we should start a new chunk
        should_split = False
        
        # Condition 1: Max size exceeded (enforce hard limit)
        if current_chunk_size >= max_chunk_size:
            should_split = True
            if verbose:
                print(f"Boundary at {i+1}: max size {max_chunk_size} reached")
        # Condition 2: Semantic boundary (low similarity) AND meets min size
        elif sim < similarity_threshold and current_chunk_size >= min_chunk_size:
            should_split = True
            if verbose:
                print(f"Boundary at {i+1}: similarity {sim:.3f} < threshold {similarity_threshold}")
        
        # Apply split
        if should_split:
            boundary_indices.append(i + 1)
            current_chunk_size = 1
    
    # Add end boundary (last chunk must meet min_chunk_size)
    if len(sentences) - boundary_indices[-1] >= min_chunk_size:
        boundary_indices.append(len(sentences))
    else:
        # Merge last small chunk with previous one
        boundary_indices[-1] = len(sentences)
    
    # Step 4: Create chunks with overlap
    chunks = []
    stats = []
    
    for i in range(len(boundary_indices) - 1):
        start_idx = boundary_indices[i]
        end_idx = boundary_indices[i + 1]
        
        # Apply overlap: extend start backward by overlap sentences
        if overlap > 0 and i > 0:
            overlap_start = max(0, start_idx - overlap)
            chunk_sentences = sentences[overlap_start:end_idx]
            actual_start = overlap_start
        else:
            chunk_sentences = sentences[start_idx:end_idx]
            actual_start = start_idx
        
        # Join sentences in chunk
        chunk_text = " ".join(chunk_sentences)
        chunks.append(chunk_text)
        
        # Calculate average similarity within chunk (excluding overlap)
        if end_idx - start_idx > 1:
            chunk_sims = similarities[start_idx:end_idx - 1]
            avg_sim = np.mean(chunk_sims) if chunk_sims else 1.0
        else:
            avg_sim = 1.0
        
        stats.append({
            'size': len(chunk_sentences),  # Total size including overlap
            'core_size': end_idx - start_idx,  # Size without overlap
            'avg_similarity': float(avg_sim),
            'boundary_indices': [actual_start, end_idx]
        })
    
    if verbose:
        print(f"\nCreated {len(chunks)} chunks")
        print(f"Chunk sizes: {[s['size'] for s in stats]}")
        avg_sims_str = [f"{s['avg_similarity']:.3f}" for s in stats]
        print(f"Avg similarities: {avg_sims_str}")
    
    return chunks, stats


def analyze_chunking(stats: List[dict]) -> dict:
    """
    Analyze chunking quality metrics.
    
    Args:
        stats: Chunk statistics from semantic_chunk()
    
    Returns:
        metrics: Dictionary with quality metrics
    """
    sizes = [s['size'] for s in stats]
    sims = [s['avg_similarity'] for s in stats]
    
    metrics = {
        'num_chunks': len(stats),
        'avg_chunk_size': np.mean(sizes),
        'chunk_size_std': np.std(sizes),
        'min_chunk_size': min(sizes),
        'max_chunk_size': max(sizes),
        'avg_within_chunk_similarity': np.mean(sims),
        'within_chunk_similarity_std': np.std(sims)
    }
    
    return metrics


def preview_chunking(
    text: str,
    model: SentenceTransformer,
    sentence_splitter,
    similarity_threshold: float = 0.75,
    max_preview_chars: int = 100
) -> None:
    """
    Preview semantic chunking on a text sample.
    
    Args:
        text: Input text to chunk
        model: SentenceTransformer model
        sentence_splitter: Function to split text into sentences
        similarity_threshold: Similarity threshold for chunking
        max_preview_chars: Max characters to show per chunk preview
    """
    # Split into sentences
    sentences = sentence_splitter(text)
    print(f"Total sentences: {len(sentences)}\n")
    
    # Perform chunking
    chunks, stats = semantic_chunk(
        sentences, model, 
        similarity_threshold=similarity_threshold,
        verbose=True
    )
    
    # Analyze
    metrics = analyze_chunking(stats)
    print("\n" + "="*60)
    print("CHUNKING METRICS")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Preview chunks
    print("\n" + "="*60)
    print("CHUNK PREVIEWS")
    print("="*60)
    for i, (chunk, stat) in enumerate(zip(chunks, stats)):
        preview = chunk[:max_preview_chars] + "..." if len(chunk) > max_preview_chars else chunk
        print(f"\nChunk {i+1}:")
        print(f"  Size: {stat['size']} sentences")
        print(f"  Avg similarity: {stat['avg_similarity']:.3f}")
        print(f"  Preview: {preview}")


if __name__ == "__main__":
    """
    Test semantic chunking on a sample document.
    """
    import sys
    sys.path.append('..')
    
    from util.pdf_utils import open_and_read_pdf
    from sentence_transformers import SentenceTransformer
    
    # Simple Chinese sentence splitter
    def chinese_sentence_splitter(text: str):
        """Split text into Chinese sentences."""
        chinese_delimiters = ['。', '！', '？', '；', '…']
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in chinese_delimiters:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    # Load model
    print("Loading embedding model...")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    # Load sample pages
    print("Loading PDF...")
    pdf_path = "../lawbook_sample.pdf"
    pages_and_texts = open_and_read_pdf(pdf_path)
    # Use first 3 pages
    text = " ".join([page["text"] for page in pages_and_texts[:3]])
    
    # Preview chunking with different thresholds
    for threshold in [0.7, 0.75, 0.8, 0.85]:
        print("\n" + "="*80)
        print(f"TESTING THRESHOLD: {threshold}")
        print("="*80)
        preview_chunking(
            text, model, chinese_sentence_splitter,
            similarity_threshold=threshold
        )
        print("\n")
