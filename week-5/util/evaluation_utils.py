"""
Evaluation utilities for RAG system using RAGAS metrics.
"""
import streamlit as st
from typing import List, Dict
import numpy as np


def calculate_context_relevance(query: str, retrieved_chunks: List[str], scores: List[float]) -> float:
    """
    Calculate context relevance score based on retrieval scores.
    Higher scores indicate more relevant retrieved context.
    
    Args:
        query: The user query
        retrieved_chunks: List of retrieved text chunks
        scores: Similarity scores for each chunk
    
    Returns:
        float: Context relevance score (0-1)
    """
    if not scores:
        return 0.0
    
    # Normalize scores to 0-1 range
    scores_array = np.array(scores)
    
    # Use exponential weighting to emphasize top results
    weights = np.exp(np.linspace(0, -2, len(scores)))
    weighted_score = np.average(scores_array, weights=weights)
    
    return float(weighted_score)


def calculate_answer_relevance(query: str, answer: str) -> float:
    """
    Simplified answer relevance based on keyword overlap.
    In production, would use LLM-based evaluation.
    
    Args:
        query: The user query
        answer: Generated answer
    
    Returns:
        float: Answer relevance score (0-1)
    """
    if not answer or not query:
        return 0.0
    
    # Simple keyword matching (Chinese-aware)
    import jieba
    
    # Check if Chinese
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
    
    if is_chinese:
        query_tokens = set(jieba.cut(query))
        answer_tokens = set(jieba.cut(answer))
    else:
        query_tokens = set(query.lower().split())
        answer_tokens = set(answer.lower().split())
    
    # Remove single characters
    query_tokens = {t for t in query_tokens if len(t) > 1}
    answer_tokens = {t for t in answer_tokens if len(t) > 1}
    
    if not query_tokens:
        return 0.5
    
    # Calculate overlap
    overlap = len(query_tokens & answer_tokens)
    score = overlap / len(query_tokens)
    
    return min(score, 1.0)


def calculate_faithfulness(answer: str, retrieved_chunks: List[str]) -> float:
    """
    Calculate faithfulness - how much the answer is grounded in retrieved context.
    Simplified version based on content overlap.
    
    Args:
        answer: Generated answer
        retrieved_chunks: List of retrieved text chunks
    
    Returns:
        float: Faithfulness score (0-1)
    """
    if not answer or not retrieved_chunks:
        return 0.0
    
    import jieba
    
    # Check if Chinese
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in answer)
    
    if is_chinese:
        answer_tokens = set(jieba.cut(answer))
    else:
        answer_tokens = set(answer.lower().split())
    
    # Remove single characters and common words
    answer_tokens = {t for t in answer_tokens if len(t) > 1}
    
    # Count how many answer tokens appear in context
    context_text = " ".join(retrieved_chunks)
    grounded_count = sum(1 for token in answer_tokens if token in context_text)
    
    if not answer_tokens:
        return 0.5
    
    score = grounded_count / len(answer_tokens)
    return min(score, 1.0)


def display_evaluation_metrics(
    query: str,
    answer: str,
    retrieved_chunks: List[str],
    scores: List[float]
):
    """
    Display comprehensive evaluation metrics in Streamlit.
    
    Args:
        query: The user query
        answer: Generated answer
        retrieved_chunks: List of retrieved text chunks
        scores: Similarity scores for each chunk
    """
    st.subheader("📊 RAG Evaluation Metrics")
    
    # Calculate metrics
    context_rel = calculate_context_relevance(query, retrieved_chunks, scores)
    answer_rel = calculate_answer_relevance(query, answer)
    faithfulness = calculate_faithfulness(answer, retrieved_chunks)
    
    # Overall RAG score (weighted average)
    overall_score = (context_rel * 0.3 + answer_rel * 0.3 + faithfulness * 0.4)
    
    # Display in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Context Relevance",
            f"{context_rel:.2f}",
            help="How relevant are the retrieved chunks to the query? (0-1)"
        )
    
    with col2:
        st.metric(
            "Answer Relevance",
            f"{answer_rel:.2f}",
            help="How well does the answer address the query? (0-1)"
        )
    
    with col3:
        st.metric(
            "Faithfulness",
            f"{faithfulness:.2f}",
            help="How grounded is the answer in retrieved context? (0-1)"
        )
    
    with col4:
        # Color-code overall score
        if overall_score >= 0.7:
            delta_color = "normal"
            emoji = "✅"
        elif overall_score >= 0.5:
            delta_color = "off"
            emoji = "⚠️"
        else:
            delta_color = "inverse"
            emoji = "❌"
        
        st.metric(
            "Overall RAG Score",
            f"{overall_score:.2f} {emoji}",
            help="Weighted average of all metrics (0-1)"
        )
    
    # Detailed breakdown
    with st.expander("📈 Metric Details"):
        st.markdown("""
        **Context Relevance**: Measures how relevant the retrieved chunks are to the query.
        - Based on vector similarity scores with exponential weighting
        - Higher scores for top-ranked results
        
        **Answer Relevance**: Measures how well the answer addresses the query.
        - Based on keyword overlap between query and answer
        - Chinese-aware using jieba tokenization
        
        **Faithfulness**: Measures how grounded the answer is in the retrieved context.
        - Checks what percentage of answer content appears in retrieved chunks
        - Helps detect hallucinations
        
        **Overall Score**: Weighted average (Context: 30%, Answer: 30%, Faithfulness: 40%)
        - ✅ Excellent: ≥ 0.7
        - ⚠️ Good: 0.5-0.7
        - ❌ Needs Improvement: < 0.5
        """)
