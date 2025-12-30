# PDF RAG UI - Baseline Version (Chinese Support with Minimal Changes)
# Based on original pdf_rag_ui_ollama.py with minimal modifications:
# 1. Chinese prompt template (necessary for Chinese documents)
# 2. Fixed Chinese tokenization issue (no space splitting)  
# 3. Uses ORIGINAL chunk() function (10 sentences, no overlap)
# 4. No hybrid search
# 5. Retrieves 10 results (same as improved version for fair comparison)
# This serves as baseline for comparison with improved versions

import streamlit as st
import spacy
import re
import torch
from util import pdf_utils
from util.embedings_utils import embed_chunks, save_embeddings, embeddings_to_tensor
from util.nlp_utils import sentencize, chunk, chunks_to_text_elems
import pandas as pd
from util.session_utils import SESSION_VARS, put_to_session, get_from_session, print_session
from util.vector_search_utils import retrieve_relevant_resources
import ollama

# Requires !pip install sentence-transformers
from sentence_transformers import SentenceTransformer

EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Display GPU info in sidebar
if torch.cuda.is_available():
    st.sidebar.success(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
    st.sidebar.info(f"CUDA Version: {torch.version.cuda}")
    st.sidebar.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    st.sidebar.warning("⚠️ No GPU detected, using CPU")

min_token_length = 30

# Ollama model name
OLLAMA_MODEL = st.sidebar.text_input("Ollama Model Name", value="llama3", help="Name of the Ollama model to use")

st.write("Initializing models")

if not get_from_session(st, SESSION_VARS.LOADED_MODELS):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("sentencizer")
    put_to_session(st, SESSION_VARS.NLP, nlp)

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device=EMBED_DEVICE
    )

    if EMBED_DEVICE == "cuda":
        st.sidebar.success(f"✅ Embedding model loaded on GPU")
    else:
        st.sidebar.info(f"ℹ️ Embedding model loaded on CPU")
    
    put_to_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU, embedding_model)

    # Ollama client - verify model availability
    try:
        models = ollama.list()
        model_names = []
        for m in models.get("models", []):
            model_names.append(m.get("name") or m.get("model") or "")

        def matches(user_name: str, candidates: list[str]) -> bool:
            return any(c == user_name or c.startswith(user_name + ":") for c in candidates)

        if not matches(OLLAMA_MODEL, model_names):
            st.warning(f"Model '{OLLAMA_MODEL}' not found in Ollama. Available models: {', '.join(model_names)}")
            st.info(f"You can pull the model with: ollama pull {OLLAMA_MODEL}")
        else:
            st.success(f"Using Ollama model: {OLLAMA_MODEL}")
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        st.info("Make sure Ollama is running. You can start it with: ollama serve")

    put_to_session(st, SESSION_VARS.MODEL, OLLAMA_MODEL)
    st.write("Done")
    put_to_session(st, SESSION_VARS.LOADED_MODELS, True)
else:
    st.write("Models were already loaded")

print_session(st)

st.title('PDF RAG Demo - Baseline (Original Chunking)')
st.info("📌 Baseline version: Original chunk() function (10 sentences, no overlap), no hybrid search, 10 results")

query = st.text_input("Type your query here", "第2条说的'平等主体'是什么意思？为什么行政关系不属于民法调整？")
gen_variant = st.selectbox(
    "Select vanilla LLM or Retrieval Augmented LLM",
    ("vanilla", "rag")
)

uploaded_file = st.file_uploader(
    label="Upload a pdf",
    help="Upload a pdf file to chat to it with RAG",
    type='pdf'
)

def format_vanilla_prompt(query: str) -> str:
    """Format a simple query prompt for vanilla LLM."""
    return query

def format_rag_prompt(query: str, context_items: list[dict]) -> str:
    """
    Augments query with retrieved context items.
    Adapted for Chinese legal document queries.
    """
    # Join context items into a bulleted list
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create Chinese prompt template with legal domain examples
    base_prompt = """请根据以下检索到的法律条文内容回答用户的问题。
请仔细阅读相关条文，提取关键信息后再作答。
答案要准确、详细，并且必须基于提供的法律条文内容。
请用中文回答。
现在，请根据以下检索到的法律条文内容回答用户的问题：

{context}

用户问题：{query}

请用中文详细回答："""

    # Fill in the template with context and query
    prompt = base_prompt.format(context=context, query=query)
    return prompt

def generate_answer_ollama(model_name: str, prompt: str, max_tokens: int = 512) -> str:
    """Generate answer using Ollama API."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'num_predict': max_tokens,
                'temperature': 0.7,
            }
        )
        answer = response['message']['content']
        
        # Check if response might be truncated
        if answer and len(answer) > 50:
            answer_stripped = answer.rstrip()
            incomplete_patterns = [
                answer_stripped.endswith(','),
                answer_stripped.endswith(';'),
                bool(re.search(r'\d+\.\s*$', answer_stripped)),
                (not answer_stripped.endswith('.') and 
                 not answer_stripped.endswith('?') and 
                 not answer_stripped.endswith('!') and
                 len(answer_stripped) > 0 and
                 answer_stripped[-1].islower() and
                 len(answer_stripped.split()) > 10),
            ]
            if any(incomplete_patterns):
                answer += "\n\n[Note: Response may be truncated. Consider increasing Max Tokens in the sidebar.]"
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Only process and generate when user clicks the Generate button
button_clicked = st.button("Generate", disabled=(uploaded_file is None))

if button_clicked and uploaded_file is not None:
    print(f"Uploaded file: {uploaded_file}")
    stored_filename = get_from_session(st, SESSION_VARS.CUR_PDF_FILENAME)
    
    if uploaded_file.name != stored_filename:
        put_to_session(st, SESSION_VARS.PROCESSED_DATA, None)
        put_to_session(st, SESSION_VARS.CUR_PDF_FILENAME, uploaded_file.name)

    # Process the file if it's new
    if not get_from_session(st, SESSION_VARS.PROCESSED_DATA):
        with st.expander("Preprocessing"):
            st.write("Reading pdf")
            pages_and_texts = pdf_utils.open_and_read_pdf(uploaded_file)
            
            st.write("Extracting sentences")
            sentencize(pages_and_texts, get_from_session(st, SESSION_VARS.NLP))
            
            # BASELINE: Use original chunk() function (10 sentences, no overlap)
            st.write("Chunking (strategy: **BASELINE** - original chunk(), 10 sentences, no overlap)")
            chunk(pages_and_texts)
            
            pages_and_chunks = chunks_to_text_elems(pages_and_texts)
            st.write("Loading to a DataFrame")
            df = pd.DataFrame(pages_and_chunks)
            
            # For Chinese PDFs, use character length filter instead of token count
            has_cjk = df["sentence_chunk"].astype(str).str.contains(r"[\u4e00-\u9fff]").any()
            if has_cjk:
                pages_and_chunks_over_min_token_len = (
                    df[df["sentence_chunk"].astype(str).str.len() > 120]
                    .to_dict(orient="records")
                )
            else:
                pages_and_chunks_over_min_token_len = (
                    df[df["chunk_token_count"] > min_token_length]
                    .to_dict(orient="records")
                )
            
            st.write("Embedding")
            embed_chunks(pages_and_chunks_over_min_token_len, get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU))
            st.write("Saving embeddings")
            filename = save_embeddings(pages_and_chunks_over_min_token_len)

            put_to_session(st, SESSION_VARS.EMBEDDINGS_FILENAME, filename)
            put_to_session(st, SESSION_VARS.PROCESSED_DATA, True)

    if get_from_session(st, SESSION_VARS.PROCESSED_DATA):
        st.write("Vector Search")
        st.write("Loading embeddings to tensor")
        tensor, pages_and_chunks = embeddings_to_tensor(get_from_session(st, SESSION_VARS.EMBEDDINGS_FILENAME))
        
        # Display tensor device info
        if tensor.device.type == "cuda":
            st.write(f"✅ Embeddings loaded on GPU ({tensor.shape[0]} chunks, {tensor.shape[1]} dimensions)")
        else:
            st.write(f"ℹ️ Embeddings on CPU ({tensor.shape[0]} chunks, {tensor.shape[1]} dimensions)")
        
        # BASELINE: Simple vector search, NO hybrid search, 10 results (same as improved)
        scores, indices = retrieve_relevant_resources(
            query, tensor, get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU), st, 
            n_resources_to_return=10,  # Same as improved for fair comparison
            pages_and_chunks=pages_and_chunks,
            use_hybrid=False  # No hybrid search in baseline
        )
        
        # Create a list of context items
        context_items = [pages_and_chunks[i] for i in indices]
        # Add score to context item
        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu()
            
        st.write(f"Query: {query}")
        with st.expander("Results"):
            # Loop through zipped together scores and indices
            for score, index in zip(scores, indices):
                st.write(f"Score: {score:.4f}")
                st.write(pages_and_chunks[index]["sentence_chunk"])
                st.write(f"Page number: {pages_and_chunks[index]['page_number']}")

        st.write("You selected:", gen_variant)
        with st.expander(f"Answer for query: {query}"):
            with st.spinner("Generating"):
                model_name = OLLAMA_MODEL
                if gen_variant == "vanilla":
                    prompt = format_vanilla_prompt(query)
                    answer = generate_answer_ollama(model_name, prompt, max_tokens=512)
                    st.write(answer)
                elif gen_variant == "rag":
                    prompt = format_rag_prompt(query, context_items)
                    answer = generate_answer_ollama(model_name, prompt, max_tokens=512)
                    st.write(answer)
        st.success("Done!")
