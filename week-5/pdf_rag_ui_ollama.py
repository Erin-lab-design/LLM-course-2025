import streamlit as st
import spacy
import re
import torch
from util import pdf_utils
from util.embedings_utils import embed_chunks, save_embeddings, embeddings_to_tensor
from util.nlp_utils import sentencize, sentencize_chinese, chunk, chunk_improved, chunks_to_text_elems
import pandas as pd
from util.session_utils import SESSION_VARS, put_to_session, get_from_session, print_session
from util.vector_search_utils import retrieve_relevant_resources
from util.evaluation_utils import display_evaluation_metrics
import ollama
import jieba

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

# Embedding model selection
EMBEDDING_MODEL_NAME = st.sidebar.selectbox(
    "Embedding Model",
    ("BAAI/bge-large-zh-v1.5", "sentence-transformers/all-mpnet-base-v2"),
    help="bge-large-zh-v1.5: Chinese-optimized (recommended for Chinese docs), all-mpnet-base-v2: Multilingual"
)

# Ollama model name - user can change this
OLLAMA_MODEL = st.sidebar.text_input("Ollama Model Name", value="llama3", help="Name of the Ollama model to use (e.g., llama3, mistral, gemma:2b)")

# Chunking strategy selection
CHUNKING_STRATEGY = st.sidebar.selectbox(
    "Chunking Strategy",
    ("improved", "original", "semantic"),
    help="Improved: 10 sentences with 2 overlap (Task 1). Original: 10 sentences, no overlap. Semantic: Variable-size based on meaning (Task 2)."
)

# Semantic chunking parameters (only shown if semantic is selected)
if CHUNKING_STRATEGY == "semantic":
    st.sidebar.markdown("### Semantic Chunking Config")
    SIMILARITY_THRESHOLD = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.65,
        max_value=0.90,
        value=0.75,
        step=0.05,
        help="Lower = larger chunks, Higher = smaller chunks"
    )
    MAX_CHUNK_SIZE = st.sidebar.number_input("Max Chunk Size (sentences)", value=15, min_value=5, max_value=25)
    MIN_CHUNK_SIZE = st.sidebar.number_input("Min Chunk Size (sentences)", value=3, min_value=2, max_value=10)
else:
    SIMILARITY_THRESHOLD = 0.75
    MAX_CHUNK_SIZE = 15
    MIN_CHUNK_SIZE = 3

# Number of results to return
N_RESULTS = st.sidebar.slider(
    "Number of Results",
    min_value=3,
    max_value=20,
    value=10,
    help="Number of top results to return from vector search"
)

# Hybrid search option
USE_HYBRID = st.sidebar.checkbox(
    "Use Hybrid Search",
    value=True,
    help="Combine vector search with keyword matching to boost definition chunks"
)

# Generation parameters
MAX_TOKENS = st.sidebar.slider(
    "Max Tokens",
    min_value=128,
    max_value=2048,
    value=512,
    step=128,
    help="Maximum number of tokens to generate (higher = longer answers, but slower)"
)

st.write("Initializing models")

if not get_from_session(st, SESSION_VARS.LOADED_MODELS):
    nlp = spacy.load("en_core_web_sm") #English()

    # uncomment this command to print the file location of the Spacy model
    # st.write(nlp._path)

    # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
    nlp.add_pipe("sentencizer")
    put_to_session(st, SESSION_VARS.NLP, nlp)

    # Load selected embedding model
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=EMBED_DEVICE
    )
    st.sidebar.info(f"📦 Using: {EMBEDDING_MODEL_NAME}")

    if EMBED_DEVICE == "cuda":
        st.sidebar.success(f"✅ Embedding model loaded on GPU")
    else:
        st.sidebar.info(f"ℹ️ Embedding model loaded on CPU")
    
    # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    put_to_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU, embedding_model)

    # Ollama client - no need to load model, it's handled by Ollama service
    # Just verify the model is available
    try:
        models = ollama.list()

        # Ollama may return keys like 'name' OR 'model' depending on version
        model_names = []
        for m in models.get("models", []):
            model_names.append(m.get("name") or m.get("model") or "")

        # allow "llama3" to match "llama3:latest"
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

    put_to_session(st, SESSION_VARS.MODEL, OLLAMA_MODEL)  # Store model name instead of model object

    st.write("Done")

    put_to_session(st, SESSION_VARS.LOADED_MODELS, True)
else:
    st.write("Models were already loaded")

print_session(st)

st.title('PDF RAG (Retrieval Augmented Generation) Demo - Ollama Version')
query = st.text_input("Type your query here", "第2条说的‘平等主体’是什么意思？为什么行政关系不属于民法调整？")
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
                'num_predict': max_tokens,  # max_new_tokens equivalent
                'temperature': 0.7,  # Balance between creativity and consistency
            }
        )
        answer = response['message']['content']
        
        # Check if response might be truncated (ends mid-sentence or with incomplete list)
        if answer and len(answer) > 50:  # Only check if answer is substantial
            answer_stripped = answer.rstrip()
            # Check for incomplete patterns that suggest truncation
            incomplete_patterns = [
                answer_stripped.endswith(','),  # Ends with comma
                answer_stripped.endswith(';'),  # Ends with semicolon
                bool(re.search(r'\d+\.\s*$', answer_stripped)),  # Ends with "2. " or similar (incomplete list)
                # Check if ends mid-sentence (no proper ending punctuation and ends with lowercase)
                (not answer_stripped.endswith('.') and 
                 not answer_stripped.endswith('?') and 
                 not answer_stripped.endswith('!') and
                 len(answer_stripped) > 0 and
                 answer_stripped[-1].islower() and
                 len(answer_stripped.split()) > 10),  # Has substantial content
            ]
            if any(incomplete_patterns):
                answer += "\n\n[Note: Response may be truncated. Consider increasing Max Tokens in the sidebar.]"
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

if uploaded_file is not None:
    print(f"Uploaded file: {uploaded_file}")
# Only process and generate when user clicks the Generate button
button_clicked = st.button("Generate", disabled=(uploaded_file is None))

if button_clicked and uploaded_file is not None:
    print(f"Uploaded file: {uploaded_file}")
    # Check if we need to reprocess: new file or chunking strategy changed
    stored_filename = get_from_session(st, SESSION_VARS.CUR_PDF_FILENAME)
    stored_chunking = st.session_state.get("chunking_strategy", None)
    
    if uploaded_file.name != stored_filename or CHUNKING_STRATEGY != stored_chunking:
        put_to_session(st, SESSION_VARS.PROCESSED_DATA, None)
        put_to_session(st, SESSION_VARS.CUR_PDF_FILENAME, uploaded_file.name)
        st.session_state["chunking_strategy"] = CHUNKING_STRATEGY

    # Process the file if it's new or strategy changed
    if not get_from_session(st, SESSION_VARS.PROCESSED_DATA):
        with st.expander("Preprocessing"):
            st.write("Reading pdf")
            pages_and_texts = pdf_utils.open_and_read_pdf(uploaded_file)
            
            # Extract sentences - method matches embedding model choice (like Test 5)
            # Chinese embedding (bge) -> jieba, Multilingual embedding (mpnet) -> spacy
            if EMBEDDING_MODEL_NAME == "BAAI/bge-large-zh-v1.5":
                st.write("📝 Extracting sentences (Chinese-optimized with jieba)")
                st.info("🔗 Using jieba because Chinese embedding model is selected (Test 5 config)")
                sentencize_chinese(pages_and_texts)
            else:
                st.write("📝 Extracting sentences (Spacy)")
                st.info("🔗 Using spacy because multilingual embedding model is selected (Baseline config)")
                sentencize(pages_and_texts, get_from_session(st, SESSION_VARS.NLP))
            
            # chunk - use same strategy as Test 5
            st.write(f"Chunking (strategy: {CHUNKING_STRATEGY})")
            if CHUNKING_STRATEGY == "semantic":
                # Task 2: Semantic chunking based on sentence similarity
                st.info(f"🧠 Semantic Chunking: threshold={SIMILARITY_THRESHOLD}, min={MIN_CHUNK_SIZE}, max={MAX_CHUNK_SIZE}")
                from task2.semantic_chunking import semantic_chunk, analyze_chunking
                
                all_chunks = []
                all_stats = []
                for item in pages_and_texts:
                    page_chunks, page_stats = semantic_chunk(
                        item["sentences"],
                        get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU),
                        similarity_threshold=SIMILARITY_THRESHOLD,
                        max_chunk_size=MAX_CHUNK_SIZE,
                        min_chunk_size=MIN_CHUNK_SIZE,
                        verbose=False
                    )
                    # Convert to dict format
                    item["sentence_chunks"] = page_chunks
                    all_stats.extend(page_stats)
                
                # Show chunking metrics
                from task2.semantic_chunking import analyze_chunking
                metrics = analyze_chunking(all_stats)
                st.success(f"✅ Created {metrics['num_chunks']} chunks | Avg size: {metrics['avg_chunk_size']:.1f} sentences | Similarity: {metrics['avg_within_chunk_similarity']:.3f}")
            
            elif CHUNKING_STRATEGY == "improved":
                # Task 1 Test 5: 10 sentences with 2 overlap
                from util.nlp_utils import split_list_overlapping
                for item in pages_and_texts:
                    item["sentence_chunks"] = split_list_overlapping(item["sentences"], 10, 2)
                st.success("✅ Chunks: 10 sentences, 2 overlap (Task 1 Test 5)")
            else:
                # Original: 10 sentences, no overlap
                chunk(pages_and_texts)
                st.success("✅ Chunks: 10 sentences, no overlap (original)")
            
            # chunks to text elems
            pages_and_chunks = chunks_to_text_elems(pages_and_texts)
            st.write("Loading to a DataFrame")
            df = pd.DataFrame(pages_and_chunks)
            # For Chinese PDFs, token_count can be misleading; use char-length filter instead
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
        
        scores, indices = retrieve_relevant_resources(
            query, tensor, get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU), st, 
            n_resources_to_return=N_RESULTS,
            pages_and_chunks=pages_and_chunks,
            use_hybrid=USE_HYBRID
        )
        # Create a list of context items
        context_items = [pages_and_chunks[i] for i in indices]
        # Add score to context item
        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu()  # return score back to CPU
        st.write(f"Query: {query}")
        
        # Calculate average score for quality assessment
        avg_score = sum(scores).item() / len(scores)
        max_score = max(scores).item()
        st.metric("Retrieval Quality", f"{avg_score:.3f}", f"Max: {max_score:.3f}")
        
        with st.expander("📄 Retrieved Chunks (with highlighted keywords)", expanded=True):
            # Extract keywords from query for highlighting
            query_keywords = set(jieba.cut(query)) if any('\u4e00' <= c <= '\u9fff' for c in query) else set(query.lower().split())
            query_keywords = {kw for kw in query_keywords if len(kw) > 1}  # Filter single chars
            
            # Loop through results
            for rank, (score, index) in enumerate(zip(scores, indices), 1):
                chunk_data = pages_and_chunks[index]
                chunk_text = chunk_data["sentence_chunk"]
                
                # Highlight keywords in chunk
                highlighted_text = chunk_text
                for keyword in query_keywords:
                    if keyword in highlighted_text:
                        highlighted_text = highlighted_text.replace(
                            keyword, 
                            f"**:green[{keyword}]**"
                        )
                
                # Display with rich formatting
                st.markdown(f"### 🔍 Rank {rank} - Score: {score:.4f}")
                st.markdown(highlighted_text)
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📖 Page: {chunk_data['page_number']}")
                with col2:
                    st.caption(f"📊 Tokens: {chunk_data.get('chunk_token_count', 'N/A')}")
                with col3:
                    st.caption(f"📏 Chars: {len(chunk_text)}")
                
                st.divider()

        st.write("You selected:", gen_variant)
        with st.expander(f"💬 Answer for query: {query}", expanded=True):
            with st.spinner("Generating"):
                # Use current model from sidebar (user can change it)
                model_name = OLLAMA_MODEL
                if gen_variant == "vanilla":
                    prompt = format_vanilla_prompt(query)
                    answer = generate_answer_ollama(model_name, prompt, max_tokens=MAX_TOKENS)
                    st.markdown(answer)
                elif gen_variant == "rag":
                    prompt = format_rag_prompt(query, context_items)
                    answer = generate_answer_ollama(model_name, prompt, max_tokens=MAX_TOKENS)
                    st.markdown(answer)
                    
                    # Display evaluation metrics for RAG
                    st.divider()
                    retrieved_text = [item["sentence_chunk"] for item in context_items]
                    score_values = [item["score"] for item in context_items]
                    display_evaluation_metrics(query, answer, retrieved_text, score_values)
        
        st.success("Done!")

