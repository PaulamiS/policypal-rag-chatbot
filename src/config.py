import os

# --- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "data", "sample_docs")
INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

# --- Embeddings ----------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retrieval -------------------------------------------------------------
TOP_K = 4
SIMILARITY_THRESHOLD = 1.5  # Lower distance score = more relevant. Above this, treat as "not in this document."
# --- LLM -----------------------------------------------------------------
LLM_MODEL_NAME = "google/flan-t5-large"
LLM_MAX_NEW_TOKENS = 256

PROMPT_TEMPLATE = """You are an insurance policy assistant. Answer the customer's
question in plain, simple English using ONLY the policy text in the context
below — never use outside knowledge about insurance in general. If the
answer isn't in the context, say the policy document doesn't specify this
and recommend they contact SecureLife support. Never guess at numbers,
waiting periods, or exclusions that aren't explicitly stated.

Policy context:
{context}

Customer question: {question}

Answer:"""