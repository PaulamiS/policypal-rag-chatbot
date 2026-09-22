# PolicyPal — Insurance Policy Q&A Assistant (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that reads insurance policy
documents and answers customer questions in plain English — grounded only
in the actual policy text, with the source clause shown for every answer.
Runs fully on open-weight, local models — no paid API required.

## The problem this solves

Insurance policy documents are long, dense, and full of jargon. Customers
routinely miss critical details buried in exclusions or waiting-period
clauses. PolicyPal lets someone type "Is dental treatment covered?" and
get a direct, sourced answer instead of digging through pages of text.

## Architecture
policy documents (.pdf/.txt)
│
▼
chunking (RecursiveCharacterTextSplitter)
│
▼
embeddings (sentence-transformers, HuggingFace)
│
▼
FAISS vector index ──▶ retriever + relevance check (similarity threshold)
│ │
│ ▼
│ prompt + retrieved clauses
│ │
│ ▼
└──────────────▶ local LLM (flan-t5-large) via LangChain
│
▼
plain-English answer + source clause
(or an honest refusal if nothing relevant was found)

## Project structure
rag_chatbot/
├── app_streamlit.py # "PolicyPal" chat UI
├── app_api.py # FastAPI service (POST /query)
├── src/
│ ├── config.py # Models, chunk size, prompt persona, similarity threshold
│ ├── ingest.py # Load → chunk → embed → build/save FAISS index
│ └── rag_chain.py # Retriever + relevance check + prompt + LLM chain
├── data/sample_docs/
│ └── sample_policy.txt # Sample health insurance policy
├── Dockerfile
└── requirements.txt

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

## Run it

**1. Build the index:**
```bash
python -m src.ingest
```

**2. Chat UI:**
```bash
streamlit run app_streamlit.py
```
Try asking: *"Is dental treatment covered?"*, *"What's the waiting period
for pre-existing diseases?"*, *"What's the co-payment for someone above
60?"*, or an off-topic question like *"What is the capital of France?"* —
the last one should be honestly refused instead of answered.

You can also upload your own policy PDF/TXT from the sidebar and rebuild
the index without touching the command line.

**3. Or the API:**
```bash
uvicorn app_api:app --reload --port 8000
```
Test with:
```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"Is dental treatment covered?\"}"
```

**4. Or Docker:**
```bash
docker build -t policypal .
docker run -p 8000:8000 policypal
```

## Key engineering decision: grounding and relevance filtering

Early testing showed the LLM would sometimes answer general-knowledge
questions (e.g. "What is the capital of France?") using its own training
knowledge instead of admitting the document didn't cover it — a common
RAG failure mode. I fixed this by adding a similarity-score threshold
(`SIMILARITY_THRESHOLD` in `config.py`): if the closest matching document
chunk isn't actually close enough to the question, the system refuses to
answer instead of guessing. The same threshold also filters which sources
are shown, so irrelevant chunks aren't displayed as if they contributed
to an answer.

## Swapping models

Everything model-related lives in `src/config.py`:
- `EMBEDDING_MODEL_NAME` — any sentence-transformers model on HuggingFace.
- `LLM_MODEL_NAME` — defaults to `google/flan-t5-large` for better
  reasoning over exclusion clauses than the smaller `flan-t5-base`. For
  even better answers with a GPU, swap to an instruction-tuned causal
  model such as `Qwen/Qwen2.5-1.5B-Instruct` or
  `mistralai/Mistral-7B-Instruct-v0.3` — you'll also need to switch the
  pipeline task in `src/rag_chain.py` from `"text2text-generation"` to
  `"text-generation"` and load with `AutoModelForCausalLM` instead of
  `AutoModelForSeq2SeqLM`.
- `SIMILARITY_THRESHOLD` — controls how strict the relevance check is.
- `PROMPT_TEMPLATE` — currently an "insurance policy assistant" persona.
- The vector store is FAISS; swapping to Chroma, Pinecone, or Weaviate
  only touches `src/ingest.py` and `src/rag_chain.py`.

## Known limitation

Smaller open-weight models are not always consistent in multi-part
reasoning tasks (e.g. reliably explaining *why* an answer is yes/no, not
just stating it). This was confirmed to be a generation-side limitation,
not a retrieval issue — retrieval consistently found the correct policy
clause across every test. A larger hosted model would resolve this
without any other architecture changes.

## Ideas to extend it further

- **Agentic layer**: wrap the retriever as a LangChain/LangGraph tool and
  build an agent that decides when to search vs. ask a clarifying question.
- **Fine-tuning**: LoRA/QLoRA fine-tune the LLM on real insurance Q&A pairs.
- **Hybrid search**: combine FAISS similarity search with BM25 keyword
  search for exact-match terms like policy clause numbers.

## What this project demonstrates

| Skill / keyword | Where it shows up |
|---|---|
| LangChain | `RetrievalQA` chain, `PromptTemplate` |
| FAISS / vector databases | `src/ingest.py`, `src/rag_chain.py` |
| HuggingFace Transformers | embeddings + local LLM pipeline |
| RAG pipelines, chunking, embeddings | `src/ingest.py` |
| Prompt engineering | `config.PROMPT_TEMPLATE` |
| Grounding / hallucination mitigation | similarity threshold in `rag_chain.py` |
| FastAPI | `app_api.py` |
| Streamlit | `app_streamlit.py` |
| Docker | `Dockerfile` |

**Suggested resume bullet:**
> Built PolicyPal, a Retrieval-Augmented Generation (RAG) assistant that
> answers plain-English questions about insurance policy documents using
> LangChain, FAISS, and open-weight HuggingFace LLMs; added a similarity-
> threshold grounding check to reduce hallucinated answers, and deployed
> via a Streamlit UI and FastAPI service, containerized with Docker.