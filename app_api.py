import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config
from src.rag_chain import build_rag_chain, answer_question

app = FastAPI(title="PolicyPal API", version="1.0")
_chain = None
_vectorstore = None


class QueryRequest(BaseModel):
    question: str


class SourceSnippet(BaseModel):
    source: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]


@app.on_event("startup")
def load_chain():
    global _chain, _vectorstore
    if os.path.exists(config.INDEX_DIR):
        _chain, _vectorstore = build_rag_chain()


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": _chain is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global _chain, _vectorstore
    if _chain is None:
        if not os.path.exists(config.INDEX_DIR):
            raise HTTPException(status_code=400, detail="Run `python -m src.ingest` first.")
        _chain, _vectorstore = build_rag_chain()
    answer, sources = answer_question(_chain, _vectorstore, req.question)
    return QueryResponse(answer=answer, sources=sources)