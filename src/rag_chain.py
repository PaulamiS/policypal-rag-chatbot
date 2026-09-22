from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from src import config
from src.ingest import load_index


def build_llm():
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.LLM_MODEL_NAME)

    text2text_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
    )
    return HuggingFacePipeline(pipeline=text2text_pipeline)


def build_rag_chain(index_dir: str = config.INDEX_DIR):
    vectorstore = load_index(index_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})

    prompt = PromptTemplate(
        template=config.PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    llm = build_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    # Return both the chain AND the raw vectorstore, so we can check
    # relevance scores before letting the LLM generate anything.
    return chain, vectorstore


def is_relevant(vectorstore, question: str, threshold: float = config.SIMILARITY_THRESHOLD) -> bool:
    """Checks whether the closest matching chunk is actually relevant enough
    to answer from, instead of always returning *something* regardless of fit."""
    top_matches = vectorstore.similarity_search_with_score(question, k=1)
    if not top_matches:
        return False
    _, best_score = top_matches[0]
    return best_score <= threshold


def answer_question(chain, vectorstore, question: str, threshold: float = config.SIMILARITY_THRESHOLD):
    """Runs the chain and returns (answer, list_of_source_snippets).
    Refuses to answer if nothing relevant enough was found in the documents."""
    if not is_relevant(vectorstore, question):
        return (
            "This document doesn't contain information to answer that question. "
            "Please ask something related to the uploaded policy.",
            [],
        )

    result = chain.invoke({"query": question})
    answer = result["result"]

    # Only show sources that are actually close matches to the question,
    # not every chunk that got "stuffed" into the prompt.
    scored_matches = vectorstore.similarity_search_with_score(question, k=config.TOP_K)
    relevant_paths = {
        doc.metadata.get("source", "unknown")
        for doc, score in scored_matches
        if score <= threshold
    }
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:200],
        }
        for doc in result.get("source_documents", [])
        if doc.metadata.get("source", "unknown") in relevant_paths
    ]
    return answer, sources