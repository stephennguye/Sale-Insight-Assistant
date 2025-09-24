"""RAG pipeline with configurable generator backend (Ollama or HuggingFace)."""

from pathlib import Path
import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer
import subprocess
import json
from transformers import pipeline

from src.utils.logger import get_logger
from src.utils.exceptions import RAGError

logger = get_logger("rag_query")

# Paths
INDEX_PATH = Path("D:/GitHub/Sale Insight Assistant/data/faiss.index")
META_PATH = Path("D:/GitHub/Sale Insight Assistant/data/docs_meta.pkl")

# Config (switch here or via env vars)
GENERATOR_BACKEND = os.getenv("GENERATOR_BACKEND", "ollama")  # "ollama" or "hf"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
HF_MODEL = os.getenv("HF_MODEL", "distilgpt2")

# Lazy globals
_embed_model = None
_index = None
_meta = None
_hf_gen = None


def _ensure_loaded():
    """Load embeddings, FAISS index, and metadata lazily."""
    global _embed_model, _index, _meta
    if _embed_model is None:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if _index is None:
        if not INDEX_PATH.exists():
            raise RAGError("Index file not found. Run rag_index first.")
        logger.info("Loading FAISS index...")
        _index = faiss.read_index(str(INDEX_PATH))

    if _meta is None:
        if not META_PATH.exists():
            raise RAGError("Metadata file not found. Run rag_index first.")
        logger.info("Loading metadata...")
        with open(META_PATH, "rb") as f:
            _meta = pickle.load(f)


def retrieve(query: str, k: int = 3):
    """Retrieve top-k documents relevant to the query."""
    try:
        _ensure_loaded()
        q_emb = _embed_model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = _index.search(q_emb, k)
        results = [_meta[idx] for idx in I[0]]
        return results
    except Exception as e:
        logger.exception("RAG retrieval failed")
        raise RAGError(str(e)) from e


def _ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama locally and return the generated text."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        output_lines = result.stdout.decode("utf-8").splitlines()
        texts = []
        for line in output_lines:
            try:
                obj = json.loads(line)
                if "response" in obj:
                    texts.append(obj["response"])
            except json.JSONDecodeError:
                continue
        return "".join(texts).strip()
    except Exception as e:
        logger.exception("Ollama generation failed")
        raise RAGError(f"Ollama generation failed: {e}") from e


def _hf_generate(prompt: str, model: str = HF_MODEL) -> str:
    """Fallback: use HuggingFace pipeline."""
    global _hf_gen
    try:
        if _hf_gen is None:
            logger.info("Loading HuggingFace generator: %s", model)
            _hf_gen = pipeline("text-generation", model=model)
        out = _hf_gen(prompt, max_new_tokens=150)[0]["generated_text"]
        return out.strip()
    except Exception as e:
        logger.exception("HF generation failed")
        raise RAGError(f"HF generation failed: {e}") from e


def generate_answer(query: str) -> str:
    """Retrieve context and generate an answer with chosen backend."""
    try:
        _ensure_loaded()
        retrieved = retrieve(query, k=3)

        ctx = "\n---\n".join([
            f"Source: {r['source']} (chunk {r['chunk']})\n{r['text']}"
            for r in retrieved
        ])

        prompt = f"""You are a helpful assistant. Use the context to answer.

Context:
{ctx}

Question: {query}
Answer concisely:"""

        if GENERATOR_BACKEND == "ollama":
            return _ollama_generate(prompt, model=OLLAMA_MODEL)
        elif GENERATOR_BACKEND == "hf":
            return _hf_generate(prompt, model=HF_MODEL)
        else:
            raise RAGError(f"Unsupported generator backend: {GENERATOR_BACKEND}")

    except Exception as e:
        logger.exception("Generation failed")
        raise RAGError(str(e)) from e


if __name__ == "__main__":
    q = "What were the sales trends in Q1?"
    print("Q:", q)
    print("A:", generate_answer(q))
