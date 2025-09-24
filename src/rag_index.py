"""Build a FAISS index from text/PDF/Word files in data/docs."""
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.exceptions import RAGError

# Extra imports for PDF & Word
from PyPDF2 import PdfReader
import docx

logger = get_logger("rag_index")
DOCS_DIR = Path("data/docs")
INDEX_PATH = Path("data/faiss.index")
META_PATH = Path("data/docs_meta.pkl")


def chunk_text(text: str, chunk_size: int = 800):
    """Split text into overlapping chunks for embedding."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf8", errors="ignore")


def load_pdf(path: Path) -> str:
    """Extract text from PDF using PyPDF2."""
    text = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            text.append(page.extract_text() or "")
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", path, e)
    return "\n".join(text)


def load_docx(path: Path) -> str:
    """Extract text from Word doc using python-docx."""
    text = []
    try:
        doc = docx.Document(str(path))
        for para in doc.paragraphs:
            text.append(para.text)
    except Exception as e:
        logger.error("Failed to read DOCX %s: %s", path, e)
    return "\n".join(text)


def main():
    if not DOCS_DIR.exists():
        logger.error("Docs folder missing: %s", DOCS_DIR)
        raise FileNotFoundError("data/docs not found")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts, meta = [], []

    # Loop through supported file formats
    for f in DOCS_DIR.iterdir():
        if f.suffix.lower() == ".txt":
            raw_text = load_txt(f)
        elif f.suffix.lower() == ".pdf":
            raw_text = load_pdf(f)
        elif f.suffix.lower() == ".docx":
            raw_text = load_docx(f)
        else:
            logger.info("Skipping unsupported file type: %s", f)
            continue

        if not raw_text.strip():
            logger.warning("No content extracted from %s", f)
            continue

        chunks = chunk_text(raw_text)
        for i, c in enumerate(chunks):
            texts.append(c)
            meta.append({"source": f.name, "chunk": i})

    if not texts:
        logger.warning("No documents found to index in %s", DOCS_DIR)
        return

    # Build embeddings + FAISS
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    logger.info("FAISS index built with %d vectors from %d docs", len(texts), len(meta))


if __name__ == "__main__":
    main()
