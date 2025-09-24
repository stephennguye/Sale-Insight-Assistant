class DataIngestionError(Exception):
    """Raised when ingestion fails."""

class ModelTrainingError(Exception):
    """Raised when model training fails."""

class RAGError(Exception):
    """Raised for RAG indexing or query issues."""
