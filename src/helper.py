
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def download_hugging_face_embeddings():
    """Load a lightweight ONNX-based embedding model."""
    return FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
