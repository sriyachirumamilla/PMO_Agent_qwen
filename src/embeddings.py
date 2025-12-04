"""
Free embeddings using Sentence Transformers.
100% open-source, runs locally.
"""

from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


def get_embeddings():
    """Get free, local embeddings model.
    
    Uses Sentence Transformers for semantic similarity.
    Model: all-MiniLM-L6-v2 (80MB, very fast)
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"Loading embeddings: {embedding_model}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            "device": "cpu",  # Force CPU for stability
        }
    )
    
    print("✓ Embeddings loaded successfully")
    return embeddings


# Test embeddings
if __name__ == "__main__":
    embeddings = get_embeddings()
    
    # Test embedding
    text = "Create a new project task"
    vec = embeddings.embed_query(text)
    print(f"\nEmbedding for: '{text}'")
    print(f"Vector dimension: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")