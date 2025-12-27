import os
from dotenv import load_dotenv
import chromadb
from typing import List, Dict

load_dotenv()
PERSIST_DIR = os.environ.get("PERSIST_DIRECTORY", "./data/chroma_db")

_chroma_client = None
_collections = {}

def get_client():
    global _chroma_client
    if _chroma_client is None:
        try:
            # Try new API (ChromaDB 1.x)
            _chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        except:
            try:
                # Fallback to old API (ChromaDB 0.4.x)
                from chromadb.config import Settings
                _chroma_client = chromadb.PersistentClient(
                    path=PERSIST_DIR,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except:
                # Last resort - use Client
                _chroma_client = chromadb.Client()
    return _chroma_client

def get_chroma_client(collection_name: str = "financial_reports"):
    """Get or create a ChromaDB collection."""
    return get_collection(name=collection_name)

def get_collection(name: str = "financial_reports"):
    global _collections
    
    if name in _collections:
        return _collections[name]
    
    client = get_client()
    try:
        collection = client.get_collection(name=name)
        print(f"✅ Loaded existing collection: {name}")
    except:
        collection = client.create_collection(name=name)
        print(f"✅ Created new collection: {name}")
    
    _collections[name] = collection
    return collection

def upsert_documents(items: List[Dict]):
    """items: list of dicts with keys: id, text, embedding, metadata"""
    col = get_collection()
    ids = [it["id"] for it in items]
    docs = [it["text"] for it in items]
    embs = [it["embedding"] for it in items]
    metadatas = [it.get("metadata", {}) for it in items]
    
    try:
        # Try upsert (ChromaDB 1.x)
        col.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)
    except:
        # Fallback to add (ChromaDB 0.4.x)
        col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)

def query_vector(embedding, n_results=10, where=None):
    col = get_collection()
    query_results = col.query(
        query_embeddings=[embedding], 
        n_results=n_results, 
        where=where
    )
    return query_results