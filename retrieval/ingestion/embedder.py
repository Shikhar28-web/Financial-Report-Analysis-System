import os
from dotenv import load_dotenv
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Union

load_dotenv()

from sentence_transformers import SentenceTransformer
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first time may take a moment)...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded successfully!")
    return _embedding_model

# Simple on-disk cache
CACHE_DIR = Path("./data/embed_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch embed a list of texts using sentence-transformers.
    Uses an on-disk cache keyed by text hash.
    """
    model = get_embedding_model()
    if model is None:
        raise ValueError("Embedding model not available.")
    
    results = []
    to_call = []
    indexes = []

    # Check cache first
    for i, t in enumerate(texts):
        key = _cache_key(t)
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            emb = json.loads(cache_file.read_text())["embedding"]
            results.append(emb)
        else:
            results.append(None)
            to_call.append(t)
            indexes.append(i)

    if to_call:
        print(f"🔄 Embedding {len(to_call)} new chunks...")
        embeddings = model.encode(to_call, show_progress_bar=False)
        for idx, emb in enumerate(embeddings):
            emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
            orig_i = indexes[idx]
            results[orig_i] = emb_list
            # write cache
            key = _cache_key(to_call[idx])
            cache_file = CACHE_DIR / f"{key}.json"
            cache_file.write_text(json.dumps({"embedding": emb_list}))
    
    return results


def embed_and_store(chunks, metadata=None, metadata_list=None, client=None):
    """
    Embed chunks and store them in ChromaDB with enhanced metadata support.
    
    Args:
        chunks: list of text strings
        metadata: dict with metadata to add to ALL chunks (legacy support)
        metadata_list: list of dicts, one per chunk (NEW - for per-chunk metadata)
        client: ChromaDB collection client (MUST be provided!)
    """
    print(f"\n{'='*80}")
    print(f"📥 EMBEDDING AND STORING {len(chunks)} CHUNKS")
    print(f"{'='*80}\n")
    
    if client is None:
        print("❌ ERROR: No client provided to embed_and_store!")
        from retrieval.vector_store.chroma_client import get_chroma_client
        client = get_chroma_client()
    
    print(f"📦 Target collection: {client.name}")
    print(f"📊 Collection currently has: {client.count()} documents")
    
    # Embed all chunks
    embeddings = embed_texts(chunks)
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Prepare documents for storage
    items = []
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest() + f"_{i}"
        
        # Handle metadata
        if metadata_list and i < len(metadata_list):
            # Use per-chunk metadata (NEW - preferred)
            chunk_metadata = metadata_list[i].copy()
        elif metadata:
            # Use global metadata (legacy support)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
        else:
            chunk_metadata = {"chunk_index": i}
        
        items.append({
            "id": chunk_id,
            "text": chunk,
            "embedding": emb,
            "metadata": chunk_metadata
        })
    
    print(f"📝 Prepared {len(items)} items")
    print(f"📋 Sample metadata: {items[0]['metadata']}")
    
    # Store DIRECTLY in the provided client collection
    try:
        ids = [it["id"] for it in items]
        docs = [it["text"] for it in items]
        embs = [it["embedding"] for it in items]
        metadatas = [it["metadata"] for it in items]
        
        print(f"💾 Storing to collection: {client.name}")
        
        # Try upsert (ChromaDB 1.x)
        try:
            client.upsert(
                ids=ids,
                documents=docs,
                embeddings=embs,
                metadatas=metadatas
            )
            print(f"✅ Upserted {len(items)} documents")
        except AttributeError:
            # Fallback to add (ChromaDB 0.4.x)
            client.add(
                ids=ids,
                documents=docs,
                embeddings=embs,
                metadatas=metadatas
            )
            print(f"✅ Added {len(items)} documents (fallback method)")
        
        # Verify storage
        new_count = client.count()
        print(f"📊 Collection now has: {new_count} documents (was {new_count - len(items)})")
        
        if new_count == 0:
            print(f"❌ WARNING: Collection is still empty after storage!")
        
    except Exception as e:
        print(f"❌ ERROR storing documents: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\n{'='*80}\n")