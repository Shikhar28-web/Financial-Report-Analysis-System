import os
import re
from retrieval.vector_store.chroma_client import get_chroma_client
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize embedding model ONCE
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

groq_api_key = os.getenv("GROQ_API_KEY")
print(f"DEBUG: API Key loaded: {'Yes' if groq_api_key else 'No'}")

try:
    client = Groq(api_key=groq_api_key)
    print("DEBUG: Groq client created successfully")
except Exception as e:
    print(f"ERROR creating Groq client: {e}")
    client = None


def normalize_answer(answer):
    """Remove newlines and extra whitespace."""
    if not answer:
        return answer
    answer = answer.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    answer = re.sub(r'\s+', ' ', answer)
    answer = re.sub(r'\s+([.,;:!?])', r'\1', answer)
    return answer.strip()


def ask_llm(prompt):
    """Ask Groq for answer."""
    if not client:
        return "Error: Groq client not initialized."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a helpful financial assistant.

RULES:
1. Answer in ONE clear paragraph
2. Format numbers: ₹1,28,933 crore or $19.3 billion
3. NEVER start with "Based on" or "According to"
4. Use EXACT numbers from context
5. Include fiscal year in answer
6. Be direct and factual"""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )
        
        answer = response.choices[0].message.content
        return normalize_answer(answer)
        
    except Exception as e:
        print(f"ERROR in ask_llm: {e}")
        return f"Error calling Groq API: {str(e)}"


def get_query_embedding(text):
    """Generate embedding for query."""
    embedding = embedding_model.encode(text)
    return embedding.tolist()


def answer_user_query(query, user_id, top_k=10, debug=True):
    """
    Answer user query with RAG.
    """
    collection_name = f"user_{user_id}"
    
    if debug:
        print(f"\n{'='*80}")
        print(f"🔍 QUERYING: {query}")
        print(f"📦 Collection: {collection_name}")
        print(f"{'='*80}\n")
    
    try:
        collection = get_chroma_client(collection_name=collection_name)
        
        # Check if collection has data
        count = collection.count()
        if debug:
            print(f"📊 Collection has {count} documents")
        
        if count == 0:
            return "No documents found. Please upload a document first."
        
    except Exception as e:
        print(f"ERROR: Could not get collection: {e}")
        return "Error: Could not access your documents. Please upload a document first."

    # Generate embedding
    if debug:
        print(f"🔄 Generating query embedding...")
    
    query_embedding = get_query_embedding(query)
    
    if debug:
        print(f"✅ Embedding generated (dimension: {len(query_embedding)})")

    # Query the collection
    try:
        if debug:
            print(f"🔍 Searching for top {top_k} results...")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if debug:
            print(f"✅ Query completed")
            
    except Exception as e:
        print(f"ERROR querying collection: {e}")
        return f"Error retrieving information: {str(e)}"

    # Check results
    if not results or not results.get('documents') or not results['documents'][0]:
        if debug:
            print(f"❌ No results returned from query")
        return "No relevant information found in your uploaded document."

    documents = results['documents'][0]
    metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
    distances = results.get('distances', [[]])[0] if results.get('distances') else []

    if debug:
        print(f"\n📊 Retrieved {len(documents)} chunks")
        print(f"\nTop 3 results:")
        for i, (doc, dist) in enumerate(zip(documents[:3], distances[:3]), 1):
            meta = metadatas[i-1] if i-1 < len(metadatas) else {}
            print(f"\n--- CHUNK {i} (Distance: {dist:.4f}) ---")
            print(f"Page: {meta.get('page_num', 'N/A')}")
            print(f"FY: {meta.get('fiscal_year', 'N/A')}")
            print(f"Text: {doc[:200]}...")

    # Build context
    context_parts = []
    for i, doc in enumerate(documents[:5]):  # Use top 5
        meta = metadatas[i] if i < len(metadatas) else {}
        fy = meta.get('fiscal_year', 'Unknown')
        page = meta.get('page_num', 'N/A')
        context_parts.append(f"[FY: {fy} | Page: {page}]\n{doc.strip()}")
    
    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""You are analyzing financial documents.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Use EXACT numbers from context
- Include fiscal year
- Be direct and concise
- Don't mention "context" or "document"

Answer:"""

    if debug:
        print(f"\n🤖 Sending to LLM...")
        print(f"Context length: {len(context)} characters")

    answer = ask_llm(prompt)
    
    if debug:
        print(f"\n✅ FINAL ANSWER: {answer}\n")

    return answer


def answer_user_query_simple(query, user_id):
    """Simple version without debug."""
    return answer_user_query(query, user_id, top_k=10, debug=False)


if __name__ == "__main__":
    user_id = "test_user"
    query = "What was Infosys revenue for FY 2024?"
    
    print("🧪 Testing RAG system...\n")
    answer = answer_user_query(query, user_id, debug=True)
    print(f"\n{'='*80}")
    print("FINAL ANSWER:")
    print("="*80)
    print(answer)