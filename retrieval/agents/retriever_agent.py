# agents/retriever_agent.py

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

load_dotenv()
PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./data/chroma_db")

# Local embeddings (sentence-transformers)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector database
vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="financial_reports"
)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# LLM for generating answers (local, free)
_llm_model = None
_llm_tokenizer = None

class QueryType(Enum):
    """Classification of query types for better handling."""
    COMPARATIVE = "comparative"  # highest, lowest, best, worst
    SPECIFIC_VALUE = "specific_value"  # value for specific entity
    COUNT = "count"  # how many
    TOTAL = "total"  # sum/aggregate
    FACTUAL = "factual"  # what is, tell me about
    ANALYTICAL = "analytical"  # why, how, explain
    LIST = "list"  # list all, show all
    TEMPORAL = "temporal"  # when, date-related
    CAUSAL = "causal"  # why, because, reason
    UNKNOWN = "unknown"


def classify_query(query: str) -> Tuple[QueryType, Dict]:
    """
    Classify the query type and extract relevant information.
    Returns query type and metadata about the query.
    """
    query_lower = query.lower()
    metadata = {
        "is_comparative": False,
        "is_question": query.strip().endswith("?"),
        "has_numbers": bool(re.search(r'\d+', query)),
        "keywords": []
    }
    
    # Comparative queries
    comparative_keywords = ["highest", "lowest", "top", "maximum", "minimum", "max", "min", 
                           "best", "worst", "most", "least", "largest", "smallest", "biggest"]
    if any(kw in query_lower for kw in comparative_keywords):
        metadata["is_comparative"] = True
        metadata["keywords"].extend([kw for kw in comparative_keywords if kw in query_lower])
        return QueryType.COMPARATIVE, metadata
    
    # Count queries
    count_keywords = ["how many", "count", "number of", "total number", "how much"]
    if any(kw in query_lower for kw in count_keywords):
        return QueryType.COUNT, metadata
    
    # Temporal queries
    temporal_keywords = ["when", "date", "time", "year", "month", "day", "period", "quarter"]
    if any(kw in query_lower for kw in temporal_keywords):
        return QueryType.TEMPORAL, metadata
    
    # Causal/Analytical queries
    causal_keywords = ["why", "reason", "because", "cause", "due to", "result of"]
    analytical_keywords = ["how", "explain", "describe", "analyze", "analysis"]
    if any(kw in query_lower for kw in causal_keywords):
        return QueryType.CAUSAL, metadata
    if any(kw in query_lower for kw in analytical_keywords):
        return QueryType.ANALYTICAL, metadata
    
    # List queries
    list_keywords = ["list", "show all", "all", "every", "each"]
    if any(kw in query_lower for kw in list_keywords) and not any(kw in query_lower for kw in ["total", "sum"]):
        return QueryType.LIST, metadata
    
    # Specific value queries (entity + metric)
    entity_keywords = ["for", "of", "in", "at"]
    metric_keywords = ["revenue", "sales", "profit", "income", "cost", "price", "value"]
    if any(ek in query_lower for ek in entity_keywords) and any(mk in query_lower for mk in metric_keywords):
        return QueryType.SPECIFIC_VALUE, metadata
    
    # Total/aggregate queries
    total_keywords = ["total", "sum", "aggregate", "overall", "combined"]
    if any(kw in query_lower for kw in total_keywords):
        return QueryType.TOTAL, metadata
    
    # Factual queries (default for "what is", "tell me", etc.)
    factual_keywords = ["what is", "what are", "tell me", "what", "define"]
    if any(kw in query_lower for kw in factual_keywords):
        return QueryType.FACTUAL, metadata
    
    return QueryType.UNKNOWN, metadata


def get_llm_model():
    """Load a local LLM model for answer generation (free, no API key required)."""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        # Allow disabling local LLM via env if it's hurting accuracy
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        if not use_local:
            _llm_model = False
            return _llm_model
        try:
            from transformers import pipeline
            print("Loading LLM model for answer generation (first time may take a moment)...")
            _llm_model = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=-1,  # Use CPU (-1) or GPU (0+)
                max_length=512
            )
            print("LLM model loaded successfully!")
        except ImportError:
            print("Warning: transformers library not found. Install with: pip install transformers torch")
            _llm_model = False
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            print("Falling back to rule-based answers only.")
            _llm_model = False
    return _llm_model


def get_api_llm_answer(query: str, context: str) -> Optional[str]:
    """
    Try to get answer from API-based LLM (OpenAI/Anthropic) if available.
    Falls back gracefully if not configured.
    """
    # Try OpenAI first
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            prompt = f"""You are a helpful assistant that answers questions based on provided context.
Your task is to provide accurate, concise, and well-structured answers.

Context:
{context[:4000]}

Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "I cannot find the answer in the provided documents."
3. Be specific and cite relevant details when possible
4. Format numbers and data clearly
5. If asked for comparisons, provide clear comparisons
6. For analytical questions, provide reasoning based on the context

Answer:"""
            
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        pass  # Fall back to local LLM
    
    # Try Anthropic Claude
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            
            prompt = f"""Context:
{context[:4000]}

Question: {query}

Answer based on the context above. If the answer is not in the context, say "I cannot find the answer in the provided documents." """
            
            response = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
    except Exception:
        pass  # Fall back to local LLM
    
    return None


def generate_llm_answer(query: str, context: str, query_type: QueryType = QueryType.UNKNOWN) -> Optional[str]:
    """
    Generate an answer using LLM from the given context.
    Tries API-based LLMs first, then falls back to local model.
    """
    # Try API-based LLM first (better quality)
    api_answer = get_api_llm_answer(query, context)
    if api_answer:
        # Safety: avoid obviously hallucinated numeric tables
        if not is_likely_hallucinated_numeric_answer(api_answer, context):
            return api_answer
    
    # Fall back to local LLM
    model = get_llm_model()
    if not model or model is False:
        return None
    
    try:
        # Create enhanced prompt based on query type
        query_type_instructions = {
            QueryType.COMPARATIVE: "Compare the values and identify the highest/lowest/best/worst based on the context.",
            QueryType.ANALYTICAL: "Provide a detailed analysis and explanation based on the context.",
            QueryType.CAUSAL: "Explain the reasons and causes based on the context.",
            QueryType.FACTUAL: "Provide a clear, factual answer based on the context.",
            QueryType.LIST: "List all relevant items mentioned in the context.",
            QueryType.TEMPORAL: "Focus on dates, times, and temporal information from the context.",
            QueryType.SPECIFIC_VALUE: "Extract the specific value for the requested entity and metric.",
            QueryType.COUNT: "Count the number of items mentioned in the context.",
            QueryType.TOTAL: "Calculate or identify the total/aggregate value from the context.",
        }
        
        instruction = query_type_instructions.get(query_type, "Answer the question based on the provided context.")
        
        prompt = f"""You are an intelligent assistant that answers questions based on provided context.

{instruction}

Context:
{context[:600]}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
- Be specific, accurate, and well-structured
- Format numbers clearly (use commas for thousands, include currency symbols if relevant)
- For comparisons, clearly state which is higher/lower/better
- For analytical questions, provide reasoning
- Keep answers concise but complete

Answer:"""
        
        # Generate answer (deterministic for stability)
        result = model(prompt, max_length=300, num_return_sequences=1, do_sample=False, temperature=0.0)
        answer = result[0]['generated_text'].strip()
        
        # Clean up the answer
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        if answer.lower().startswith("the answer is"):
            answer = answer[13:].strip()
        
        if not answer or len(answer) <= 5:
            return None

        # Safety: avoid obviously hallucinated numeric tables
        if is_likely_hallucinated_numeric_answer(answer, context):
            return None

        return answer
    except Exception as e:
        print(f"Error generating LLM answer: {e}")
        return None


def format_docs(docs: List[Document], max_length: int = 5000) -> str:
    """Format documents into a single context string with better structure."""
    formatted_parts = []
    current_length = 0
    
    for i, doc in enumerate(docs):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        meta = doc.metadata if hasattr(doc, 'metadata') else {}
        source = meta.get('source', 'Unknown')
        
        # Truncate if needed
        remaining = max_length - current_length
        if remaining <= 0:
            break
        
        if len(content) > remaining:
            content = content[:remaining] + "..."
        
        formatted_parts.append(f"[Source: {source}]\n{content}")
        current_length += len(content) + len(source) + 15
    
    return "\n\n---\n\n".join(formatted_parts)


def extract_key_value_pairs(text):
    """
    Dynamically extract key-value pairs from text.
    Handles formats like:
    - "Key: Value"
    - "Key : Value"
    - "Key: Value |"
    - "Key: $123.45"
    """
    pairs = {}
    # Pattern to match "Key: Value" or "Key : Value"
    # Allow $ in values, stop at | or end of line
    pattern = r'([^:|\n]+?)\s*:\s*([^|\n]+?)(?:\s*\||\s*$)'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    for match in matches:
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value:
            pairs[key] = value
    
    return pairs


def extract_numeric_value(text):
    """Extract numeric value from text, handling currency, commas, parentheses for negatives."""
    # Remove currency symbols and extract number
    numeric_pattern = r'[\d,()]+\.?\d*'
    match = re.search(numeric_pattern, text)
    if match:
        num_str = match.group(0).replace(',', '').replace('(', '-').replace(')', '').strip()
        try:
            return float(num_str)
        except:
            return None
    return None


def find_matching_field(query_lower, field_name):
    """Check if a field name semantically matches the query with improved matching."""
    field_lower = field_name.lower()
    
    # Direct matches
    if query_lower in field_lower or field_lower in query_lower:
        return True
    
    # Word-level matches
    query_words = set(query_lower.split())
    field_words = set(field_lower.split())
    if query_words.intersection(field_words):
        return True
    
    # Semantic matches for common fields (expanded)
    semantic_map = {
        'revenue': ['revenue', 'gross sales', 'income', 'sales', 'gross', 'earnings'],
        'profit': ['profit', 'net income', 'earnings', 'income', 'net profit', 'bottom line'],
        'sales': ['sales', 'revenue', 'gross sales', 'gross', 'turnover'],
        'country': ['country', 'nation', 'region', 'location', 'territory'],
        'product': ['product', 'item', 'goods', 'merchandise', 'sku', 'article'],
        'units': ['units', 'quantity', 'count', 'number', 'qty', 'amount'],
        'price': ['price', 'cost', 'amount', 'value', 'rate', 'fee'],
        'date': ['date', 'time', 'period', 'year', 'month', 'day', 'quarter'],
        'customer': ['customer', 'client', 'buyer', 'purchaser'],
        'category': ['category', 'type', 'class', 'group', 'segment', 'division']
    }
    
    for key, synonyms in semantic_map.items():
        # Check if query contains any synonym and field contains any synonym
        query_has_syn = any(syn in query_lower for syn in synonyms)
        field_has_syn = any(syn in field_lower for syn in synonyms)
        if query_has_syn and field_has_syn:
            return True
    
    return False


def is_generic_answer(answer: str) -> bool:
    """Check if the answer is too generic and should be replaced with LLM answer."""
    generic_phrases = [
        "relevant information found",
        "check sources for details",
        "information found in documents",
        "see sources",
        "check the sources",
        "i cannot find",
        "not found"
    ]
    answer_lower = answer.lower()
    # Treat very short answers as generic / low quality
    if len(answer_lower.strip()) < 25:
        return True
    return any(phrase in answer_lower for phrase in generic_phrases)


def normalize_answer_format(answer: str) -> str:
    """
    Clean up answer formatting so the UI doesn't show ugly '\\n' sequences.
    For this project we always return a single plain-text paragraph (no newlines),
    so the front-end JSON never contains '\\n' inside the answer string.
    """
    if not answer:
        return answer

    # Normalize different newline styles, then collapse everything into one line
    text = answer.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _numbers_in_text(text: str) -> List[str]:
    """Extract all numeric tokens (years, amounts, etc.) from text."""
    if not text:
        return []
    return re.findall(r"\d[\d,]*", text)


def is_likely_hallucinated_numeric_answer(answer: str, context: str) -> bool:
    """
    Heuristic: if most of the numbers in the answer do NOT appear in the context,
    treat it as likely hallucinated (e.g., made‑up 5‑year tables).
    We only use this as a safety check; we do NOT change the core retrieval logic.
    """
    ans_nums = _numbers_in_text(answer)
    if not ans_nums:
        return False

    ctx = context or ""
    missing = 0
    for n in ans_nums:
        if n not in ctx:
            missing += 1

    # If more than half of the numbers are not present in the context, it's suspicious
    return missing / len(ans_nums) > 0.5


def extract_entities_from_query(query: str) -> Dict[str, List[str]]:
    """Extract potential entities (countries, products, dates, etc.) from query."""
    entities = {
        "locations": [],
        "products": [],
        "dates": [],
        "numbers": []
    }
    
    # Extract numbers
    numbers = re.findall(r'\d+', query)
    entities["numbers"] = numbers
    
    # Extract potential dates
    date_patterns = [
        r'\d{4}',  # Years
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(q[1-4]|quarter [1-4])\b'
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities["dates"].extend(matches)
    
    return entities


def answer_query(query: str, top_k: int = 8) -> Dict:
    """
    Main function to answer queries with advanced understanding.
    Handles all types of questions with improved accuracy.
    """
    retriever.search_kwargs.update({"k": top_k})
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return {
            "answer": "I couldn't find any relevant documents to answer your question. Please make sure your documents are properly indexed.",
            "sources": [],
            "query_type": "unknown"
        }

    # Classify the query
    query_type, query_metadata = classify_query(query)
    query_lower = query.lower()
    
    # Extract entities from query
    entities = extract_entities_from_query(query)
    
    # Dynamically extract all data from documents
    all_records = []  # List of dicts with all key-value pairs
    field_values = defaultdict(list)  # {field_name: [all values]}
    categorical_data = defaultdict(lambda: defaultdict(list))  # {category_field: {category_value: {metric_field: [values]}}}
    
    for doc in retrieved_docs:
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        lines = content.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            # Extract all key-value pairs from this line
            pairs = extract_key_value_pairs(line)
            if pairs:
                all_records.append(pairs)
                
                # Store all field values
                for key, value in pairs.items():
                    field_values[key].append(value)
                    
                    # Extract numeric value if present
                    numeric_val = extract_numeric_value(value)
                    
                    # For comparative queries, group by categorical fields
                    if query_metadata.get("is_comparative") and numeric_val is not None:
                        # Find categorical fields (non-numeric fields that might be grouping fields)
                        for cat_key, cat_value in pairs.items():
                            if cat_key != key and extract_numeric_value(cat_value) is None:
                                # This is a categorical field
                                if cat_value not in categorical_data[cat_key]:
                                    categorical_data[cat_key][cat_value] = defaultdict(list)
                                categorical_data[cat_key][cat_value][key].append(numeric_val)
    
    answer = ""
    confidence = "high"
    
    # Handle queries based on type
    if query_type == QueryType.COMPARATIVE:
        # Find the metric field (what we're comparing)
        metric_field = None
        metric_keywords = ['revenue', 'sales', 'profit', 'income', 'units', 'price', 'cost', 'amount', 'value', 'gross', 'net']
        for keyword in metric_keywords:
            if keyword in query_lower:
                for field in field_values.keys():
                    if find_matching_field(keyword, field):
                        metric_field = field
                        break
            if metric_field:
                break
        
        # If still not found, try all fields with numeric values
        if not metric_field:
            for field in field_values.keys():
                # Check if field has numeric values
                has_numeric = any(extract_numeric_value(v) is not None for v in field_values[field])
                if has_numeric:
                    metric_field = field
                    break
        
        # Find the category field (what we're grouping by)
        category_field = None
        category_keywords = ['country', 'product', 'region', 'category', 'type', 'segment', 'department', 'location']
        for keyword in category_keywords:
            if keyword in query_lower:
                for field in categorical_data.keys():
                    if find_matching_field(keyword, field):
                        category_field = field
                        break
            if category_field:
                break
        
        # If no specific category found, use first categorical field with data
        if not category_field and categorical_data:
            category_field = list(categorical_data.keys())[0]
        
        # Calculate totals by category
        if category_field and metric_field and category_field in categorical_data:
            category_totals = {}
            for cat_value, metrics in categorical_data[category_field].items():
                if metric_field in metrics:
                    category_totals[cat_value] = sum(metrics[metric_field])
            
            if category_totals:
                # Determine if highest or lowest
                is_highest = any(word in query_lower for word in ["highest", "top", "maximum", "max", "best", "most", "largest", "biggest"])
                if is_highest:
                    top_item = max(category_totals, key=category_totals.get)
                    value = category_totals[top_item]
                    # Format value appropriately
                    if abs(value) >= 1000:
                        formatted_value = f"${value:,.2f}"
                    else:
                        formatted_value = f"${value:.2f}"
                    answer = f"The {category_field.lower()} with the highest {metric_field.lower()} is **{top_item}** with {formatted_value}."
                else:
                    top_item = min(category_totals, key=category_totals.get)
                    value = category_totals[top_item]
                    if abs(value) >= 1000:
                        formatted_value = f"${value:,.2f}"
                    else:
                        formatted_value = f"${value:.2f}"
                    answer = f"The {category_field.lower()} with the lowest {metric_field.lower()} is **{top_item}** with {formatted_value}."
    
    elif query_type == QueryType.SPECIFIC_VALUE:
        # Check for queries asking for a specific value filtered by category
        for cat_keyword in ['country', 'product', 'region', 'segment', 'category', 'type']:
            if cat_keyword in query_lower:
                # Extract the category value from query
                for field in field_values.keys():
                    if find_matching_field(cat_keyword, field):
                        # Try to find the category value in the query
                        for cat_value in field_values[field]:
                            if cat_value.lower() in query_lower:
                                # Found a match, now find the metric
                                for metric_keyword in ['revenue', 'sales', 'profit', 'income', 'units', 'price', 'cost']:
                                    if metric_keyword in query_lower:
                                        for metric_field in field_values.keys():
                                            if find_matching_field(metric_keyword, metric_field):
                                                # Sum values for this category and metric
                                                total = 0
                                                count = 0
                                                for record in all_records:
                                                    if record.get(field, "").lower() == cat_value.lower():
                                                        metric_val = extract_numeric_value(record.get(metric_field, ""))
                                                        if metric_val is not None:
                                                            total += metric_val
                                                            count += 1
                                                if count > 0:
                                                    if abs(total) >= 1000:
                                                        formatted_total = f"${total:,.2f}"
                                                    else:
                                                        formatted_total = f"${total:.2f}"
                                                    answer = f"The {metric_field.lower()} for {cat_value} is {formatted_total} (based on {count} record{'s' if count > 1 else ''})."
                                                    break
                                        if answer:
                                            break
                                if answer:
                                    break
                        if answer:
                            break
                if answer:
                    break
    
    elif query_type == QueryType.COUNT:
        count_keywords = ['how many', 'count', 'number of', 'total number']
        if any(kw in query_lower for kw in count_keywords):
            for cat_keyword in ['country', 'product', 'region', 'segment', 'item', 'category', 'type']:
                if cat_keyword in query_lower:
                    for field in field_values.keys():
                        if find_matching_field(cat_keyword, field):
                            unique_values = set(field_values[field])
                            count = len(unique_values)
                            answer = f"There are **{count}** unique {field.lower()}{'s' if count != 1 else ''} in the dataset."
                            break
                    if answer:
                        break
    
    elif query_type == QueryType.TOTAL:
        # Find relevant metric fields
        metric_keywords = ['revenue', 'sales', 'profit', 'income', 'units', 'price', 'cost', 'amount']
        for keyword in metric_keywords:
            if keyword in query_lower:
                for field in field_values.keys():
                    if find_matching_field(keyword, field):
                        # Sum all numeric values for this field
                        total = 0
                        count = 0
                        for value in field_values[field]:
                            num_val = extract_numeric_value(value)
                            if num_val is not None:
                                total += num_val
                                count += 1
                        
                        if count > 0:
                            if abs(total) >= 1000:
                                formatted_total = f"${total:,.2f}"
                            else:
                                formatted_total = f"${total:.2f}"
                            answer = f"The total {field.lower()} is {formatted_total} (from {count} record{'s' if count > 1 else ''})."
                            break
                if answer:
                    break
    
    elif query_type == QueryType.LIST:
        # Try to list items from a specific field
        for cat_keyword in ['country', 'product', 'region', 'segment', 'item', 'category']:
            if cat_keyword in query_lower:
                for field in field_values.keys():
                    if find_matching_field(cat_keyword, field):
                        unique_values = sorted(set(field_values[field]))
                        if unique_values:
                            if len(unique_values) <= 20:
                                items_str = ", ".join(unique_values)
                                answer = f"The {field.lower()}s in the dataset are: {items_str}."
                            else:
                                items_str = ", ".join(unique_values[:20])
                                answer = f"Here are the first 20 {field.lower()}s: {items_str}. (Total: {len(unique_values)})"
                            break
                if answer:
                    break
    
    # Generic fallback - extract key information from documents
    if not answer and all_records:
        first_record = all_records[0]
        # Get first few key-value pairs
        key_parts = []
        for i, (key, value) in enumerate(list(first_record.items())[:3]):
            # Truncate long values
            if len(value) > 50:
                value = value[:47] + "..."
            key_parts.append(f"{key}: {value}")
        
        if key_parts:
            answer = " | ".join(key_parts)
            confidence = "medium"
        else:
            answer = "Relevant information found. Check sources for details."
            confidence = "low"
    elif not answer:
        answer = "Relevant information found. Check sources for details."
        confidence = "low"

    # Use LLM for better answers if:
    # 1. Answer is generic/poor quality
    # 2. Query type suggests need for reasoning (analytical, causal, factual)
    # 3. No structured answer was found
    use_llm = (
        is_generic_answer(answer) or 
        not answer or 
        query_type in [QueryType.ANALYTICAL, QueryType.CAUSAL, QueryType.FACTUAL, QueryType.TEMPORAL] or
        confidence == "low"
    )
    
    if use_llm:
        # Prepare context from retrieved documents
        context = format_docs(retrieved_docs[:min(8, len(retrieved_docs))])
        llm_answer = generate_llm_answer(query, context, query_type)
        if llm_answer and not is_generic_answer(llm_answer) and len(llm_answer) > 10:
            answer = llm_answer
            confidence = "high" if query_type in [QueryType.ANALYTICAL, QueryType.CAUSAL, QueryType.FACTUAL] else "medium"
        # If LLM also fails, keep the original answer
    # Final cleanup of answer formatting (avoid ugly '\n' in UI)
    answer = normalize_answer_format(answer)

    # Prepare sources
    source_list = []
    for sd in retrieved_docs:
        meta = sd.metadata or {}
        source_list.append({
            "id": meta.get("file_hash", "") + f"_{meta.get('chunk_index', '?')}",
            "metadata": meta,
            "text_snippet": sd.page_content[:500]
        })

    return {
        "answer": answer,
        "sources": source_list,
        "query_type": query_type.value,
        "confidence": confidence
    }




