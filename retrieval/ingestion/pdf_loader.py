import os
import re
import pandas as pd
from retrieval.ingestion.chunker import chunk_text
from retrieval.ingestion.embedder import embed_and_store
from retrieval.vector_store.chroma_client import get_chroma_client
from dotenv import load_dotenv

load_dotenv()


def read_pdf_with_metadata(file_path):
    """Reads PDF and extracts text with page numbers and metadata."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        
        pages_data = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_data.append({
                    "text": page_text,
                    "page_num": page_num,
                    "total_pages": len(reader.pages)
                })
        return pages_data
    except Exception as e:
        print(f"[PDF ERROR] pypdf failed: {e}")
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            pages_data = []
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_data.append({
                        "text": page_text,
                        "page_num": page_num,
                        "total_pages": len(reader.pages)
                    })
            return pages_data
        except Exception as e2:
            print(f"[PDF ERROR] All loaders failed: {e2}")
            raise Exception(f"Could not read PDF: {e2}")


def extract_fiscal_year(text):
    """Extract fiscal year from text."""
    print(f"\n{'='*80}")
    print(f"🔍 SEARCHING FOR FISCAL YEAR")
    print(f"{'='*80}")
    print(f"First 300 characters:")
    print(repr(text[:300]))
    print(f"{'='*80}\n")
    
    patterns = [
        (r'(\d{4})-(\d{2})\b', 'YYYY-YY'),
        (r'(\d{4})\s*-\s*(\d{2})', 'YYYY - YY'),
        (r'FY\s*(\d{4})', 'FY YYYY'),
    ]
    
    for pattern, desc in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            print(f"✅ MATCH: {desc} → {match.group(0)}")
            if len(match.groups()) > 1:
                year1 = match.group(1)
                result = str(int(year1) + 1)
                print(f"   Result: FY {result}\n")
                return result
            else:
                result = match.group(1)
                print(f"   Result: FY {result}\n")
                return result
    
    print(f"❌ NO MATCH\n")
    return None


def detect_document_fiscal_year(pages_data):
    """Scan first 10 pages for fiscal year."""
    print(f"\n🔍 SCANNING FIRST 10 PAGES\n")
    
    pages_to_check = pages_data[:min(10, len(pages_data))]
    fiscal_years_found = []
    
    for i, page in enumerate(pages_to_check, 1):
        print(f"--- PAGE {i} ---")
        fy = extract_fiscal_year(page["text"])
        if fy:
            fiscal_years_found.append(fy)
    
    if fiscal_years_found:
        most_common = max(set(fiscal_years_found), key=fiscal_years_found.count)
        print(f"\n📅 DOCUMENT FISCAL YEAR: {most_common}\n")
        return most_common
    
    print(f"\n📅 DOCUMENT FISCAL YEAR: None\n")
    return None


def detect_section(text):
    """Detect document section."""
    text_lower = text.lower()
    
    if any(k in text_lower for k in ['revenue by geography', 'geographical revenue']):
        return "Revenue by Geography"
    elif any(k in text_lower for k in ['income statement', 'profit and loss']):
        return "Income Statement"
    elif any(k in text_lower for k in ['balance sheet']):
        return "Balance Sheet"
    elif any(k in text_lower for k in ['cash flow']):
        return "Cash Flow Statement"
    elif any(k in text_lower for k in ['risk factor', 'risks']):
        return "Risk Factors"
    
    return "General"


def read_excel(file_path):
    """Reads Excel file."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        df = pd.read_excel(file_path)
    
    if df.empty:
        return ""
    
    text = ""
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
        text += row_text + "\n"
    return text


def read_csv(file_path):
    """Reads CSV file."""
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, encoding='latin-1')
    
    if df.empty:
        return ""
    
    text = ""
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
        text += row_text + "\n"
    return text


def ingest_user_file(file_path, user_id):
    """Ingest uploaded file with metadata."""
    ext = file_path.lower()
    filename = os.path.basename(file_path)

    if ext.endswith(".pdf"):
        pages_data = read_pdf_with_metadata(file_path)
        
        if not pages_data:
            raise Exception("PDF is empty!")
        
        document_fiscal_year = detect_document_fiscal_year(pages_data)
        all_chunks_with_metadata = []
        
        for page_data in pages_data:
            page_text = page_data["text"]
            page_num = page_data["page_num"]
            section = detect_section(page_text)
            fiscal_year = document_fiscal_year
            
            chunks = chunk_text(page_text, chunk_size=1000, chunk_overlap=200)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = {
                    "source": filename,
                    "page_num": page_num,
                    "total_pages": page_data["total_pages"],
                    "section": section,
                    "chunk_index_in_page": chunk_idx
                }
                
                if fiscal_year:
                    chunk_metadata["fiscal_year"] = fiscal_year
                
                all_chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })
        
        print(f"\n📊 Sample of first 3 chunks:")
        for i, item in enumerate(all_chunks_with_metadata[:3], 1):
            meta = item['metadata']
            print(f"Chunk {i}: FY={meta.get('fiscal_year', 'None')}, Page={meta.get('page_num')}")
        
        client = get_chroma_client(collection_name=f"user_{user_id}")
        chunks_only = [item["text"] for item in all_chunks_with_metadata]
        metadatas = [item["metadata"] for item in all_chunks_with_metadata]
        
        embed_and_store(chunks_only, metadata_list=metadatas, client=client)
        
        return {
            "message": "PDF ingested successfully!",
            "chunks": len(all_chunks_with_metadata),
            "collection": f"user_{user_id}",
            "pages_processed": len(pages_data),
            "fiscal_year": document_fiscal_year
        }
    
    elif ext.endswith((".xlsx", ".xls")):
        text = read_excel(file_path)
    elif ext.endswith(".csv"):
        text = read_csv(file_path)
    else:
        raise Exception("Unsupported file type!")

    if not text or len(text.strip()) == 0:
        raise Exception("File is empty!")

    chunks = chunk_text(text)
    client = get_chroma_client(collection_name=f"user_{user_id}")
    embed_and_store(chunks, metadata={"source": filename}, client=client)

    return {
        "message": "File ingested successfully!",
        "chunks": len(chunks),
        "collection": f"user_{user_id}"
    }