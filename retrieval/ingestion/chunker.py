# ingestion/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Accepts a list of LangChain Document objects and returns a list of chunks (Document objects).
    You can tune chunk_size & chunk_overlap for your use case.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Chunks plain text into smaller pieces.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)




