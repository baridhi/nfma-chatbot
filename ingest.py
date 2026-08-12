#!/usr/bin/env python3
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration paths
DOCS_DIR = Path("./source_documents")
DB_DIR = Path("./chroma_db")
EMBED_MODEL = "nomic-embed-text"

def load_documents():
    documents = []
    for file_path in DOCS_DIR.iterdir():
        if file_path.suffix == ".txt":
            print(f"Loading text file: {file_path.name}")
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
        elif file_path.suffix == ".pdf":
            print(f"Loading PDF file: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
    return documents

def main():
    # 1. Gather all documents from the source folder
    raw_docs = load_documents()
    if not raw_docs:
        print("No documents found in source_documents/. Exiting.")
        return

    # 2. Chunk text into overlapping segments to keep context intact
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Split {len(raw_docs)} documents into {len(chunks)} structural chunks.")

    # 3. Spin up local embedding connection to Ollama
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 4. Generate embeddings and save them to the persistent local disk store
    print(f"Generating vectors and initializing database at '{DB_DIR}'...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )
    print("Database built and saved successfully!")

if __name__ == "__main__":
    main()