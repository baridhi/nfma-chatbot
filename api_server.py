#!/usr/bin/env python3
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

app = Flask(__name__)
CORS(app)

# Configuration
DB_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
# Change "llama3" below to match the model name you saw in 'ollama list'
LLM_MODEL = "llama3" 

print("Loading local vector database...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
print("Vector store connected successfully.")

def build_sources(docs):
    by_file = {}
    for doc in docs:
        fname = os.path.basename(doc.metadata.get("source", "Unknown"))
        entry = by_file.setdefault(fname, {"file": fname, "pages": []})
        page = doc.metadata.get("page")
        if isinstance(page, int) and (page + 1) not in entry["pages"]:
            entry["pages"].append(page + 1)   # PyPDFLoader is 0-indexed
    for entry in by_file.values():
        entry["pages"].sort()
    return list(by_file.values())


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    question = data.get("question", "")
    # Extract the model from the request, fallback to default if not provided
    # selected_model = data.get("model", LLM_MODEL)
    selected_model = data.get("model") or LLM_MODEL #to allow for default
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # 1. Retrieve the top 3 closest text snippets
        docs = db.similarity_search(question, k=3)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        # sources = list(set([os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs])) #older
        sources = build_sources(docs) #with source links via json

        # 2. Build the RAG Prompt context
        prompt = f"""You are a helpful research assistant. Answer the user's question accurately using ONLY the provided context...
Context:
{context_text}

Question: {question}
Answer:"""

        # 3. Stream the prompt to the selected model
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": selected_model, # Use the dynamic model variable here
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        
        ollama_response = response.json()
        answer = ollama_response.get("response", "Error: Empty generation string.")

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        print(f"Error handling request: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

@app.route("/health", methods=["GET"])
def health():
    # Return the currently configured default model
    return jsonify({"status": "ok", "model": LLM_MODEL})
    
if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 5001))
    print(f"Starting production-ready RAG API server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
