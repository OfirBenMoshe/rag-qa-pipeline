## 🧠 Retrieval-Augmented QA Pipeline (RAG) 
## Using FAISS + Azure OpenAI

This project implements a fully local **RAG system** that:
- Extracts structured chunks from `.docx` documents
- Indexes them using FAISS Vector Database
- Re-ranks results using a cross-encoder
- Generates answers using Azure OpenAI (GPT-35-Turbo)
- Supports a Gradio UI for interactive querying

---

## 🚀 Setup Instructions

### ✅ Step 1: Set Up Python Environment (Python 3.12.3)

a. Create and activate a virtual environment:
python -m venv venv

b. Activate the venv:
source venv/bin/activate  # Or venv\\Scripts\\activate on Windows

c. allow local script execution:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

d. Then activate again:
source venv/bin/activate  # Or venv\\Scripts\\activate on Windows

### ✅ Step 2: Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### ✅ Step 3: Install Other Dependencies
pip install -r requirements.txt

### 🧩 Project Structure
.
├── faiss_client.py        # Handles .docx parsing, chunking, and FAISS indexing
├── qa_pipeline.py         # Full RAG pipeline: retrieval, reranking, prompt, and LLM answer
├── gradio_app.py          # Gradio interface for querying the system
├── word_files/            # Folder with your input Word documents (*.docx)
├── faiss_index.index      # Saved FAISS vector index (auto-generated)
├── faiss_metadata.json    # Metadata for each indexed chunk (auto-generated)
└── requirements.txt



## 🛠️ Running the System

### 🔹 1. Build the FAISS Index
#### Run the FAISS indexer to process your .docx files and save the index:
RUN: python faiss_client.py

This will:
Chunk all .docx files in ./word_files/
Create faiss_index.index and faiss_metadata.json

### 🔹 3. Launch the Gradio UI
RUN: python gradio_app.py
