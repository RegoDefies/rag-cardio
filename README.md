# 🫀 Cardiology RAG – Retrieval-Augmented Generation with PDFs

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG) pipeline** for question answering on a collection of **cardiology research articles in PDF format**.  

The system uses **LangChain**, **ChromaDB**, and a **local Hugging Face model** (Qwen-7B-Instruct by default) to retrieve relevant text chunks and generate grounded answers.

---

## ✨ Features

- Parse multiple PDF documents (e.g., cardiology articles)
- Split text into chunks for efficient retrieval
- Store embeddings in a persistent **ChromaDB vector database**
- Run a **local LLM** (no API key required)
- Answer questions with **sources included** for transparency

---

## 📂 Project Structure

```
cardiology_rag/
│── articles/                # Folder with cardiology PDFs
│    ├── atrial_fibrillation.pdf
│    ├── heart_failure.pdf
│    └── coronary_disease.pdf
│── chroma_db/               # Vector DB (auto-created)
│── main.py                  # Main RAG pipeline script
```

---

## ⚙️ Installation

Create a clean environment and install dependencies:

```bash
# Clone repository
git clone https://github.com/yourusername/cardiology_rag.git
cd cardiology_rag

# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate   # Linux / Mac
rag_env\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, install manually:

```bash
pip install langchain langchain-community langchain-huggingface langchain-chroma chromadb pypdf sentence-transformers transformers torch accelerate
```

---

## 🚀 Usage

1. Place your cardiology PDFs inside the `articles/` folder.
2. Run the main script:

```bash
python main.py
```

Example output:

```
Loaded 90 pages from 10 PDF files.
Total chunks: 634

=== Answer ===
Atrial fibrillation can cause stroke, heart failure, and increased mortality.

=== Sources ===
[1] atrial_fibrillation.pdf | p.12
[2] heart_failure.pdf       | p.4
```

---

## 🔧 Customization

- **Change model:**  
  Default is `Qwen/Qwen2.5-7B-Instruct`. You can try smaller models like `google/flan-t5-base` or larger ones like `mistralai/Mistral-7B-Instruct-v0.2`.

- **Adjust retrieval:**  
  Modify `k` in `db.as_retriever(search_kwargs={"k": 4})` to return more or fewer chunks.

- **Expand knowledge base:**  
  Add more PDFs (guidelines, textbooks, papers) into `articles/`.

---

## 📌 Applications

- Medical students preparing for exams  
- Clinicians reviewing updated guidelines  
- Researchers summarizing across papers  
- Startups building intelligent assistants  

⚠️ **Disclaimer:** This project is for **educational purposes only**. Always verify medical information with trusted clinical sources before applying it in practice.

---

## 📜 License

MIT License – feel free to use, modify, and share.
