# main.py

### =============================================================
# Parsing files
### =============================================================
import os
from langchain_community.document_loaders import PyPDFLoader

# Path to folder containing cardiology PDFs
pdf_folder = "articles"

documents = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        docs = loader.load()
        documents.extend(docs)

print(f"Loaded {len(documents)} pages from {len(os.listdir(pdf_folder))} PDF files.")

### =============================================================
# Chunking
### =============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

print(f"Total chunks: {len(docs)}")

### =============================================================
# Creating a Vector Database
### =============================================================

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"
COLLECTION = "cardiology"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# If directory exists and is not empty, just load.
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,   # on __init__, use embedding_function=
        collection_name=COLLECTION
    )
else:
    # First indexing from `docs`
    db = Chroma.from_documents(
        docs,
        embedding=embeddings,            # on from_documents, name is embedding=
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION
    )

### =============================================================
# Build the generation pipeline
### =============================================================

import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = 0 if torch.cuda.is_available() else -1  # GPU se disponível

generator = pipeline(
    task="text-generation",                # decoder-only models use text-generation
    model=MODEL_NAME,
    device=DEVICE,                         # 0 with CUDA, -1 for CPU
    do_sample=True,      # Enable sampling
    temperature=0.7,     # Balance creativity
    top_p=0.9,           # Nucleus sampling for diversity
    top_k=50,            # Or use top_p alone
    repetition_penalty=1.05,
    max_new_tokens=512,
)

# Important: LangChain expects just the generated completion; avoid echoing prompt
local_llm = HuggingFacePipeline(pipeline=generator)

### =============================================================
# Wire the LLM into a RAG chain (LCEL, sem deprecations)
### =============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1) Build the retriever from the vector store (top-4 parts)
retriever = db.as_retriever(search_kwargs={"k": 4})

# 2) Prompt NEEDS to have context (where the recovered parets are "stuffed")
prompt = ChatPromptTemplate.from_template(
    "Answer ONLY with facts from the context below.\n\n{context}\n\nQuestion: {input}"
)

# 3) Chain combines docs + LLM (similar to old 'stuff')
doc_chain = create_stuff_documents_chain(local_llm, prompt)

# 4) Retrieval chain : retriever -> doc_chain
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

# 5) Consultation (use .invoke at LangChain >= 0.1)
query = "According to these articles, summarize some heart diseases."
res = retrieval_chain.invoke({"input": query})

print("\n=== Answer ===\n", res["answer"])
print("\n=== Sources ===")
for i, d in enumerate(res["context"], 1):
    print(f"[{i}] {d.metadata.get('source','?')} | p.{d.metadata.get('page','?')}")
