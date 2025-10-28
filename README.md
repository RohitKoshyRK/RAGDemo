**RAG Demo (Retrieval-Augmented Generation)
**
This project demonstrates a simple end-to-end Retrieval-Augmented Generation (RAG) pipeline in Python.
It loads documents, chunks them, generates embeddings, stores them in a vector database, retrieves relevant context based on a user query, and finally uses an LLM to answer using that context.

**Features**

- Load text/PDF documents from a directory
- Split documents into semantic chunks
- Generate embeddings using SentenceTransformer
- Persist embeddings using ChromaDB
- Perform similarity search (vector retrieval)
- Feed retrieved context into a Large Language Model (LLM)
- Return grounded, context-aware answers

**Tech Stack
**
- Embeddings: sentence-transformers
- Vector Store: chromadb
- LLM: langchain-openai
- Document Loading: langchain_community
- Text Splitting: RecursiveCharacterTextSplitter

If you don’t have a requirements file yet, create one later with: pip freeze > requirements.txt
