# ## Data Ingestion

### Document Structure

from langchain_core.documents import Document

doc = Document(
    page_content = "this is the main text content I am using to create RAG",
    metadata = {
        "source":"example.txt",
        "pages":1,
        "author":"Rohit Koshy",
        "date_created":"2025-10-27"
    }
)

## Create a simple txt file

import os
os.makedirs("../data/text_files", exist_ok=True)


sample_texts = {
    "../data/text_files/python_intro.txt":
    
    """Python Programming Introduction
    
    Python is a high-level, interpreted programming language known for its simplicity and readability.
    Created by Guido van Rossum and first released in 1991, Python has become one of the most 
    popular programming languages in the world.
    
    Key Features:
    - Easy to learn and use
    - Extensive standard library
    - Cross-platform compatibility
    - Strong community support
    
    Python is widely used in web development, data science, artificial intelligence, and automation.
    """,

    "../data/text_files/machine_learning.txt":
    """Machine Learning Basics
    
    Machine learning is a subset of artificial intelligence that enables systems to learn and improve 
    from experience without being explicitly programmed. It focuses on developing computer programs 
    that can access data and use it to learn for themselves.

    Types of Machine Learning:    
    1. Supervised Learning: Learning with labeled data
    2. Unsupervised Learning: Finding patterns in unlabeled data
    3. Reinforcement Learning: Learning through rewards and penalties

    Applications include image recognition, speech processing, and recommendation systems.
    """
}

sample_texts

type(sample_texts)

for filepath, content in sample_texts.items():
    with open(filepath,'w',encoding="utf-8") as f:
        f.write(content)

print("Sample text files created!")

sample_texts.items()

## Text loader

from langchain_community.document_loaders import TextLoader

loader = TextLoader("../data/text_files/python_intro.txt", encoding = "utf-8")
document = loader.load()
print(document)

## Directory Loader

from langchain_community.document_loaders import DirectoryLoader

# load all text files from the directory

dir_loader = DirectoryLoader(
    "../data/text_files",
    glob = "**/*.txt", # Pattern to match files
    loader_cls = TextLoader, #loader class to use
    loader_kwargs={'encoding':'utf-8'},
    show_progress=False
)

documents = dir_loader.load()
documents

## Directory Loader for the PDFs

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

# load all text files from the directory

dir_loader = DirectoryLoader(
    "../data/pdf",
    glob = "**/*.pdf", # Pattern to match files
    loader_cls = PyMuPDFLoader, #loader class to use
    show_progress=False
)

pdf_documents = dir_loader.load()
pdf_documents

type(pdf_documents[0])

"""Split documents into smaller chunks for better RAG performance"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_size = 1000
chunk_overlap = 200

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    length_function = len,
    separators = ["\n\n","\n"," ",""]
)

split_docs = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

chunks = text_splitter.split_documents(pdf_documents)
chunks

# ### Embdeddings and Vector Store DB

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List,Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Sentence Transformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}:{e}")
            raise

    def generate_embeddings(self,texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """

        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)}texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

## Initialize the embedding manager

embedding_manager = EmbeddingManager()
embedding_manager

# ### Vector Store

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""

        try:
            # Create persistent Chroma DB client
            os.makedirs(self.persist_directory, exist_ok = True)
            self.client = chromadb.PersistentClient(path = self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {"description":"PDF document embeddings for RAG"}
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings:np.ndarray):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the number of embeddings")
        
        print(f"Adding {len(documents)} documents to the vector store...")

        # Prepare data for Chroma DB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc,embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embeddings
            embeddings_list.append(embedding.tolist())

    # Add to collection

        try:
            self.collection.add(
                ids=ids,
                embeddings = embeddings_list,
                metadatas=metadatas,
                documents = documents_text
            )

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

vectorstore = VectorStore()
vectorstore

chunks

### Convert the text to embeddings

texts = [doc.page_content for doc in chunks]
texts

## Generate the embeddings

embeddings = embedding_manager.generate_embeddings(texts)


## Store into the vector database

vectorstore.add_documents(chunks, embeddings)

# ### Retriever Pipeline From VectorStore

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store:VectorStore, embedding_manager: EmbeddingManager):
        """Initialize the retriever
        
        Args:
            vector_store: Vector store containing the embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query:str, top_k:int = 5, score_threshold: float=0.0)->List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """

        print(f"Retrieving documents for query: {query}")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results = top_k
            )

            # Process results

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids,documents,metadatas,distances)):
                    # Convert distance into similarity score (ChromaDB uses consine distance)
                    similarity_score = 1-distance

                    if similarity_score>= score_threshold:
                        retrieved_docs.append({
                            'id':doc_id,
                            'content':document,
                            'metadata':metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i+1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

            else:
                print("No documents found")

            return retrieved_docs
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        
rag_retriever = RAGRetriever(vectorstore,embedding_manager)

rag_retriever

rag_retriever.retrieve("What is refrain in chorus?")

# ### Integration VectorDB Context pipeline with LLM Output

### Simple RAG pipeline with the OpenAI LLM

from langchain_openai import ChatOpenAI
#openai = ChatOpenAI(model_name="gpt-3.5-turbo")

import os
from dotenv import load_dotenv
load_dotenv()

## Initialize the OpenAI LLM (set your OPENAI_API_KEY in environment)

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
    api_key=openai_api_key
)

# Simple RAG function: retrieve context + generate response

def rag_simple(query, retriever,llm,top_k=3):
    # Retrieve the context
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."
    
    # generate the answer using OpenAI LLM
    prompt = f"""Use the following context to answer the question concisely.
            Context:
            {context}

            Question:{query}
            Answer:"""
    
    response = llm.invoke([prompt.format(context=context, query = query)])
    return response.content

answer = rag_simple("What is chorus?", rag_retriever, llm)
print(answer)

# This should give the answer if you have sufficient credits in OpenAI. 
# Note: It doesn't work with $0 in credits


