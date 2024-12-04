from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List
from langchain_core.vectorstores import InMemoryVectorStore
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader("C:\\Users\\msaun\\Downloads\\Pathfinder - Core Rulebook (5th Printing).pdf")
docs = loader.load()

text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_split.split_documents(docs)

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vector store
vector_store = InMemoryVectorStore(embedding=embedding_model)
doc_ids = vector_store.add_documents(documents=chunks)

print(doc_ids[:3])

