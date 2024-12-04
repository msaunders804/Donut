import os
import pickle
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.schema import Document

# Define file paths
pdf_path = "C:\\Users\\msaun\\Downloads\\Pathfinder - Core Rulebook (5th Printing).pdf"
chunks_file = "pathfinder_chunks.pkl"
vector_store_file = "faiss_pathfinder_index"

# Step 1: Load or Split PDF into Chunks
if os.path.exists(chunks_file):
    print("Loading pre-saved chunks...")
    with open(chunks_file, "rb") as f:
        all_splits = pickle.load(f)
else:
    print("Splitting the PDF into chunks...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Save chunks for reuse
    with open(chunks_file, "wb") as f:
        pickle.dump(all_splits, f)
    print("Chunks saved to disk.")

# Step 2: Initialize MiniLM for Embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = SentenceTransformerEmbeddings(model_name=embedding_model_name)

# Step 3: Create or Load Vector Store
if os.path.exists(f"{vector_store_file}"):
    print("Loading pre-existing vector store...")
    vector_store = FAISS.load_local(vector_store_file, embeddings=embedding, allow_dangerous_deserialization=True)
else:
    print("Creating a new vector store...")
    documents = [Document(page_content=chunk.page_content) for chunk in all_splits]
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding
    )
    vector_store.save_local(vector_store_file)
    print("Vector store saved to disk.")

# Step 4: Define Retrieval Function
def retrieve_context(question, k=5):
    """
    Retrieve the top-k relevant chunks from the vector store for a given question.
    """
    print(f"Retrieving top-{k} documents for question: {question}")
    retrieved_docs = vector_store.similarity_search(question, k=k)
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Step 5: Initialize Qwen2 for Text Generation
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
generator = pipeline("text-generation", model=model_name)

def generate_answer(question, context, max_new_tokens=200):
    """
    Generate an answer using Qwen2 with the retrieved context.
    """
    max_input_length = 8192 - max_new_tokens
    truncated_context = context[-max_input_length:]
    prompt = f"Context:\n{truncated_context}\n\nQuestion: {question}\n\nAnswer:"
    print(f"Prompt:\n{prompt}\n")
    response = generator(prompt,
                         max_new_tokens=max_new_tokens,
                         num_return_sequences=1,
                         pad_token_id=generator.model.config.eos_token_id)
    return response[0]["generated_text"]

# Step 6: Test the RAG Pipeline
question = "What are the types of flanking in Pathfinder?"
context = retrieve_context(question, k=5)
answer = generate_answer(question, context)

print("\nGenerated Answer:\n", answer)
