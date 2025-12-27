import os
import sys
from dotenv import load_dotenv
import pandas as pd

# Add the project root to sys.path to ensure absolute imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from src.model import llm
from src.dataset_filter import get_dataset_metadata

load_dotenv()

# Configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "dataset.csv")
vector_db_path = os.path.join(base_dir, "vector_db")

if not os.path.exists(csv_path):
    print(f"Error: Dataset not found at {csv_path}")
    sys.exit(1)

def initialize_vector_store():
    """
    Initialize a persistent FAISS vector store with semantic chunking.
    """
    print("--- Initializing Semantic Vector Store ---")
    
    # 1. Initialize Embeddings
    print("Loading FastEmbed Embeddings...")
    cache_dir = os.path.join(base_dir, ".fastembed_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    embeddings = FastEmbedEmbeddings(cache_dir=cache_dir)
    
    # 2. Check if vector store already exists
    if os.path.exists(os.path.join(vector_db_path, "index.faiss")):
        print("Loading existing vector store from disk...")
        return FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    
    # 3. Load and Chunk Data
    print("Loading CSV data and metadata...")
    df = pd.read_csv(csv_path)
    metadata_str = get_dataset_metadata(df)
    
    loader = CSVLoader(csv_path)
    documents = loader.load()
    
    print("Performing Semantic Chunking...")
    # SemanticChunker uses the embeddings to find breakpoints between semantically different sentences
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95  # Adjust as needed for sensitivity
    )
    chunks = text_splitter.split_documents(documents)
    
    # Append Metadata Document
    chunks.append(Document(page_content=metadata_str, metadata={"source": "dataset_metadata"}))
    
    print(f"Created {len(chunks)} semantic chunks (including metadata) from {len(documents)} rows.")
    
    # 4. Create and Persist Vector Store
    print("Creating FAISS index (this may take a moment)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving vector store to {vector_db_path}...")
    vector_store.save_local(vector_db_path)
    
    return vector_store

def main():
    # Initialize components
    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 100})
    
    # Create the RetrievalQA chain
    # Traditional chains like RetrievalQA use a specific template format
    template = """You are an assistant for question-answering tasks focusing on dataset analysis. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise and professional.

Context: {context}

Question: {question}
Answer:"""
    
    QA_CHAIN_PROMPT = ChatPromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    print("\n--- Semantic Dataset Chat Session ---")
    print("Type 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Ending session. Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            print("Assistant parsing dataset...")
            # Traditional chains use .invoke or __call__
            response = qa_chain.invoke({"query": user_input})
            
            print(f"\nAssistant: {response['result']}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
