import os
import shutil
from dotenv import load_dotenv

# --- Updated Imports for HuggingFace Embeddings ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document

# --- Configuration ---
load_dotenv()

# Define paths and constants
DATA_PATH = "../knowledge_base/docs" 
VECTOR_DB_PATH = "../knowledge_base/vector_db"
# Using HuggingFace embedding model (FREE!)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents():
    """Loads PDF documents from the DATA_PATH directory."""
    print(f"Loading documents from {DATA_PATH}...")
    try:
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
        print(f"Found and loaded {len(documents)} document pages.")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def split_documents(documents: list[Document]):
    """Splits documents into smaller, overlapping chunks."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total number of chunks created: {len(chunks)}")
    return chunks

def add_to_chroma(chunks: list[Document]):
    """
    Initializes the embedding function and stores the document chunks
    in the Chroma vector database.
    """
    # 1. Initialize the HuggingFace Embeddings (FREE!)
    print(f"Initializing HuggingFace Embeddings with model: {EMBEDDING_MODEL_NAME}...")
    print("(This may take a moment on first run as it downloads the model...)")
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
    except Exception as e:
        print(f"Error initializing HuggingFace Embeddings: {e}")
        return

    # 2. Clean up previous vector store data
    if os.path.exists(VECTOR_DB_PATH):
        print(f"Removing existing vector store at {VECTOR_DB_PATH}...")
        shutil.rmtree(VECTOR_DB_PATH)
    
    # 3. Create a new Chroma instance and persist the data
    print(f"Creating new Chroma store and embedding {len(chunks)} chunks...")
    try:
        vectorstore = Chroma.from_documents(
            chunks, 
            embedding_function, 
            persist_directory=VECTOR_DB_PATH
        )
        print(f"✅ Indexing complete! Vector store saved at {VECTOR_DB_PATH}")
    except Exception as e:
        print(f"Error during Chroma vector store creation: {e}")

def main():
    """Main function to run the document processing pipeline."""
    print("🚀 Starting document processing pipeline...")
    print("=" * 60)
    
    documents = load_documents()
    if not documents:
        print("❌ No documents found. Please ensure your PDFs are in the 'knowledge_base/docs' folder.")
        return

    chunks = split_documents(documents)
    
    if chunks:
        add_to_chroma(chunks)
        print("=" * 60)
        print("✅ All done! Your knowledge base is ready to use.")
    else:
        print("❌ No chunks were generated. Stopping process.")

if __name__ == "__main__":
    main()