import os
import shutil
from dotenv import load_dotenv
# --- Updated Imports for Google Generative AI ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
# CORRECTED IMPORT: Moved from 'langchain.text_splitter' to 'langchain_text_splitters'
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# CORRECTED IMPORT: Document is now imported from langchain_core.documents
from langchain_core.documents import Document

# --- Configuration ---
load_dotenv()
# Check for the correct environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define paths and constants
# UPDATED PATH: Now points to the directory where download_data.py stores the PDFs
DATA_PATH = "knowledge_base/docs" 
VECTOR_DB_PATH = "knowledge_base/vector_db"
# Using the desired Gemini Embedding Model
EMBEDDING_MODEL_NAME = "text-embedding-004" 

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please make sure you have a .env file with your key set as GEMINI_API_KEY.")
    exit()

def load_documents():
    """Loads PDF documents from the DATA_PATH directory."""
    print(f"Loading documents from {DATA_PATH}...")
    try:
        # PyPDFDirectoryLoader will recursively search for PDFs in all subdirectories of DATA_PATH
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
    # 1. Initialize the Google Generative AI Embeddings
    print(f"Initializing Gemini Embeddings with model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=GEMINI_API_KEY
        )
    except Exception as e:
        print(f"Error initializing Gemini Embeddings: {e}")
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
        vectorstore.persist()
        print(f"Indexing complete! Vector store saved at {VECTOR_DB_PATH}")
    except Exception as e:
        print(f"Error during Chroma vector store creation: {e}")

def main():
    """Main function to run the document processing pipeline."""
    documents = load_documents()
    if not documents:
        # Updated prompt for the user
        print("No documents found. Please ensure your PDFs are in the 'knowledge_base/docs' folder.")
        return

    chunks = split_documents(documents)
    
    if chunks:
        add_to_chroma(chunks)
    else:
        print("No chunks were generated. Stopping process.")

if __name__ == "__main__":
    main()