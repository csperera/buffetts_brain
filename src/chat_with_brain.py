import os
from dotenv import load_dotenv
import time

# --- CORRECTED LangChain Imports ---
# Imports for Prompts, Output Parsers, and Runnables are now from langchain_core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Imports for Google Generative AI components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define paths and constants (must match process_documents.py)
VECTOR_DB_PATH = "knowledge_base/vector_db"
EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATION_MODEL_NAME = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    exit()

def setup_rag_chain():
    """
    Sets up the Retrieval-Augmented Generation (RAG) chain.
    """
    print("--- Setting up RAG System ---")

    # 1. Initialize Embeddings (must match the model used for creating the store)
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GEMINI_API_KEY
    )
    
    # 2. Load the Vector Store
    print(f"Loading vector store from {VECTOR_DB_PATH}...")
    try:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding_function
        )
    except Exception as e:
        print(f"Error loading vector store. Did you run process_documents.py? Error: {e}")
        return None

    # 3. Create Retriever
    # Using k=4 to retrieve 4 relevant document chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("Retriever initialized.")

    # 4. Initialize the LLM (Gemini 2.5 Flash for speed)
    llm = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0 # Keep it factual
    )
    print("LLM initialized.")

    # 5. Define the RAG Prompt Template
    template = """
    You are 'Buffett's Brain', an expert financial analyst and wise investor. 
    Your goal is to answer questions based ONLY on the provided context (annual letters, books, transcripts). 
    Do not use external knowledge. If the context does not contain the answer, say "I could not find a relevant answer in the collected wisdom of Buffett and Munger."
    
    Context: {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 6. Build the RAG Chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain successfully built.")
    return rag_chain

def main():
    """
    Main function to run the interactive chat loop.
    """
    rag_chain = setup_rag_chain()
    if not rag_chain:
        return

    print("\n--- Chat with Buffett's Brain ---")
    print(f"LLM: {GENERATION_MODEL_NAME} | Retriever k=4")
    print("Ask me anything about the investment philosophy of Warren Buffett and Charlie Munger.")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        query = input("You: ")
        if query.lower() in ['quit', 'exit']:
            print("Exiting chat. Goodbye!")
            break
        
        if not query.strip():
            continue

        try:
            start_time = time.time()
            
            # Invoke the RAG chain
            response = rag_chain.invoke(query)
            
            end_time = time.time()
            
            print(f"\nBuffett's Brain (Time: {end_time - start_time:.2f}s):\n{response}\n")

        except Exception as e:
            print(f"\nAn error occurred during query execution: {e}")
            print("Please check your API key and connection.")


if __name__ == "__main__":
    main()