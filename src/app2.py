# Version 0.3 (Layout Fix): Forces the assistant's dynamic response to render inside the fixed-height container, fixing the "out-of-bounds" rendering issue.
# Note: This may re-introduce the minor response flicker.
import os
import streamlit as st
from dotenv import load_dotenv

# --- CORRECTED LangChain Imports ---
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
SYSTEM_PROMPT = """
You are 'Buffett's Brain', an expert financial analyst and wise investor. 

1. **For philosophy, strategy, or historical facts**, answer based ONLY on the provided context (annual letters, books, transcripts). 
2. **For real-time facts (like current stock prices, recent news, or general knowledge)**, use your general knowledge or search capabilities.

If the context does not contain the answer and the question is NOT a real-time fact, say "I could not find a relevant answer in the collected wisdom of Buffett and Munger."
"""

# Check for API Key on load
if not GEMINI_API_KEY:
    st.error("Error: GEMINI_API_KEY not found. Please ensure your .env file is configured correctly.")
    st.stop()


# Use st.cache_resource to load the RAG chain only once
@st.cache_resource
def setup_rag_chain():
    """
    Sets up the Retrieval-Augmented Generation (RAG) chain.
    """
    # 1. Initialize Embeddings (must match the model used for creating the store)
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GEMINI_API_KEY
    )
    
    # 2. Load the Vector Store
    try:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding_function
        )
    except Exception as e:
        # Note: Streamlit stops execution with st.error + st.stop()
        st.error(f"Error loading vector store. Did you run process_documents.py? Error: {e}")
        return None

    # 3. Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. Initialize the LLM (Gemini 2.5 Flash for speed)
    llm = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.0,
        tools=[{"google_search": {}}] 
    )

    # 5. Define the RAG Prompt Template
    template = SYSTEM_PROMPT + "\n\nContext: {context}\n\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)
    
    # 6. Build the RAG Chain
    rag_chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit UI Setup ---

# Initialize the RAG chain
rag_chain = setup_rag_chain()
if rag_chain is None:
    st.stop()

st.set_page_config(page_title="Buffett's Brain RAG Chat", layout="wide")

# 1st line: "Buffett's Brain" - Very large font, centered
st.markdown("<h1 style='text-align: center; font-size: 4.5em;'>Buffett's Brain</h1>", unsafe_allow_html=True)

# 2nd line: "RAG-Enabled AI Agent" - Large font, centered
st.markdown("<h2 style='text-align: center; font-size: 2em; color: #AAAAAA;'>RAG-Enabled AI Agent</h2>", unsafe_allow_html=True)

# Optional small caption (retained original concept but cleaner)
st.markdown(f"<p style='text-align: center;'>Powered by Gemini 2.5 Flash and <code>text-embedding-004</code></p>", unsafe_allow_html=True)

# --- START SCROLLING CHAT CONTAINER LOGIC ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am 'Buffett's Brain'. Ask me anything about the investment philosophy of Warren Buffett and Charlie Munger, or ask for current stock prices!"}
    ]

# Use a dedicated container for the chat history with a fixed height
# This is the key element that prevents the rest of the page from scrolling.
chat_history_container = st.container(height=500) 

# Display chat messages from history inside the scrolling container
with chat_history_container:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# --- END SCROLLING CHAT CONTAINER LOGIC ---


# Handle user input and invoke the RAG chain (this remains outside the scroll container)
if prompt := st.chat_input("Ask me a question..."):
    # 1. Add user message to history. This triggers the script rerun.
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Invoke the RAG chain and show the response, forcing it to render inside the container.
    with chat_history_container: # <-- FIX: Re-entering the container context to ensure placement
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke(prompt)
                    st.write(response)

                    # 3. Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"An error occurred: {e}. Please check your Gemini API key and network connection."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})