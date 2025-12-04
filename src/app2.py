# Version 0.7: RAG with Tavily Search - GROQ EDITION! 🚀
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Core Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Groq for LLM (FAST & FREE!)
from langchain_groq import ChatGroq

# HuggingFace Embeddings (FREE!)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Tools
from langchain_tavily import TavilySearch 

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Define paths and constants
VECTOR_DB_PATH = "../knowledge_base/vector_db"  # Updated path since we're in /src
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Free, fast embeddings
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Super fast!

# Check for API Keys
if not GROQ_API_KEY:
    st.error("Error: GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()
if not TAVILY_API_KEY:
    st.error("Error: TAVILY_API_KEY not found. Please add it to your .env file.")
    st.stop()


@st.cache_resource
def setup_rag_and_search():
    """
    Sets up both RAG retrieval and web search capabilities.
    """
    # 1. Initialize Embeddings (HuggingFace - Free!)
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # 2. Load the Vector Store
    try:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embedding_function
        )
    except Exception as e:
        st.error(f"Error loading vector store. Did you run process_documents.py? Error: {e}")
        return None, None, None

    # 3. Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. Create Tavily Search Tool
    search_tool = TavilySearch(
        api_key=TAVILY_API_KEY,
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False
    )

    # 5. Initialize Groq LLM (BLAZING FAST! 🔥)
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=2048
    )
    
    return retriever, search_tool, llm


def process_query(query, retriever, search_tool, llm):
    """
    Processes a query by determining whether to use RAG, search, or both.
    """
    # Keywords that suggest we need real-time search
    search_keywords = [
        'current', 'today', 'now', 'latest', 'recent', 'price', 'stock',
        'news', 'this year', 'this month', '2024', '2025', 'happening'
    ]
    
    # Keywords that suggest we need RAG knowledge base
    rag_keywords = [
        'buffett', 'munger', 'philosophy', 'principle', 'moat', 'margin of safety',
        'investment strategy', 'annual letter', 'what does buffett', 'berkshire'
    ]
    
    query_lower = query.lower()
    needs_search = any(keyword in query_lower for keyword in search_keywords)
    needs_rag = any(keyword in query_lower for keyword in rag_keywords)
    
    # Default: if unclear, use RAG
    if not needs_search and not needs_rag:
        needs_rag = True
    
    results = []
    
    # Get RAG context if needed
    if needs_rag:
        try:
            rag_docs = retriever.invoke(query) 
            rag_context = "\n\n".join([doc.page_content for doc in rag_docs])
            results.append(f"**From Buffett's Knowledge Base:**\n{rag_context}")
        except Exception as e:
            results.append(f"**RAG Retrieval Error:** {str(e)}")
    
    # Get search results if needed
    if needs_search:
        try:
            search_results = search_tool.invoke(query)
            if search_results:
                search_context = "\n\n".join([
                    f"- {result.get('content', '')}" 
                    for result in search_results 
                    if isinstance(result, dict)
                ])
                results.append(f"**From Web Search:**\n{search_context}")
        except Exception as e:
            results.append(f"**Search Error:** Could not retrieve web results: {e}")
    
    # Combine all context
    combined_context = "\n\n---\n\n".join(results) if results else "No relevant information found."
    
    # Create prompt
    prompt_template = """You are 'Buffett's Brain', an expert financial analyst and wise investor.

Based on the following information, answer the user's question thoroughly and accurately.
If using web search results, cite the sources. If using knowledge base, reference Buffett/Munger's wisdom.

{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Generate response with error handling
    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": combined_context, "question": query})
        return response
        
    except Exception as e:
        return f"⚠️ **Error generating response:** {str(e)}\n\nPlease try again or rephrase your question."


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Buffett's Brain RAG Chat", layout="wide")

# Initialize components
retriever, search_tool, llm = setup_rag_and_search()
if retriever is None:
    st.stop()

# Header
st.markdown("<h1 style='text-align: center; font-size: 4.5em;'>🧠 Buffett's Brain</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 2em; color: #AAAAAA;'>RAG-Enabled AI Agent with Real-Time Search</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>Powered by Groq (Llama 3.1 70B), Tavily Search & HuggingFace Embeddings</p>", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header("⚡ About")
    st.markdown("""
    **Buffett's Brain** combines:
    - 📚 RAG knowledge base (Buffett/Munger wisdom)
    - 🌐 Real-time web search (current market data)
    - 🚀 Groq LLM (blazing fast & free!)
    
    **Model:** `llama-3.1-70b-versatile`
    
    **Tips:**
    - Ask about investment philosophy
    - Check current stock prices
    - Get company analysis
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Chat cleared! How can I help you?"}
        ]
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": """Hello! I am **Buffett's Brain** 🧠

Ask me about:
- 📊 Investment philosophy from Buffett & Munger (from my knowledge base)
- 💹 Current stock prices and market news (via real-time search)
- 🏢 Company analysis combining both sources!

*Powered by Groq - expect lightning-fast responses! ⚡*"""}
    ]

# Chat container with fixed height
chat_history_container = st.container(height=500)

# Display chat history
with chat_history_container:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask me a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with chat_history_container:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking at lightning speed..."):
                response = process_query(prompt, retriever, search_tool, llm)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})