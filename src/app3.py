# Version 1.0: Buffett's Brain - Production RAG System 🚀
# Hybrid RAG with intelligent routing and web search fallback
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch 

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

VECTOR_DB_PATH = "../knowledge_base/vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    st.error("Error: GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()
if not TAVILY_API_KEY:
    st.error("Error: TAVILY_API_KEY not found. Please add it to your .env file.")
    st.stop()


@st.cache_resource
def setup_rag_and_search():
    """
    Initializes the RAG pipeline with vector store, embeddings, LLM, and web search.
    """
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    try:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=embedding_function
        )
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None, None, None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    search_tool = TavilySearch(
        api_key=TAVILY_API_KEY, 
        max_results=3, 
        search_depth="advanced", 
        include_answer=True, 
        include_raw_content=False
    )
    llm = ChatGroq(
        model=GROQ_MODEL_NAME, 
        groq_api_key=GROQ_API_KEY, 
        temperature=0.0, 
        max_tokens=2048
    )
    
    return retriever, search_tool, llm


def evaluate_rag_relevance(query, rag_context, llm):
    """
    Uses LLM to evaluate if RAG context can directly answer the query.
    Returns relevance score 1-10.
    """
    evaluation_prompt = f"""You are a STRICT evaluator. Determine if the context can DIRECTLY answer the EXACT question asked.

Context:
{rag_context[:1500]}

Question: {query}

Rate relevance 1-10:
- 1-2: Context completely irrelevant to the question
- 3-4: Context mentions related topics BUT does NOT answer the specific question
- 5-6: Context partially relevant but missing key information needed to answer
- 7-8: Context has most information needed but may lack some specifics
- 9-10: Context directly and completely answers the question

CRITICAL: If question asks for CURRENT/RECENT/YESTERDAY data but context only has HISTORICAL information, score 1-3 maximum.

Respond ONLY with a number 1-10."""

    try:
        response = llm.invoke(evaluation_prompt)
        score = int(''.join(filter(str.isdigit, response.content[:3])))
        return min(max(score, 1), 10)
    except:
        return 7


def process_query(query, retriever, search_tool, llm):
    """
    Intelligent query routing:
    1. Detects time-sensitive queries → triggers web search
    2. For other queries → evaluates RAG relevance, falls back to search if needed
    3. Returns comprehensive answer with appropriate sources
    """
    results = []
    use_search = False
    
    # Time-sensitive keyword detection
    time_keywords = [
        'yesterday', 'today', 'current', 'now', 'latest', 'recent', 
        'this week', 'last week', 'price', 'stock'
    ]
    query_lower = query.lower()
    is_time_sensitive = any(keyword in query_lower for keyword in time_keywords)
    
    try:
        if is_time_sensitive:
            # Skip RAG for time-sensitive queries, go straight to search
            use_search = True
        else:
            # Evaluate RAG relevance for non-time-sensitive queries
            rag_docs = retriever.invoke(query) 
            rag_context = "\n\n".join([doc.page_content for doc in rag_docs])
            relevance_score = evaluate_rag_relevance(query, rag_context, llm)
            
            if relevance_score >= 5:
                results.append(
                    f"**From Buffett's Knowledge Base:** (Relevance: {relevance_score}/10)\n{rag_context}"
                )
            else:
                results.append(
                    f"**🔍 RAG Check:** Knowledge base relevance score: {relevance_score}/10 (too low). Searching the web..."
                )
                use_search = True
            
    except Exception as e:
        results.append(f"**RAG Retrieval Error:** {str(e)}")
        use_search = True
    
    # Execute web search if needed
    if use_search:
        try:
            search_results = search_tool.invoke(query)
            
            if search_results:
                search_summaries = []
                for i, result in enumerate(search_results):
                    if isinstance(result, dict):
                        content = result.get('content', '')
                        url = result.get('url', '')
                        if content:
                            search_summaries.append(
                                f"**Source {i+1}:** {content}\n📎 URL: {url}"
                            )
                
                if search_summaries:
                    results.append(
                        f"**Real-Time Web Search Results:**\n\n" + "\n\n".join(search_summaries)
                    )
                
        except Exception as e:
            results.append(f"**Search Error:** {e}")
    
    combined_context = "\n\n---\n\n".join(results) if results else "No relevant information found."
    
    # Generate final response
    prompt_template = """You are 'Buffett's Brain', an expert financial analyst and wise investor.

Based on the following information, answer the user's question thoroughly and accurately.

IMPORTANT: 
1. Start by briefly restating the question
2. If web search results are provided with specific data (numbers, prices, dates), USE THEM directly
3. If search results only provide URLs without actual data, acknowledge this limitation and provide the URLs
4. DO NOT make up data that isn't in the search results
5. If using knowledge base, reference Buffett/Munger's wisdom with appropriate citations

{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": combined_context, "question": query})
        return response
    except Exception as e:
        return f"⚠️ **Error generating response:** {str(e)}"


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Buffett's Brain RAG Chat", layout="wide")

retriever, search_tool, llm = setup_rag_and_search()
if retriever is None:
    st.stop()

# Header
st.markdown(
    "<h1 style='text-align: center; font-size: 4.5em;'>🧠 Buffett's Brain</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center; font-size: 2em; color: #AAAAAA;'>RAG-Enabled AI Agent with Intelligent Routing</h2>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Powered by Groq (Llama 3.1 8B), Tavily Search & HuggingFace Embeddings<br/>✨ Hybrid RAG with LLM-based relevance scoring ✨</p>", 
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚡ About")
    st.markdown("""
    **Buffett's Brain** combines:
    - 📚 **RAG Knowledge Base**: 2,100+ pages of Buffett/Munger wisdom (50 years of annual letters, Poor Charlie's Almanack, speeches)
    - 🌐 **Real-Time Web Search**: Current market data and news via Tavily
    - 🧠 **Intelligent Routing**: LLM evaluates query relevance and chooses optimal source
    
    **How it works:**
    1. Detects time-sensitive queries (prices, news) → web search
    2. Evaluates RAG relevance for other queries (1-10 score)
    3. Falls back to web search if RAG score < 5
    
    **Note on Real-Time Data:**
    Web search provides *references* to current data sources, not always the raw data itself. For production use, consider integrating dedicated financial APIs.
    
    **Tech Stack:**
    - LLM: Groq (Llama 3.1 8B) - blazing fast!
    - Embeddings: HuggingFace (all-MiniLM-L6-v2)
    - Vector DB: Chroma
    - Search: Tavily Advanced
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Chat cleared! Ask me anything."}
        ]
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": """Hello! I am **Buffett's Brain** 🧠

I have deep knowledge of Warren Buffett and Charlie Munger's investment philosophy from 50 years of writings.

**Try asking me:**
- 📖 "What is Buffett's circle of competence principle?"
- 💭 "What would Charlie Munger say about cryptocurrency?"
- 🏢 "Explain the concept of economic moats"
- 📰 "What are recent developments with Berkshire Hathaway?" (web search)

*Intelligent routing powered by Groq - expect lightning-fast responses!* ⚡"""
        }
    ]

# Chat container
chat_history_container = st.container(height=500)

with chat_history_container:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# User input handler
if prompt := st.chat_input("Ask me a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with chat_history_container:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing query and retrieving information..."):
                response = process_query(prompt, retriever, search_tool, llm)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})