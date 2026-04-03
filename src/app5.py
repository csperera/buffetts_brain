# app4.py v4.3.0 - Buffett's Brain - FREE TIER OPTIMIZED 🚀
# Lightweight version for AWS free tier (t2.micro)
# Removed: sentence-transformers, BM25, nltk (too heavy for free tier)
# Uses: Pre-computed Chroma DB from S3, simple vector search
import os
import re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch

def download_vector_db_from_s3():
    """Download vector database from S3 if running on AWS."""
    if os.getenv('AWS_REGION') or os.getenv('AWS_EXECUTION_ENV'):
        st.sidebar.info("☁️ Running on AWS - downloading vector DB from S3...")
        try:
            import boto3
            local_path = "/tmp/vector_db"
            s3 = boto3.client('s3')
            bucket_name = "buffetts-brain-knowledge-base"
            Path(local_path).mkdir(parents=True, exist_ok=True)
            
            paginator = s3.get_paginator('list_objects_v2')
            file_count = 0
            for page in paginator.paginate(Bucket=bucket_name, Prefix='vector_db/'):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('/'):
                            continue
                        local_file = os.path.join('/tmp', key)
                        Path(os.path.dirname(local_file)).mkdir(parents=True, exist_ok=True)
                        s3.download_file(bucket_name, key, local_file)
                        file_count += 1
            st.sidebar.success(f"✅ Downloaded {file_count} files from S3")
            return local_path
        except Exception as e:
            st.sidebar.error(f"❌ S3 download error: {e}")
            st.stop()
    else:
        st.sidebar.info("💻 Running locally - using local vector DB")
        return "knowledge_base/vector_db"

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

VECTOR_DB_PATH = download_vector_db_from_s3()
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
    Lightweight RAG pipeline - just vector store, no heavy dependencies.
    Uses Chroma's built-in embeddings (already computed in S3).
    """
    st.sidebar.info("🔨 Loading vector database...")
    
    try:
        # Load pre-computed vector database (no embedding function needed)
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH
        )
        st.sidebar.success(f"✅ Vector database loaded successfully")
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None, None, None
    
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
    
    return vectorstore, search_tool, llm


def evaluate_rag_relevance(query, rag_context, llm):
    """Evaluates if RAG context can answer the query."""
    evaluation_prompt = f"""You are evaluating whether retrieved context from Warren Buffett's letters can answer a question.

Context: {rag_context[:3500]}

Question: {query}

Rate how well this context answers the question (1-10):
- 1-3: Completely unrelated
- 4-5: Related topic but doesn't answer directly
- 6-7: Partially answers
- 8-9: Mostly/fully answers
- 10: Completely answers

Respond with ONLY a single number 1-10."""

    try:
        response = llm.invoke(evaluation_prompt)
        score_text = ''.join(filter(str.isdigit, response.content[:3]))
        if score_text:
            score = int(score_text)
            return min(max(score, 1), 10)
        else:
            return 7
    except:
        return 7


def process_query(query, vectorstore, search_tool, llm):
    """
    Simplified query routing - semantic search or web search.
    Returns: (response, debug_info)
    """
    debug_messages = []
    results = []
    use_search = False
    
    # Time-sensitive keyword detection
    time_keywords = [
        'yesterday', 'today', 'current', 'now', 'latest', 'recent', 
        'this week', 'last week', 'this month', 'this year',
        'price', 'stock price', 'as of',
        'news', 'announcement', 'just announced'
    ]
    
    query_lower = query.lower()
    is_time_sensitive = any(keyword in query_lower for keyword in time_keywords)
    
    if is_time_sensitive:
        matched_keywords = [k for k in time_keywords if k in query_lower]
        debug_messages.append(f"⏰ Detected time-sensitive query")
        debug_messages.append(f"📌 Matched keywords: {', '.join(matched_keywords)}")
        debug_messages.append(f"🔄 Skipping RAG → Going directly to web search")
        use_search = True
    
    try:
        if not use_search:
            debug_messages.append(f"📚 Non-time-sensitive query → Using Vector Search")
            
            # Simple vector search (using pre-computed embeddings)
            rag_docs = vectorstore.similarity_search(query, k=5)
            rag_context = "\n\n".join([doc.page_content for doc in rag_docs])
            
            debug_messages.append(f"📖 Retrieved {len(rag_docs)} documents from knowledge base")
            
            if rag_docs:
                first_source = rag_docs[0].metadata.get('source', 'Unknown')
                source_name = os.path.basename(first_source) if first_source != 'Unknown' else 'Unknown'
                debug_messages.append(f"📄 Top source: {source_name}")
            
            # Evaluate relevance
            relevance_score = evaluate_rag_relevance(query, rag_context, llm)
            debug_messages.append(f"🎯 RAG Relevance Score: {relevance_score}/10")
            
            if relevance_score >= 4:
                debug_messages.append(f"✅ Score ≥ 4 → Using RAG knowledge base")
                results.append(f"**From Buffett's Knowledge Base:** (Relevance: {relevance_score}/10)\n{rag_context}")
            else:
                debug_messages.append(f"⚠️ Score < 4 → RAG insufficient, triggering web search")
                use_search = True
            
    except Exception as e:
        debug_messages.append(f"❌ RAG Error: {str(e)}")
        import traceback
        debug_messages.append(f"📋 Traceback: {traceback.format_exc()[:200]}")
        results.append(f"**RAG Retrieval Error:** {str(e)}")
        use_search = True
    
    # Execute web search if needed
    if use_search:
        debug_messages.append(f"🌐 Executing Tavily web search...")
        
        try:
            search_results = search_tool.invoke(query)
            debug_messages.append(f"📦 Search result type: {type(search_results).__name__}")
            
            if isinstance(search_results, dict):
                if 'answer' in search_results:
                    answer = search_results['answer']
                    debug_messages.append(f"✅ Found 'answer' field: {len(answer)} chars")
                    results.append(f"**Tavily Direct Answer:**\n{answer}")
                
                if 'results' in search_results:
                    result_list = search_results['results']
                    debug_messages.append(f"📊 Found 'results' field with {len(result_list)} items")
                    
                    search_summaries = []
                    for i, result in enumerate(result_list):
                        if isinstance(result, dict):
                            content = result.get('content', '')
                            url = result.get('url', '')
                            if content:
                                search_summaries.append(f"**Source {i+1}:** {content}\n📎 URL: {url}")
                    
                    if search_summaries:
                        debug_messages.append(f"✅ Extracted {len(search_summaries)} additional sources")
                        results.append(f"**Additional Web Sources:**\n\n" + "\n\n".join(search_summaries))
                
        except Exception as e:
            debug_messages.append(f"❌ Search exception: {str(e)}")
            results.append(f"**Search Error:** {str(e)}")
    
    combined_context = "\n\n---\n\n".join(results) if results else "No relevant information found."
    debug_messages.append(f"📊 Final context: {len(combined_context)} chars, {len(results)} sections")
    
    # Generate final response
    prompt_template = """You are 'Buffett's Brain', an expert financial analyst with deep knowledge of Warren Buffett and Charlie Munger's investment philosophy.

Based on the following information, answer the user's question thoroughly and accurately.

IMPORTANT INSTRUCTIONS: 
1. Start by briefly restating the question to confirm understanding
2. If the context is from Buffett's letters, quote or paraphrase naturally with years when relevant
3. If web search results are provided, use them directly and cite sources
4. DO NOT make up data that isn't in the context

{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": combined_context, "question": query})
        debug_messages.append(f"✅ LLM response generated successfully")
        return response, debug_messages
    except Exception as e:
        debug_messages.append(f"❌ LLM error: {str(e)}")
        return f"⚠️ **Error generating response:** {str(e)}", debug_messages


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Buffett's Brain RAG Chat", layout="wide")

vectorstore, search_tool, llm = setup_rag_and_search()
if vectorstore is None:
    st.stop()

# Header
st.markdown(
    "<h1 style='text-align: center; font-size: 4.5em;'>🧠 Buffett's Brain</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center; font-size: 2em; color: #AAAAAA;'>AI-Powered Investment Wisdom</h2>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Powered by Groq (Llama 3.1 8B) & Tavily Search</p>", 
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚡ About")
    st.markdown("""
    **Buffett's Brain** (Free Tier Edition) combines:
    - 📚 **RAG Knowledge Base**: 
      - Partnership Letters (1957-1969)
      - Berkshire Hathaway Letters (1977-2024)
      - Poor Charlie's Almanack
      - **Total: 2,250+ pages, 8,518 chunks**
    - 🔍 **Vector Search**: Semantic similarity
    - 🌐 **Real-Time Web Search**: Current data via Tavily
    
    **Optimized for AWS Free Tier:**
    - ✅ Lightweight dependencies
    - ✅ Pre-computed embeddings
    - ✅ Fast startup time
    - ✅ Low memory footprint
    
    **Tech Stack:**
    - LLM: Groq (Llama 3.1 8B)
    - Vector DB: Chroma (pre-computed)
    - Search: Tavily Advanced
    - Storage: AWS S3
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display welcome message
if len(st.session_state["messages"]) == 0:
    st.chat_message("assistant").write("""Hello! I am **Buffett's Brain** 🧠

I can answer questions about Warren Buffett and Charlie Munger's investment philosophy.

**Try asking me:**
- 📖 "What did Buffett say about float in his insurance business?"
- 💭 "What is Buffett's circle of competence principle?"
- 🏢 "Explain the concept of economic moats"
- 📰 "What is Berkshire's stock price today?" (uses web search)

*Ask me anything about value investing!* ⚡""")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "debug" in msg:
            with st.expander("📊 Query Information", expanded=False):
                for debug_line in msg["debug"]:
                    st.text(debug_line)

# User input handler
if prompt := st.chat_input("Ask me a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response, debug_info = process_query(
                prompt, vectorstore, search_tool, llm
            )
            
            st.write(response)
            
            with st.expander("📊 Query Information", expanded=False):
                for debug_line in debug_info:
                    st.text(debug_line)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "debug": debug_info
            })