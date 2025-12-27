# Version 1.6: Buffett's Brain - Production RAG System 🚀
# FIXED: Robust year filtering with large candidate pool (100 docs)
# Hybrid RAG with intelligent routing and web search fallback
import os
import re
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
        return None, None, None, None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
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
    
    return vectorstore, retriever, search_tool, llm


def extract_year_from_query(query):
    """
    Extracts a 4-digit year from the query (1950s-2020s).
    Returns year as string if found, None otherwise.
    """
    matches = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', query)
    if matches:
        return matches[0]
    return None


def retrieve_with_year_filter(vectorstore, query, year=None, k=5):
    """
    Retrieves documents with year filtering.
    PRIMARY METHOD: Manual filtering with large candidate pool (most reliable)
    FALLBACK: Native ChromaDB filter if manual filter fails
    """
    if year:
        # PRIMARY APPROACH: Get large pool, then manually filter
        # This is most reliable across different ChromaDB/LangChain versions
        try:
            # Get 100+ candidates to ensure we have all year's documents
            # (1962 has 25 chunks, we want to see them all)
            large_pool = vectorstore.similarity_search(query, k=100)
            
            # Manually filter to only documents with year in source path
            filtered_docs = [
                doc for doc in large_pool 
                if year in doc.metadata.get('source', '')
            ]
            
            if filtered_docs:
                # Found year-specific docs! Return top k by relevance
                return filtered_docs[:k]
            else:
                # No matches in large pool - try native filter as fallback
                try:
                    docs = vectorstore.similarity_search(
                        query,
                        k=k,
                        filter={"source": {"$contains": year}}
                    )
                    if docs:
                        return docs
                except:
                    pass
                
                # Both approaches failed - return unfiltered top results
                return large_pool[:k]
                
        except Exception as e:
            # Large pool failed - fall back to standard retrieval
            return vectorstore.similarity_search(query, k=k)
    else:
        # No year specified, standard retrieval
        return vectorstore.similarity_search(query, k=k)


def evaluate_rag_relevance(query, rag_context, llm):
    """
    Uses LLM to evaluate if RAG context can answer the query.
    Returns relevance score 1-10.
    """
    evaluation_prompt = f"""You are evaluating whether retrieved context from Warren Buffett's letters and writings can answer a question.

Context (from Buffett's Partnership Letters and Berkshire Hathaway shareholder letters):
{rag_context[:3500]}

Question: {query}

Rate how well this context answers the question (1-10):

SCORING GUIDE:
- 1-3: Context is completely unrelated to the question (different topic entirely)
- 4-5: Context is on a related topic but doesn't directly address the question
- 6-7: Context contains relevant information that partially answers the question
- 8-9: Context has most/all information needed to answer the question well
- 10: Context completely and directly answers the question

IMPORTANT GUIDELINES:
✓ If question asks about a SPECIFIC YEAR's letter (e.g., "1962 letter", "the 1974 letter") AND context contains content from that year → score 8-10
✓ If question asks about Buffett's views on a topic AND context discusses that topic → score 7-10
✓ Historical questions about past letters/writings CAN be answered from historical documents
✓ Only score low (1-4) if context is truly irrelevant or about completely different topic

✗ Do NOT penalize historical context for not being "current" - that's what web search is for
✗ Do NOT require exact phrase matches - concepts and ideas count

Respond with ONLY a single number 1-10, nothing else."""

    try:
        response = llm.invoke(evaluation_prompt)
        score_text = ''.join(filter(str.isdigit, response.content[:3]))
        if score_text:
            score = int(score_text)
            return min(max(score, 1), 10)
        else:
            return 7
    except Exception as e:
        return 7


def process_query(query, vectorstore, retriever, search_tool, llm):
    """
    Intelligent query routing with year-aware retrieval and debug output.
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
    
    # Extract year from query for precise filtering
    mentioned_year = extract_year_from_query(query)
    if mentioned_year:
        debug_messages.append(f"📅 Detected year in query: {mentioned_year}")
    
    # Debug: Show detection results
    if is_time_sensitive:
        matched_keywords = [k for k in time_keywords if k in query_lower]
        debug_messages.append(f"⏰ Detected time-sensitive query")
        debug_messages.append(f"📌 Matched keywords: {', '.join(matched_keywords)}")
        debug_messages.append(f"🔄 Skipping RAG → Going directly to web search")
    
    try:
        if is_time_sensitive:
            use_search = True
        else:
            debug_messages.append(f"📚 Non-time-sensitive query → Checking RAG knowledge base")
            
            # Use year-filtered retrieval if year was mentioned
            if mentioned_year:
                debug_messages.append(f"🎯 Using large pool (100 docs) + manual filter for year {mentioned_year}")
                rag_docs = retrieve_with_year_filter(vectorstore, query, year=mentioned_year, k=5)
            else:
                debug_messages.append(f"📖 Using standard semantic retrieval")
                rag_docs = retriever.invoke(query)
            
            rag_context = "\n\n".join([doc.page_content for doc in rag_docs])
            
            debug_messages.append(f"📖 Retrieved {len(rag_docs)} documents from knowledge base")
            
            # Show what we retrieved
            if rag_docs:
                first_source = rag_docs[0].metadata.get('source', 'Unknown')
                source_name = os.path.basename(first_source) if first_source != 'Unknown' else 'Unknown'
                debug_messages.append(f"📄 Top source: {source_name}")
                
                # Check if year filter worked
                if mentioned_year:
                    if mentioned_year in source_name:
                        debug_messages.append(f"✅ Year filter SUCCESS - retrieved {mentioned_year} document")
                    else:
                        debug_messages.append(f"⚠️ Year filter fallback - no {mentioned_year} docs found, showing semantic matches")
            
            relevance_score = evaluate_rag_relevance(query, rag_context, llm)
            debug_messages.append(f"🎯 RAG Relevance Score: {relevance_score}/10")
            
            if relevance_score >= 4:
                debug_messages.append(f"✅ Score ≥ 4 → Using RAG knowledge base")
                results.append(
                    f"**From Buffett's Knowledge Base:** (Relevance: {relevance_score}/10)\n{rag_context}"
                )
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
            
            if search_results is None:
                debug_messages.append(f"❌ Search returned None!")
                results.append("**⚠️ Search Issue:** No results returned from Tavily API.")
                
            elif isinstance(search_results, dict):
                debug_messages.append(f"📖 Tavily returned dict format (include_answer=True)")
                
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
                                search_summaries.append(
                                    f"**Source {i+1}:** {content}\n📎 URL: {url}"
                                )
                    
                    if search_summaries:
                        debug_messages.append(f"✅ Extracted {len(search_summaries)} additional sources")
                        results.append(
                            f"**Additional Web Sources:**\n\n" + "\n\n".join(search_summaries)
                        )
                
            elif isinstance(search_results, list):
                debug_messages.append(f"📊 Tavily returned list format with {len(search_results)} results")
                
                if len(search_results) > 0:
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
                        debug_messages.append(f"✅ Successfully extracted {len(search_summaries)} search results")
                        results.append(
                            f"**Real-Time Web Search Results:**\n\n" + "\n\n".join(search_summaries)
                        )
                
        except Exception as e:
            debug_messages.append(f"❌ Search exception: {str(e)}")
            results.append(f"**Search Error:** {str(e)}")
    
    combined_context = "\n\n---\n\n".join(results) if results else "No relevant information found."
    debug_messages.append(f"📊 Final context: {len(combined_context)} chars, {len(results)} sections")
    
    # Generate final response
    prompt_template = """You are 'Buffett's Brain', an expert financial analyst with deep knowledge of Warren Buffett and Charlie Munger's investment philosophy.

Based on the following information, answer the user's question thoroughly and accurately.

IMPORTANT INSTRUCTIONS: 
1. IF the context says "No relevant information found", respond: 
   "I apologize, but I don't have sufficient information available to answer this question."

2. Start by briefly restating the question to confirm understanding

3. If the context is from Buffett's Partnership Letters or Berkshire Hathaway letters:
   - These are historical documents containing Buffett's actual writings
   - Quote or paraphrase them naturally
   - Cite the year when relevant (e.g., "In his 1962 letter, Buffett discussed...")
   - Provide specific examples and data when present

4. If web search results are provided:
   - Use them directly and cite sources
   - Include URLs when provided

5. DO NOT make up data that isn't in the context

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

vectorstore, retriever, search_tool, llm = setup_rag_and_search()
if vectorstore is None:
    st.stop()

# Header
st.markdown(
    "<h1 style='text-align: center; font-size: 4.5em;'>🧠 Buffett's Brain</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center; font-size: 2em; color: #AAAAAA;'>RAG-Enabled AI Agent with Robust Year Filtering</h2>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Powered by Groq (Llama 3.1 8B), Tavily Search & HuggingFace Embeddings<br/>✨ Now with large-pool year filtering for maximum context ✨<br/>🔍 DEBUG MODE ACTIVE 🔍</p>", 
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚡ About")
    st.markdown("""
    **Buffett's Brain** combines:
    - 📚 **RAG Knowledge Base**: 
      - Partnership Letters (1957-1969)
      - Berkshire Hathaway Letters (1977-2024)
      - Poor Charlie's Almanack
      - Munger Transcripts
      - **Total: 2,250+ pages across 67 years**
    - 🌐 **Real-Time Web Search**: Current market data via Tavily
    - 🧠 **Intelligent Routing**: 
      - Year-specific queries → ChromaDB metadata filter
      - Time-sensitive queries → Web search
      - Philosophical queries → Semantic search
    
    **Recent Updates (v1.6):**
    - ✅ **FIXED: Robust year filtering** using large candidate pool
    - ✅ Retrieves 5 most relevant docs from target year (not just 1)
    - ✅ More context = better LLM responses
    - ✅ Reliable across different ChromaDB versions
    
    **Tech Stack:**
    - LLM: Groq (Llama 3.1 8B)
    - Embeddings: HuggingFace (all-MiniLM-L6-v2)
    - Vector DB: Chroma (8,518 chunks)
    - Search: Tavily Advanced
    
    ---
    
    **🔍 DEBUG MODE**
    
    See diagnostic info in expandable sections.
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

I have deep knowledge of Warren Buffett and Charlie Munger's investment philosophy spanning 67 years (1957-2024).

**Try asking me:**
- 📖 "What were the key points of the 1962 letter?"
- 📊 "How did Buffett perform in 1960 versus the Dow Jones?"
- 💭 "What is Buffett's circle of competence principle?"
- 🧠 "What would Munger say about cryptocurrency?"
- 📰 "What is Berkshire's stock price today?" (web search)

*Now with robust large-pool year filtering for rich context!* ⚡""")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "debug" in msg:
            with st.expander("🔍 Debug Information", expanded=False):
                for debug_line in msg["debug"]:
                    st.text(debug_line)

# User input handler
if prompt := st.chat_input("Ask me a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing query and retrieving information..."):
            response, debug_info = process_query(prompt, vectorstore, retriever, search_tool, llm)
            
            st.write(response)
            
            with st.expander("🔍 Debug Information", expanded=True):
                for debug_line in debug_info:
                    st.text(debug_line)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "debug": debug_info
            })