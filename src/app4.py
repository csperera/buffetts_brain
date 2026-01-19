# app4.py v4.2.2 - Buffett's Brain - Production RAG System 🚀
# MAJOR ARCHITECTURAL UPGRADE: BM25 + Semantic Hybrid Search Engine
# v4.2.1: Fixed keyword extraction to filter generic terms (company, buffett, etc.)
# Intelligently prioritizes: earlier mentions (temporal) × more detail (content)
import os
import re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

# BM25 for keyword search
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (only first run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

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

VECTOR_DB_PATH = "knowledge_base/vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Hybrid search configuration
BM25_WEIGHT = 0.4  # 40% weight for keyword matching
SEMANTIC_WEIGHT = 0.6  # 60% weight for semantic similarity
HYBRID_K = 20  # Get 20 candidates from each method

if not GROQ_API_KEY:
    st.error("Error: GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()
if not TAVILY_API_KEY:
    st.error("Error: TAVILY_API_KEY not found. Please add it to your .env file.")
    st.stop()


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25, removing stopwords."""
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        # Keep important words, remove common stopwords
        tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
        return tokens
    except:
        # Fallback to simple split if NLTK fails
        return text.lower().split()


@st.cache_resource
def setup_rag_and_search():
    """
    Initializes the RAG pipeline with:
    - Vector store (semantic search)
    - BM25 index (keyword search)
    - LLM and web search
    """
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    try:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=embedding_function
        )
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None, None, None, None, None
    
    # Build BM25 index from all documents
    st.sidebar.info("🔨 Building BM25 index... (first load only)")
    
    try:
        # Get all documents from vector store
        all_docs = vectorstore.get()
        
        # Extract documents and metadata
        documents = []
        metadatas = []
        
        if 'documents' in all_docs and 'metadatas' in all_docs:
            documents = all_docs['documents']
            metadatas = all_docs['metadatas']
        
        # Tokenize all documents for BM25
        tokenized_corpus = [tokenize_text(doc) for doc in documents]
        
        # Build BM25 index
        bm25_index = BM25Okapi(tokenized_corpus)
        
        # Create document objects for retrieval
        doc_objects = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(documents, metadatas)
        ]
        
        st.sidebar.success(f"✅ BM25 index built: {len(documents)} documents")
        
    except Exception as e:
        st.sidebar.error(f"Error building BM25 index: {e}")
        bm25_index = None
        doc_objects = []

    retriever = vectorstore.as_retriever(search_kwargs={"k": HYBRID_K})
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
    
    return vectorstore, bm25_index, doc_objects, search_tool, llm


def extract_year_from_query(query):
    """Extracts a 4-digit year from the query (1950s-2020s)."""
    matches = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', query)
    if matches:
        return matches[0]
    return None


def extract_year_from_filename(filepath: str) -> int:
    """
    Extract 4-digit year from filename.
    Returns year as int, or None if not found.
    """
    import re
    match = re.search(r'(\d{4})', filepath)
    if match:
        year = int(match.group(1))
        # Sanity check: reasonable year range
        if 1950 <= year <= 2030:
            return year
    return None


def extract_keywords_from_query(query: str) -> List[str]:
    """
    Extract important keywords from query for mention counting.
    Focuses on unique entities, filters out generic terms and author names.
    """
    # Tokenize and get significant terms
    tokens = tokenize_text(query)
    
    # Also capture multi-word proper nouns (e.g., "Sanborn Map Company")
    words = query.split()
    capitalized = [w.strip('.,!?;:') for w in words if w and w[0].isupper()]
    
    # Combine tokenized terms and capitalized phrases
    all_keywords = list(set(tokens + [w.lower() for w in capitalized if len(w) > 2]))
    
    # CRITICAL: Filter out generic terms and author names
    # These appear in almost every document and poison the earliest_mention search
    generic_stopwords = {
        # Query meta-words
        'what', 'when', 'where', 'how', 'why', 'did', 'say', 'said', 'about', 
        'letter', 'letters', 'tell', 'mention', 'mentioned', 'discuss', 'discussed',
        # Author names (appear in every document!)
        'buffett', 'warren', 'munger', 'charlie', 'charles',
        # Super generic business terms
        'company', 'companies', 'corporation', 'corp', 'inc', 'business',
        'investment', 'investments', 'stock', 'stocks', 'share', 'shares',
        'market', 'price', 'value', 'year', 'years',
        # Generic verbs
        'make', 'made', 'do', 'does', 'get', 'give', 'take'
    }
    
    keywords = [k for k in all_keywords if k.lower() not in generic_stopwords]
    
    # Additional filter: Remove very short words (likely noise)
    keywords = [k for k in keywords if len(k) >= 3]
    
    return keywords


def count_keyword_mentions(content: str, keywords: List[str]) -> int:
    """
    Count how many times keywords appear in content.
    """
    content_lower = content.lower()
    count = 0
    for keyword in keywords:
        count += content_lower.count(keyword.lower())
    return count


def find_earliest_mention_year(keywords: List[str], doc_objects: List[Document]) -> int:
    """
    Find the earliest year where any of the keywords are mentioned.
    This identifies the primary source year.
    """
    earliest_year = None
    
    for doc in doc_objects:
        # Check if any keywords appear in this document
        content_lower = doc.page_content.lower()
        has_keyword = any(kw.lower() in content_lower for kw in keywords)
        
        if has_keyword:
            year = extract_year_from_filename(doc.metadata.get('source', ''))
            if year:
                if earliest_year is None or year < earliest_year:
                    earliest_year = year
    
    return earliest_year


def calculate_temporal_boost(year_diff: int) -> float:
    """
    Calculate boost based on temporal distance from first mention.
    First mention = primary source = highest boost.
    
    Decay curve:
    - 0 years: 2.0x (first mention)
    - 1-2 years: 1.6x (very close)
    - 3-5 years: 1.3x (same era)
    - 6-10 years: 1.1x (same decade)
    - 10+ years: 1.0x (later reference)
    """
    if year_diff == 0:
        return 2.0
    elif year_diff <= 2:
        return 1.6
    elif year_diff <= 5:
        return 1.3
    elif year_diff <= 10:
        return 1.1
    else:
        return 1.0


def calculate_content_boost(mention_count: int) -> float:
    """
    Calculate boost based on number of keyword mentions.
    More mentions = more detailed discussion.
    
    Linear boost: 1.0 + (count * 0.1)
    Examples:
    - 1 mention: 1.1x (brief reference)
    - 10 mentions: 2.0x (substantial)
    - 22 mentions: 3.2x (extensive - like Sanborn 1960)
    - 50 mentions: 6.0x (comprehensive)
    """
    return 1.0 + (mention_count * 0.1)


def get_combined_boost(
    doc: Document, 
    keywords: List[str], 
    earliest_year: int
) -> Tuple[float, dict]:
    """
    Calculate final boost combining temporal priority and content depth.
    
    final_boost = temporal_boost * content_boost
    
    Returns: (boost_value, debug_info)
    """
    debug_info = {}
    
    # Extract year from this document
    doc_year = extract_year_from_filename(doc.metadata.get('source', ''))
    
    if not doc_year:
        # Can't determine year, use default boost
        debug_info['temporal_boost'] = 1.0
        debug_info['content_boost'] = 1.0
        debug_info['final_boost'] = 1.0
        debug_info['mention_count'] = 0
        debug_info['year_diff'] = 'N/A'
        debug_info['reason'] = 'No year in filename'
        return 1.0, debug_info
    
    if earliest_year is None:
        # No earliest year found, use default boost
        debug_info['temporal_boost'] = 1.0
        debug_info['content_boost'] = 1.0
        debug_info['final_boost'] = 1.0
        debug_info['mention_count'] = 0
        debug_info['year_diff'] = 'N/A'
        debug_info['reason'] = 'No earliest year found'
        return 1.0, debug_info
    
    # Calculate temporal boost
    year_diff = doc_year - earliest_year
    temporal_boost = calculate_temporal_boost(year_diff)
    
    # Count keyword mentions in this document
    mention_count = count_keyword_mentions(doc.page_content, keywords)
    content_boost = calculate_content_boost(mention_count)
    
    # Combine multiplicatively
    final_boost = temporal_boost * content_boost
    
    # Debug info
    debug_info['doc_year'] = doc_year
    debug_info['earliest_year'] = earliest_year
    debug_info['year_diff'] = year_diff
    debug_info['temporal_boost'] = temporal_boost
    debug_info['mention_count'] = mention_count
    debug_info['content_boost'] = content_boost
    debug_info['final_boost'] = final_boost
    
    return final_boost, debug_info


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Document, float]], 
    semantic_results: List[Tuple[Document, float]], 
    query: str,
    doc_objects: List[Document],
    k: int = 5
) -> Tuple[List[Document], dict]:
    """
    Reciprocal Rank Fusion with Temporal + Content Boosting.
    
    Now returns: (documents, debug_info)
    
    RRF formula: score = sum(1 / (rank + 60)) * combined_boost
    combined_boost = temporal_boost * content_boost
    
    Temporal: Earlier mentions = higher boost (first mention = primary source)
    Content: More mentions = higher boost (more detail)
    """
    debug_info = {'boosts': []}
    
    # Extract keywords from query
    keywords = extract_keywords_from_query(query)
    debug_info['keywords'] = keywords
    
    # Find earliest year these keywords were mentioned
    earliest_year = find_earliest_mention_year(keywords, doc_objects)
    debug_info['earliest_mention_year'] = earliest_year
    
    scores = defaultdict(float)
    doc_map = {}
    doc_boost_info = {}
    
    # Process BM25 rankings
    for rank, (doc, score) in enumerate(bm25_results):
        doc_id = doc.page_content[:100]
        
        # Calculate combined boost
        combined_boost, boost_debug = get_combined_boost(doc, keywords, earliest_year)
        
        # Store boost info for top documents
        if rank < 10:  # Store debug info for top 10
            doc_boost_info[doc_id] = boost_debug
        
        scores[doc_id] += BM25_WEIGHT * (1.0 / (rank + 60)) * combined_boost
        doc_map[doc_id] = doc
    
    # Process semantic rankings
    for rank, (doc, score) in enumerate(semantic_results):
        doc_id = doc.page_content[:100]
        
        # Calculate combined boost (may already be cached from BM25)
        if doc_id not in doc_boost_info:
            combined_boost, boost_debug = get_combined_boost(doc, keywords, earliest_year)
            if rank < 10:
                doc_boost_info[doc_id] = boost_debug
        else:
            combined_boost = doc_boost_info[doc_id]['final_boost']
        
        scores[doc_id] += SEMANTIC_WEIGHT * (1.0 / (rank + 60)) * combined_boost
        
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
    
    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Store boost info for top results
    top_docs = [doc_map[doc_id] for doc_id, score in ranked[:k]]
    debug_info['boosts'] = [
        doc_boost_info.get(doc.page_content[:100], {}) 
        for doc in top_docs
    ]
    
    return top_docs, debug_info


def hybrid_search(
    query: str, 
    vectorstore, 
    bm25_index, 
    doc_objects: List[Document], 
    year: str = None,
    k: int = 5
) -> Tuple[List[Document], dict]:
    """
    Performs hybrid BM25 + Semantic search.
    Returns: (documents, debug_info)
    """
    debug_info = {}
    
    # Filter by year first if specified
    if year and doc_objects:
        filtered_docs = [
            doc for doc in doc_objects 
            if year in doc.metadata.get('source', '')
        ]
        
        if filtered_docs:
            debug_info['year_filter'] = f"Filtered to {len(filtered_docs)} docs from {year}"
            # Rebuild BM25 for filtered corpus
            filtered_corpus = [tokenize_text(doc.page_content) for doc in filtered_docs]
            year_bm25 = BM25Okapi(filtered_corpus)
            
            # Use filtered sets
            search_docs = filtered_docs
            search_bm25 = year_bm25
        else:
            debug_info['year_filter'] = f"No {year} docs found, using all docs"
            search_docs = doc_objects
            search_bm25 = bm25_index
    else:
        search_docs = doc_objects
        search_bm25 = bm25_index
    
    # BM25 search
    query_tokens = tokenize_text(query)
    bm25_scores = search_bm25.get_scores(query_tokens)
    
    # Get top BM25 results
    bm25_top_indices = sorted(
        range(len(bm25_scores)), 
        key=lambda i: bm25_scores[i], 
        reverse=True
    )[:HYBRID_K]
    
    bm25_results = [
        (search_docs[i], bm25_scores[i]) 
        for i in bm25_top_indices
    ]
    
    debug_info['bm25_top_score'] = bm25_scores[bm25_top_indices[0]] if bm25_top_indices else 0
    debug_info['bm25_results'] = len(bm25_results)
    
    # Semantic search
    if year:
        # For year queries, do filtered semantic search
        semantic_docs = vectorstore.similarity_search(query, k=100)
        semantic_docs = [
            doc for doc in semantic_docs 
            if year in doc.metadata.get('source', '')
        ][:HYBRID_K]
    else:
        semantic_docs = vectorstore.similarity_search(query, k=HYBRID_K)
    
    semantic_results = [(doc, 1.0) for doc in semantic_docs]
    debug_info['semantic_results'] = len(semantic_results)
    
    # Combine with RRF (now with temporal + content boosting)
    final_docs, rrf_debug = reciprocal_rank_fusion(
        bm25_results, 
        semantic_results, 
        query,  # Pass query for keyword extraction
        doc_objects,  # Pass all docs to find earliest mention
        k=k
    )
    
    # Merge RRF debug info
    debug_info.update(rrf_debug)
    debug_info['final_count'] = len(final_docs)
    
    return final_docs, debug_info


def evaluate_rag_relevance(query, rag_context, llm):
    """Evaluates if RAG context can answer the query."""
    evaluation_prompt = f"""You are evaluating whether retrieved context from Warren Buffett's letters and writings can answer a question.

Context (from Buffett's Partnership Letters and Berkshire Hathaway shareholder letters):
{rag_context[:3500]}

Question: {query}

Rate how well this context answers the question (1-10):

SCORING GUIDE:
- 1-3: Context is completely unrelated to the question
- 4-5: Context is on a related topic but doesn't directly address the question
- 6-7: Context contains relevant information that partially answers the question
- 8-9: Context has most/all information needed to answer well
- 10: Context completely and directly answers the question

GUIDELINES:
✓ If question asks about a SPECIFIC YEAR's letter AND context contains that year → score 8-10
✓ If question asks about Buffett's views AND context discusses that topic → score 7-10
✓ If question asks about a SPECIFIC COMPANY/INVESTMENT AND context discusses it → score 8-10
✓ Historical questions CAN be answered from historical documents

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


def process_query(query, vectorstore, bm25_index, doc_objects, search_tool, llm):
    """
    Intelligent query routing with hybrid search and debug output.
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
    
    # Extract year from query
    mentioned_year = extract_year_from_query(query)
    if mentioned_year:
        debug_messages.append(f"📅 Detected year in query: {mentioned_year}")
    
    if is_time_sensitive:
        matched_keywords = [k for k in time_keywords if k in query_lower]
        debug_messages.append(f"⏰ Detected time-sensitive query")
        debug_messages.append(f"📌 Matched keywords: {', '.join(matched_keywords)}")
        debug_messages.append(f"🔄 Skipping RAG → Going directly to web search")
        use_search = True
    
    try:
        if not use_search:
            debug_messages.append(f"📚 Non-time-sensitive query → Using Hybrid Search (BM25 + Semantic)")
            
            # Hybrid search
            rag_docs, hybrid_debug = hybrid_search(
                query, 
                vectorstore, 
                bm25_index, 
                doc_objects,
                year=mentioned_year,
                k=5
            )
            
            # Add hybrid search debug info
            if 'year_filter' in hybrid_debug:
                debug_messages.append(f"🎯 Year Filter: {hybrid_debug['year_filter']}")
            
            debug_messages.append(f"🔍 BM25 Results: {hybrid_debug.get('bm25_results', 0)} (top score: {hybrid_debug.get('bm25_top_score', 0):.2f})")
            debug_messages.append(f"🧠 Semantic Results: {hybrid_debug.get('semantic_results', 0)}")
            debug_messages.append(f"🔀 RRF Combined: {hybrid_debug.get('final_count', 0)} final documents")
            
            rag_context = "\n\n".join([doc.page_content for doc in rag_docs])
            
            debug_messages.append(f"📖 Retrieved {len(rag_docs)} documents from knowledge base")
            
            # Show what we retrieved
            if rag_docs:
                first_source = rag_docs[0].metadata.get('source', 'Unknown')
                source_name = os.path.basename(first_source) if first_source != 'Unknown' else 'Unknown'
                debug_messages.append(f"📄 Top source: {source_name}")
                
                # Show temporal + content boost info if available
                if 'earliest_mention_year' in hybrid_debug:
                    earliest = hybrid_debug['earliest_mention_year']
                    if earliest:
                        debug_messages.append(f"📅 Earliest mention: {earliest}")
                
                if 'keywords' in hybrid_debug:
                    keywords = hybrid_debug['keywords']
                    if keywords:
                        debug_messages.append(f"🔑 Keywords (filtered): {', '.join(keywords[:5])}")  # Show first 5
                
                # Show boost details for top document
                if 'boosts' in hybrid_debug and len(hybrid_debug['boosts']) > 0:
                    top_boost = hybrid_debug['boosts'][0]
                    if top_boost and 'final_boost' in top_boost:
                        temporal = top_boost.get('temporal_boost', 1.0)
                        content = top_boost.get('content_boost', 1.0)
                        final = top_boost.get('final_boost', 1.0)
                        mentions = top_boost.get('mention_count', 0)
                        year_diff = top_boost.get('year_diff', 'N/A')
                        
                        if final > 1.0:
                            debug_messages.append(f"⚡ Boost applied: {final:.2f}x (Temporal: {temporal:.1f}x × Content: {content:.1f}x)")
                            debug_messages.append(f"   └─ {mentions} keyword mentions, {year_diff} years from first mention")
                
                if mentioned_year and mentioned_year in source_name:
                    debug_messages.append(f"✅ Year filter SUCCESS - retrieved {mentioned_year} document")
            
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
1. IF the context says "No relevant information found", respond: 
   "I apologize, but I don't have sufficient information available to answer this question."

2. Start by briefly restating the question to confirm understanding

3. If the context is from Buffett's Partnership Letters or Berkshire Hathaway letters:
   - These are historical documents containing Buffett's actual writings
   - Quote or paraphrase them naturally
   - Cite the year when relevant (e.g., "In his 1960 letter, Buffett discussed...")
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

vectorstore, bm25_index, doc_objects, search_tool, llm = setup_rag_and_search()
if vectorstore is None:
    st.stop()

# Header
st.markdown(
    "<h1 style='text-align: center; font-size: 4.5em;'>🧠 Buffett's Brain</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='text-align: center; font-size: 2em; color: #AAAAAA;'>Hybrid Search with Temporal + Content Intelligence</h2>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Powered by Groq (Llama 3.1 8B), Tavily Search & HuggingFace Embeddings</p>", 
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚡ About")
    st.markdown(f"""
    **Buffett's Brain** combines:
    - 📚 **RAG Knowledge Base**: 
      - Partnership Letters (1957-1969)
      - Berkshire Hathaway Letters (1977-2024)
      - Poor Charlie's Almanack
      - **Total: 2,250+ pages, 8,518 chunks**
    - 🔍 **Hybrid Search Engine**:
      - **BM25**: Keyword/entity matching (40%)
      - **Semantic**: Conceptual understanding (60%)
      - **RRF Fusion**: Intelligent ranking combination
    - 🌐 **Real-Time Web Search**: Current data via Tavily
    
    **Recent Updates (v4.2):**
    - ✅ **NEW: Temporal + Content Boost System**
    - ✅ First mention = primary source (temporal priority)
    - ✅ More mentions = more detail (content depth)
    - ✅ Multiplicative combination: temporal × content
    - ✅ Automatic, no hardcoded rules, works for any entity
    
    **Tech Stack:**
    - LLM: Groq (Llama 3.1 8B)
    - Embeddings: HuggingFace (all-MiniLM-L6-v2)
    - Vector DB: Chroma ({len(doc_objects)} docs)
    - Keyword Search: BM25Okapi
    - Search: Tavily Advanced
    
    **Hybrid Weights:**
    - BM25 (Keyword): {int(BM25_WEIGHT*100)}%
    - Semantic: {int(SEMANTIC_WEIGHT*100)}%
    
    **Smart Boosting:**
    - Temporal: Earlier mention = 2.0x → 1.0x decay
    - Content: More mentions = 1.0 + (count × 0.1)
    - Final = Temporal × Content (up to ~10x boost!)
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

I now use **hybrid search** combining keyword matching (BM25) and semantic understanding for best-in-class retrieval.

**Try asking me:**
- 📖 "What did Buffett say about Sanborn Map Company?" ← Now works perfectly!
- 📊 "How did Buffett perform in 1960 versus the Dow Jones?"
- 💭 "What is Buffett's circle of competence principle?"
- 🏢 "Explain the concept of economic moats"
- 📰 "What is Berkshire's stock price today?" (web search)

*Smart retrieval: First mention gets priority, more detail gets boosted!* ⚡""")

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
        with st.spinner("🤔 Analyzing query with hybrid search..."):
            response, debug_info = process_query(
                prompt, vectorstore, bm25_index, doc_objects, search_tool, llm
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