# 🏗️ Buffett's Brain - System Architecture & Technical Documentation

## Overview

This document provides a comprehensive technical view of Buffett's Brain, from initial setup through query processing with intelligent routing and response generation.

---

## 🎯 High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BUFFETT'S BRAIN V1.0                           │
│                    Hybrid RAG with Intelligent Routing                  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
            ┌───────▼────────┐              ┌──────────▼─────────┐
            │  Knowledge Base │              │   Real-Time Data   │
            │  (RAG Pipeline) │              │  (Web Search API)  │
            └───────┬────────┘              └──────────┬─────────┘
                    │                                   │
                    │        ┌──────────────┐          │
                    └────────►  LLM Router  ◄──────────┘
                             │ (Llama 3.1)  │
                             └──────┬───────┘
                                    │
                             ┌──────▼───────┐
                             │   Response   │
                             │  Synthesis   │
                             └──────────────┘
```

---

## 📊 Complete System Flowchart
```
                              [USER STARTS HERE]
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    PHASE 1: SETUP & INSTALLATION                    │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
            [Install Python 3.11+]              [Clone Repository]
            [Create venv]                       [Install dependencies]
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                            [Create .env file]
                    [Add GROQ_API_KEY & TAVILY_API_KEY]
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                 PHASE 2: KNOWLEDGE BASE CREATION                    │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   download_data.py      │
                        │   [Fetch Documents]     │
                        └─────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
    [Berkshire Letters]      [Poor Charlie's]         [Munger Speeches]
    [1977-2024 PDFs]         [Almanack PDF]           [Transcripts]
    [~1,900 pages]           [~150 pages]             [~54 pages]
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                         [2,104 total pages stored]
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │ process_documents.py    │
                        │ [Process & Embed]       │
                        └─────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
    [Load PDFs]              [Split into chunks]      [Generate embeddings]
    [PyPDFDirectoryLoader]   [1000 chars, 200         [HuggingFace]
                              overlap]                [all-MiniLM-L6-v2]
    [2,104 pages]            [7,921 chunks]           [384 dimensions]
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
                        [Store in ChromaDB Vector DB]
                        [knowledge_base/vector_db/]
                        [Persisted locally - no API calls]
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   PHASE 3: USER INTERACTION                         │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
            [streamlit run app3.py]         [streamlit run app2.py]
            (V1.0 - LLM Routing)            (V0.7 - Keyword Routing)
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                            [User Enters Query]
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              PHASE 4: INTELLIGENT QUERY ROUTING (V1.0)              │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │  Time-Sensitive Check   │
                        │  (yesterday, today,     │
                        │   current, price, etc.) │
                        └─────────────────────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                    YES │                           │ NO
                        ▼                           ▼
            ┌───────────────────┐       ┌───────────────────┐
            │  WEB SEARCH PATH  │       │   RAG PATH        │
            │  (Skip RAG)       │       │   (Check KB First)│
            └─────────┬─────────┘       └─────────┬─────────┘
                      │                           │
                      │                           ▼
                      │               ┌───────────────────────┐
                      │               │ 1. Retrieve from RAG  │
                      │               │    (k=4 chunks)       │
                      │               └───────────┬───────────┘
                      │                           │
                      │                           ▼
                      │               ┌───────────────────────┐
                      │               │ 2. LLM Evaluator      │
                      │               │    Score 1-10         │
                      │               │    (Llama 3.1 8B)     │
                      │               └───────────┬───────────┘
                      │                           │
                      │               ┌───────────┴───────────┐
                      │               │                       │
                      │           Score ≥5?              Score <5?
                      │               │                       │
                      │               ▼                       ▼
                      │       [USE RAG CONTEXT]      [TRIGGER SEARCH]
                      │               │                       │
                      └───────────────┴───────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    PHASE 5: INFORMATION RETRIEVAL                   │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                        ▼                           ▼
            ┌───────────────────┐       ┌───────────────────┐
            │   RAG RETRIEVAL   │       │   WEB SEARCH      │
            └───────────────────┘       └───────────────────┘
                        │                           │
                        ▼                           ▼
            [Embed query with]          [Tavily Advanced Search]
            [HuggingFace model]         [max_results=3]
                        │                           │
                        ▼                           ▼
            [Search ChromaDB]           [Return search snippets]
            [Semantic similarity]       [with URLs]
                        │                           │
                        ▼                           ▼
            [Top 4 chunks from]         [Format results with]
            [Buffett/Munger docs]       [source attribution]
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                            [Combine Retrieved Context]
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   PHASE 6: RESPONSE GENERATION                      │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   Format Prompt         │
                        │   • System instruction  │
                        │   • Retrieved context   │
                        │   • User question       │
                        └─────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   Groq API              │
                        │   Llama 3.1 8B Instant  │
                        │   (temperature=0.0)     │
                        └─────────────────────────┘
                                      │
                                      ▼
                        [Generate Response]
                        [Grounded in sources]
                        [~200ms latency]
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   PHASE 7: RESPONSE DELIVERY                        │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                        [Display in Streamlit]
                        [Scrolling chat interface]
                        [With source attribution]
                                      │
                                      ▼
                        [User Can Ask Follow-up]
                                      │
                                      ▼
                        [Loop back to Query Input]
```

---

## 🔑 Key Architectural Decisions

### Decision 1: Groq over OpenAI/Anthropic

**Rationale:**
- ⚡ 10-100x faster inference (~200ms vs 2-5s)
- 💰 Cost-effective free tier
- 🔧 No dependency conflicts
- 📈 Sufficient quality for domain tasks

**Tradeoff:** Slightly less capable than GPT-4, but speed makes up for it in user experience.

### Decision 2: HuggingFace Embeddings over OpenAI

**Rationale:**
- 🆓 Completely free (no API costs)
- 🏠 Runs locally (no network latency)
- 🚫 No quota limits
- ✅ Excellent quality (384-dim all-MiniLM-L6-v2)

**Tradeoff:** Initial model download (~90MB), but cached after first run.

### Decision 3: Hybrid Routing (Time-Sensitive + LLM Evaluation)

**Rationale:**
- 🎯 Keywords catch obvious time-sensitive queries (yesterday, today, price)
- 🧠 LLM evaluation handles nuanced cases
- 🛡️ Fallback strategy prevents false negatives

**Tradeoff:** Adds ~200ms per query for evaluation, but worth it for accuracy.

### Decision 4: Tavily over Google Search API

**Rationale:**
- 🤖 Built specifically for LLM applications
- 📊 Returns structured, clean snippets
- 💵 More generous free tier
- 📝 Better result quality for text-based queries

**Tradeoff:** Doesn't extract structured data (stock prices) - acknowledged limitation.

---

## 🔬 RAG Pipeline: Technical Deep Dive

### Component Specifications

| Component | Technology | Configuration | Purpose |
|-----------|-----------|---------------|---------|
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | 384 dimensions | Convert text to semantic vectors |
| **Vector DB** | ChromaDB | Persistent storage | Fast similarity search |
| **Retriever** | LangChain | k=4 chunks | Fetch most relevant context |
| **LLM (Main)** | Groq Llama 3.1 8B | temp=0.0, max_tokens=2048 | Generate final responses |
| **LLM (Evaluator)** | Groq Llama 3.1 8B | temp=0.0, max_tokens=100 | Score RAG relevance |
| **Search** | Tavily Advanced | max_results=3 | Real-time web data |

### Document Processing Pipeline
```python
# Step 1: Load Documents
PyPDFDirectoryLoader("knowledge_base/docs/")
    → 2,104 pages loaded

# Step 2: Chunk Documents
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
    → 7,921 chunks created

# Step 3: Generate Embeddings
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    → 7,921 × 384-dimensional vectors

# Step 4: Store in Vector DB
Chroma.from_documents(
    chunks,
    embedding_function,
    persist_directory="knowledge_base/vector_db"
)
    → Indexed and persisted locally
```

### Query Processing Pipeline (V1.0)
```python
# Stage 1: Time-Sensitive Detection
time_keywords = ['yesterday', 'today', 'current', 'price', ...]
is_time_sensitive = any(keyword in query.lower() for keyword in time_keywords)

if is_time_sensitive:
    # Skip RAG, go straight to search
    use_search = True
else:
    # Stage 2: RAG Retrieval
    rag_docs = retriever.invoke(query)  # k=4 chunks
    rag_context = combine(rag_docs)
    
    # Stage 3: LLM Evaluation
    relevance_score = evaluate_rag_relevance(query, rag_context, llm)
    # Returns 1-10 score
    
    # Stage 4: Routing Decision
    if relevance_score >= 5:
        use_rag = True
    else:
        use_search = True  # Fallback

# Stage 5: Execute Search (if needed)
if use_search:
    search_results = tavily_search.invoke(query)
    context = format(search_results)

# Stage 6: Generate Response
prompt = format_prompt(context, query)
response = groq_llm.invoke(prompt)
```

---

## 🎨 Version Comparison

### V0.7 (app2.py) - Keyword-Based Routing
```python
search_keywords = ['current', 'today', 'price', ...]
rag_keywords = ['buffett', 'munger', 'philosophy', ...]

if any(keyword in query for keyword in search_keywords):
    use_search = True
elif any(keyword in query for keyword in rag_keywords):
    use_rag = True
else:
    use_rag = True  # Default
```

**Pros:** Simple, fast, predictable  
**Cons:** Brittle, misses nuanced queries, hard to maintain

### V1.0 (app3.py) - LLM-Based Routing
```python
# Explicit time-sensitive detection
if is_time_sensitive(query):
    use_search = True
else:
    # LLM evaluates if RAG can answer
    score = llm_evaluate(query, rag_context)  # 1-10
    
    if score >= 5:
        use_rag = True
    else:
        use_search = True
```

**Pros:** Intelligent, handles nuance, adapts to query intent  
**Cons:** Adds ~200ms latency, requires extra LLM call

---

## 📈 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Documents** | 2,104 pages | Buffett/Munger corpus |
| **Total Chunks** | 7,921 | After text splitting |
| **Chunk Size** | 1,000 chars | With 200-char overlap |
| **Embedding Dimension** | 384 | all-MiniLM-L6-v2 |
| **Retrieval Count** | k=4 | Top chunks per query |
| **LLM Latency** | ~200ms | Groq Llama 3.1 8B |
| **Total Query Time** | 0.5-1.5s | Including retrieval + generation |
| **Evaluation Overhead** | ~200ms | LLM relevance scoring |
| **Search Latency** | ~500ms | Tavily API call |

---

## 🛡️ Security & Best Practices

### API Key Management
- ✅ Stored in `.env` file (never committed)
- ✅ `.gitignore` prevents accidental exposure
- ✅ `.env.example` provided as template
- ⚠️ Users must obtain their own keys

### Data Privacy
- ✅ Vector DB stored locally
- ✅ No document data sent to third parties
- ✅ Only queries sent to LLM APIs
- ✅ HuggingFace embeddings run locally

### Code Quality
- ✅ Comprehensive test suite (pytest)
- ✅ Type hints where appropriate
- ✅ Docstrings for all functions
- ✅ Error handling with graceful fallbacks

---

## 🔄 Data Flow Summary

### RAG Query Flow
```
User Query
    ↓
Embed with HuggingFace (local)
    ↓
Search ChromaDB (similarity)
    ↓
Retrieve Top 4 Chunks
    ↓
LLM Evaluates Relevance (1-10)
    ↓
If score ≥5: Use RAG context
If score <5: Trigger web search
    ↓
Format Prompt with Context
    ↓
Groq Llama 3.1 8B Generation
    ↓
Display Response to User
```

### Time-Sensitive Query Flow
```
User Query (contains: yesterday, today, price, etc.)
    ↓
Detect Time-Sensitive Keywords
    ↓
Skip RAG Evaluation
    ↓
Tavily Web Search (max_results=3)
    ↓
Format Search Results
    ↓
Groq Llama 3.1 8B Generation
    ↓
Display Response to User
```

---

## 📁 File Structure
```
buffetts_brain/
├── .env                                # API keys (gitignored)
├── .env.example                        # Template for users
├── .gitignore                          # Git exclusions
├── LICENSE                             # MIT License
├── README.md                           # Main documentation
├── ARCHITECTURE.md                     # This file
├── CONTRIBUTING.md                     # Contribution guidelines
├── CHANGELOG.md                        # Version history
├── requirements.txt                    # Python dependencies
├── download_data.py                    # Document fetcher
├── process_documents.py                # Document processor (HuggingFace embeddings)
├── src/
│   ├── app2.py                         # V0.7 (keyword-based routing)
│   └── app3.py                         # V1.0 (LLM-based routing) ⭐
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures
│   ├── test_rag_pipeline.py            # RAG tests
│   ├── test_query_routing.py          # Routing logic tests
│   ├── test_embeddings.py              # Embedding tests
│   ├── test_integration.py             # End-to-end tests
│   ├── requirements-test.txt           # Test dependencies
│   └── README.md                       # Test documentation
└── knowledge_base/
    ├── docs/                           # Source documents
    │   ├── Berkshire_Letters/          # 1977-2024 annual letters
    │   ├── Munger_Transcripts/         # Speeches & transcripts
    │   └── Poor_Charlies_Almanack.pdf  # Full book
    └── vector_db/                      # ChromaDB storage (gitignored)
```

---

## 🚀 Deployment Considerations

### Local Development
- ✅ Works out of the box with free tier APIs
- ✅ Vector DB stored locally (no cloud required)
- ✅ HuggingFace models cached locally

### Production Deployment (Future V2.0)
**Recommended Stack:**
- **Backend**: FastAPI for API endpoints
- **Vector DB**: Keep Chroma or migrate to Pinecone for scale
- **LLM**: Groq or self-hosted Llama
- **Search**: Dedicated financial APIs (Alpha Vantage, Polygon.io)
- **Hosting**: AWS/GCP with Docker containers
- **Monitoring**: LangSmith for tracing

---

## 🐛 Known Limitations & Workarounds

### Limitation 1: Web Search Snippets
**Issue**: Tavily returns references to data sources, not raw numerical data  
**Workaround**: Acknowledge limitation in response, provide URLs  
**V2 Solution**: Integrate Alpha Vantage or Polygon.io for structured financial data

### Limitation 2: LLM Context Ignoring
**Issue**: LLM occasionally gives generic advice despite provided context  
**Workaround**: Very explicit prompting ("USE THE DATA PROVIDED")  
**V2 Solution**: Fine-tune model on financial Q&A pairs

### Limitation 3: Relevance Evaluation Strictness
**Issue**: Early versions scored too generously (historical info for current queries)  
**Workaround**: Explicit time-sensitive keyword detection + strict evaluation criteria  
**V2 Solution**: Train a specialized classifier for "can RAG answer this?"

### Limitation 4: Knowledge Base Freshness
**Issue**: RAG only knows documents up to processing date  
**Workaround**: Re-run `process_documents.py` periodically  
**V2 Solution**: Automated document fetching + incremental indexing

---

## 🔮 Future Architecture (V2.0 Roadmap)

### Multi-Agent System
```
                    [User Query]
                         │
                         ▼
            ┌────────────────────────┐
            │    Router Agent        │
            │  (Classify query type) │
            └────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  RAG Agent   │  │ Search Agent │  │Financial Agent│
│ (Historical) │  │ (Current)    │  │ (Numerical)   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
            ┌────────────────────────┐
            │  Synthesizer Agent     │
            │  (Combine & Format)    │
            └────────────────────────┘
```

### Fine-Tuning Strategy

1. **Collect Training Data**: Log queries + preferred responses
2. **Create Q&A Pairs**: Format as instruction-tuning dataset
3. **Fine-Tune Llama 3.1 8B**: On financial domain + Buffett style
4. **A/B Test**: Compare base model vs fine-tuned
5. **Iterate**: Continuous improvement loop

### Financial API Integration
```python
# V2 Example
if query_about_stock_price(query):
    # Use Alpha Vantage instead of web search
    price_data = alpha_vantage.get_quote(ticker)
    context = format_price_data(price_data)
else:
    # Use RAG as before
    context = rag_retrieval(query)
```

---

## 🎓 Learning Resources

### Understanding This Architecture

- **RAG Basics**: [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- **Vector Databases**: [ChromaDB Documentation](https://docs.trychroma.com/)
- **Embeddings**: [Sentence Transformers Guide](https://www.sbert.net/)
- **Groq API**: [Groq Documentation](https://console.groq.com/docs)

---

## 📞 Questions About Architecture?

For questions about architectural decisions, tradeoffs, or implementation details:
- See [README.md](README.md) for high-level overview
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub for specific technical questions

---

**Built with ❤️ and rigorous engineering by Cristian Perera**

*"The best investment you can make is in yourself." - Warren Buffett*