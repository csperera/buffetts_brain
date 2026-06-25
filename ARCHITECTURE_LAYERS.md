# Buffett's Brain — ARCHITECTURE_LAYERS.md

Buffett's Brain is a production hybrid RAG system over 2,104 pages of Buffett and Munger content —
Berkshire Hathaway annual letters (1977–2024), Poor Charlie's Almanack, and Munger speeches and
transcripts. It answers questions about Buffett and Munger's investment philosophy by routing between
a static knowledge base and live web search based on query type, grounding every response in source
material with attribution. Live at buffettsbrain.net.

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    BUFFETT'S BRAIN — 7-LAYER HYBRID RAG PIPELINE                     ║
║          Data flow order: Ingest → Chunk → Embed → Store → Route → Retrieve →        ║
║                           Generate → Deliver                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — DOCUMENT INGESTION                                                        │
│                                                                                      │
│  IN:   Three source corpora of Buffett and Munger primary source material            │
│  OUT:  2,104 pages of raw text loaded into memory for processing                     │
│                                                                                      │
│  How it works:                                                                       │
│  → PyPDFDirectoryLoader("knowledge_base/docs/") loads all PDFs from three folders:   │
│       Berkshire_Letters/  — annual shareholder letters 1977–2024 (~1,900 pages)      │
│       Poor_Charlies_Almanack.pdf  — full book (~150 pages)                           │
│       Munger_Transcripts/  — speeches and interviews (~54 pages)                     │
│  → All 2,104 pages loaded as LangChain Document objects with page metadata           │
│  → download_data.py handles fetching source documents before processing              │
│                                                                                      │
│  Key decision:                                                                       │
│  → Primary source material only — letters, speeches, and books written by            │
│    Buffett and Munger themselves, not third-party commentary about them              │
│    Every answer grounded in their own words                                          │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │  2,104 pages as Document objects
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — CHUNKING AND EMBEDDING                                                    │
│                                                                                      │
│  IN:   2,104 pages of raw Document objects                                           │
│  OUT:  7,921 embedded chunks stored as 384-dimensional vectors                       │
│                                                                                      │
│  How it works:                                                                       │
│  → RecursiveCharacterTextSplitter splits documents into chunks:                      │
│       chunk_size=1000 characters                                                     │
│       chunk_overlap=200 characters — ensures context is not lost at chunk boundaries │
│       length_function=len                                                            │
│  → 2,104 pages → 7,921 chunks                                                        │
│  → HuggingFaceEmbeddings converts each chunk to a 384-dimensional vector:            │
│       model: sentence-transformers/all-MiniLM-L6-v2                                  │
│       runs locally — no API call, no cost, no quota, no network latency              │
│       ~90MB model download on first run, cached locally after                        │
│  → 7,921 chunks → 7,921 × 384-dimensional vectors ready for storage                  │
│                                                                                      │
│  Key decision — HuggingFace over OpenAI embeddings:                                  │
│  → Completely free — no API cost at any scale                                        │
│  → Runs locally — zero network latency at query time                                 │
│  → No quota limits — can re-embed entire corpus any time                             │
│  → Quality sufficient for domain-specific retrieval task                             │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │  7,921 × 384-dim vectors
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — VECTOR STORAGE (ChromaDB)                                                 │
│                                                                                      │
│  IN:   7,921 embedded chunk vectors + original chunk text + metadata                 │
│  OUT:  Persistent ChromaDB index at knowledge_base/vector_db/ ready for retrieval    │
│                                                                                      │
│  How it works:                                                                       │
│  → Chroma.from_documents(chunks, embedding_function,                                 │
│        persist_directory="knowledge_base/vector_db")                                 │
│  → ChromaDB stores three things per chunk:                                           │
│       The 384-dim vector — for similarity search                                     │
│       The original text — returned as context at query time                          │
│       Metadata — source document, page number — for attribution                      │
│  → Persisted locally to disk — survives restarts without re-embedding                │
│  → LangChain retriever wraps ChromaDB: retriever = vectorstore.as_retriever(k=4)     │
│  → k=4 means top 4 most similar chunks returned per query                            │
│                                                                                      │
│  Key decision — ChromaDB local over Pinecone cloud:                                  │
│  → Zero cost — no managed vector DB subscription                                     │
│  → Zero latency — no network round trip at query time                                │
│  → Sufficient for 7,921 chunks — ChromaDB handles this scale easily                  │
│  → Production upgrade path: migrate to Pinecone for multi-user scale                 │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │  persisted vector index ready
                                           ▼
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 4 — INTELLIGENT QUERY ROUTING (two-stage)                                     ║
║                                                                                      ║
║  IN:   Raw user query string                                                         ║
║  OUT:  Routing decision — RAG path or web search path                                ║
║                                                                                      ║
║  How it works — two stages:                                                          ║
║                                                                                      ║
║  STAGE 1 — TIME-SENSITIVE KEYWORD DETECTION (fast, cheap):                           ║
║  → Scan query for time-sensitive keywords:                                           ║
║       ['yesterday', 'today', 'current', 'price', 'now', 'latest', 'recent', ...]     ║
║  → If match found → skip RAG entirely → go straight to web search                    ║
║  → If no match → proceed to Stage 2                                                  ║
║                                                                                      ║
║  STAGE 2 — LLM RELEVANCE EVALUATION (intelligent, ~200ms overhead):                  ║
║  → Retrieve top 4 chunks from ChromaDB for the query                                 ║
║  → Send query + retrieved chunks to Groq Llama 3.1 8B with evaluation prompt:        ║
║       "Score 1-10 how well this context answers this question.                       ║
║        Return only a number."                                                        ║
║  → Score ≥ 5 → use RAG context for response                                          ║
║  → Score < 5 → discard RAG context → trigger Tavily web search                       ║
║                                                                                      ║
║  V0.7 vs V1.0:                                                                       ║
║  → V0.7: keyword-only routing — fast but brittle, misses nuanced queries             ║
║  → V1.0: keyword detection + LLM evaluation — handles nuance, ~200ms overhead        ║
║                                                                                      ║
║  Key decision — two stages not one:                                                  ║
║  → Keywords catch obvious time-sensitive queries instantly with zero cost            ║
║  → LLM evaluation handles the nuanced middle ground keywords cannot classify         ║
║  → Fallback design — if RAG cannot answer confidently, web search fills the gap      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
          RAG PATH (score ≥ 5                          WEB SEARCH PATH
          or no time-sensitive keyword)                (score < 5 or
                    │                                  time-sensitive)
                    │                                             │
                    ▼                                             ▼
┌───────────────────────────────────┐   ┌──────────────────────────────────────┐
│  LAYER 5a — RAG RETRIEVAL         │   │  LAYER 5b — WEB SEARCH RETRIEVAL     │
│                                   │   │                                      │
│  IN:   User query string          │   │  IN:   User query string             │
│  OUT:  Top 4 relevant chunks      │   │  OUT:  3 web search result snippets  │
│        with source attribution    │   │        with URLs                     │
│                                   │   │                                      │
│  How it works:                    │   │  How it works:                       │
│  → Embed query using HuggingFace  │   │  → Tavily Advanced Search API:       │
│    all-MiniLM-L6-v2 locally       │   │    tavily_search.invoke(query,       │
│    → 384-dim query vector         │   │      max_results=3)                  │
│  → ChromaDB cosine similarity     │   │  → Returns structured snippets       │
│    search across all 7,921 chunks │   │    with clean text and source URLs   │
│  → Top 4 chunks returned with     │   │  → Built specifically for LLM        │
│    original text and page source  │   │    consumption — cleaner than        │
│  → Combine into context string    │   │    Google Search API results         │
│    for prompt assembly            │   │  → Format results with URL           │
│                                   │   │    attribution for transparency      │
│  Key decision — k=4:              │   │                                      │
│  → Enough context for complex     │   │  Key decision — Tavily over Google:  │
│    philosophical questions        │   │  → Built for LLM applications        │
│  → Not so many chunks that        │   │  → Returns structured clean text     │
│    context window is overwhelmed  │   │  → More generous free tier           │
└────────────────┬──────────────────┘   └──────────────────┬───────────────────┘
                 │                                         │
                 └──────────────────┬──────────────────────┘
                                    │  context string (RAG chunks or web snippets)
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 6 — RESPONSE GENERATION                                                       ║
║                                                                                      ║
║  IN:   Context string (RAG chunks or web snippets) + user query                      ║
║  OUT:  Grounded natural language response with source attribution                    ║
║                                                                                      ║
║  How it works:                                                                       ║
║  → Prompt assembled in three parts:                                                  ║
║       System instruction: "You are Buffett's Brain. Answer using ONLY the provided   ║
║       context. Cite your sources. Do not invent information."                        ║
║       Retrieved context: RAG chunks or Tavily snippets                               ║
║       User question: raw query string                                                ║
║  → Sent to Groq API:                                                                 ║
║       model: llama-3.1-8b-instant                                                    ║
║       temperature: 0.0 — deterministic, no hallucination drift                       ║
║       max_tokens: 2048                                                               ║
║  → Groq inference: ~200ms latency — 10-100x faster than OpenAI GPT-4                 ║
║  → Response grounded in provided context, never from model memory alone              ║
║                                                                                      ║
║  Key decision — Groq Llama over OpenAI GPT-4:                                        ║
║  → ~200ms vs 2-5s inference latency — 10-100x faster                                 ║
║  → Cost-effective free tier sufficient for portfolio traffic                         ║
║  → temperature=0.0 enforces deterministic grounded responses                         ║
║  → Quality sufficient for domain Q&A — philosophical content not financial modeling  ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
                                           │  generated response string
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7 — RESPONSE DELIVERY                                                         │
│                                                                                      │
│  IN:   Generated response string + source metadata                                   │
│  OUT:  Rendered chat response with source attribution displayed to user              │
│                                                                                      │
│  How it works:                                                                       │
│  → Streamlit chat interface renders response in scrolling conversation format        │
│  → Source attribution displayed below response:                                      │
│       RAG path: source document name + page number from ChromaDB metadata            │
│       Web path: Tavily result URLs displayed as clickable links                      │
│  → Conversation history maintained within session — supports follow-up questions     │
│  → User can ask follow-up — loops back to Layer 4 routing with full context          │
│  → Two app versions available:                                                       │
│       app2.py — V0.7 keyword-based routing                                           │
│       app3.py — V1.0 LLM-based routing (production version)                          │
│                                                                                      │
│  Key decision — Streamlit over custom React frontend:                                │
│  → Zero frontend development time — focus on RAG architecture not UI                 │
│  → Sufficient for portfolio demonstration purposes                                   │
│  → Production upgrade: FastAPI backend + React frontend for scale                    │
└──────────────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════════════╗
║  FULL PIPELINE SUMMARY                                                               ║
║                                                                                      ║
║  KNOWLEDGE BASE BUILD (one-time):                                                    ║
║  PDFs → Layer 1 (ingest) → Layer 2 (chunk + embed) → Layer 3 (store ChromaDB)        ║
║                                                                                      ║
║  QUERY FLOW (every request):                                                         ║
║  Question → Layer 4 (route) → Layer 5a (RAG) or 5b (web) → Layer 6 (generate)        ║
║           → Layer 7 (deliver with attribution)                                       ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  TECH STACK                                                                          ║
║                                                                                      ║
║  Ingestion:    PyPDFDirectoryLoader (LangChain)                                      ║
║  Chunking:     RecursiveCharacterTextSplitter — 1,000 chars / 200 overlap            ║
║  Embeddings:   HuggingFace all-MiniLM-L6-v2 — 384-dim — runs locally                 ║
║  Vector DB:    ChromaDB — persistent local storage                                   ║
║  Routing:      Keyword detection + Groq Llama 3.1 8B relevance evaluation            ║
║  Web Search:   Tavily Advanced Search API — max_results=3                            ║
║  LLM:          Groq Llama 3.1 8B Instant — temperature=0.0 — ~200ms latency          ║
║  Frontend:     Streamlit — scrolling chat interface with source attribution          ║
║  Deployment:   Render — live at buffettsbrain.net                                    ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  KEY ARCHITECTURAL DECISIONS                                                         ║
║                                                                                      ║
║  HuggingFace embeddings locally  →  zero cost, zero latency, no quota limits         ║
║  ChromaDB local                  →  zero cost, zero latency, sufficient for scale    ║
║  Groq over OpenAI                →  10-100x faster inference, cost-effective         ║
║  Two-stage routing               →  keywords for speed, LLM for nuance               ║
║  temperature=0.0                 →  deterministic, grounded, no hallucination drift  ║
║  k=4 retrieval                   →  enough context, not too much                     ║
║  Primary sources only            →  every answer in Buffett and Munger's own words   ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  PERFORMANCE CHARACTERISTICS                                                         ║
║                                                                                      ║
║  Total documents:    2,104 pages                                                     ║
║  Total chunks:       7,921                                                           ║
║  Chunk size:         1,000 chars / 200 overlap                                       ║
║  Embedding dims:     384 (all-MiniLM-L6-v2)                                          ║
║  Retrieval count:    k=4 chunks per query                                            ║
║  LLM latency:        ~200ms (Groq Llama 3.1 8B)                                      ║
║  Total query time:   0.5–1.5s including retrieval and generation                     ║
║  Routing overhead:   ~200ms for LLM relevance evaluation                             ║ 
║  Web search latency: ~500ms (Tavily API)                                             ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

LEGEND
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  ╔══╗  ║    Core processing stage or routing decision boundary                       │
│  ┌──┐       Component, storage layer, or retrieval path                              │
│  ──►  │  ▼  Data flow direction                                                      │
│  IN / OUT   What enters and exits each layer                                         │
│  k=4        Top K chunks retrieved by ChromaDB similarity search                     │
│  384-dim    HuggingFace all-MiniLM-L6-v2 embedding dimension                         │
│  RAG        Retrieval-Augmented Generation — retrieve → augment → generate           │
│  Groq       Fast LLM inference platform — Llama 3.1 8B Instant                       │
│  Tavily     Web search API built specifically for LLM applications                   │
└──────────────────────────────────────────────────────────────────────────────────────┘