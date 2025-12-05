# 🧠 Buffett's Brain - Agentic AI RAG-Enhanced Knowledge Base System

> A production-grade Agentic AI Retrieval-Augmented Generation (RAG) knowledge base system that combines 50 years of Warren Buffett and Charlie Munger's investment wisdom with real-time web search capabilities.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Click To Watch Demo Video](https://img.shields.io/badge/▶️-Click%20To%20Watch%20Demo%20Video-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=Y5FleSE2BBo)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Knowledge Base](#knowledge-base)
- [Installation](#installation)
- [Usage](#usage)
- [Major Discoveries & Learnings](#major-discoveries--learnings)
- [Limitations & Future Work](#limitations--future-work)
- [Technical Stack](#technical-stack)
- [Demo Queries](#demo-queries)
- [Project Journey](#project-journey)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Buffett's Brain** is an AI-powered assistant that provides expert investment insights by combining:
- **Deep historical knowledge**: 2,100+ pages of Buffett/Munger writings processed via RAG
- **Real-time information**: Web search integration for current market data
- **Intelligent routing**: LLM-based relevance scoring to choose the optimal information source

This project demonstrates cutting-edge RAG architecture with hybrid retrieval strategies, making it an excellent portfolio piece for ML/AI engineering roles.

---

## ✨ Key Features

### 🧠 **Intelligent Query Routing**
- **Time-Sensitive Detection**: Automatically detects queries about current events, prices, or recent news
- **LLM-Based Relevance Scoring**: Uses a fast LLM (Llama 3.1 8B) to evaluate if RAG context can answer the query (1-10 score)
- **Graceful Fallback**: Switches to web search when RAG relevance is insufficient (<5/10)

### 📚 **Comprehensive Knowledge Base**
- 50 years of Berkshire Hathaway annual letters (1977-2024)
- Poor Charlie's Almanack
- Daily Journal meeting transcripts (2018-2023)
- Key speeches: "Psychology of Human Misjudgment", USC Commencement 2007
- **Total: 2,104 document pages → 7,921 semantic chunks**

### ⚡ **Blazing Fast Performance**
- **Groq API**: Lightning-fast inference (~200ms response times)
- **Efficient Embeddings**: HuggingFace all-MiniLM-L6-v2 (free, local)
- **Optimized Chunking**: 1000-character chunks with 200-character overlap

### 🔍 **Hybrid Search**
- RAG retrieval for historical wisdom
- Tavily advanced search for real-time data
- Intelligent source selection per query

---

## 🏗️ Architecture
```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
    ┌────▼────────────────────────┐
    │ Time-Sensitive Detection?   │
    │ (yesterday, current, price) │
    └─────┬──────────────┬────────┘
          │              │
      YES │              │ NO
          │              │
    ┌─────▼──────┐  ┌───▼────────────────┐
    │ Web Search │  │ RAG Retrieval      │
    │  (Tavily)  │  │ (Vector DB)        │
    └─────┬──────┘  └───┬────────────────┘
          │              │
          │         ┌────▼──────────────┐
          │         │ LLM Evaluator     │
          │         │ (Relevance 1-10)  │
          │         └────┬──────────────┘
          │              │
          │         Score < 5? ──YES──┐
          │              │             │
          │             NO             │
          │              │             │
    ┌─────▼──────────────▼─────────────▼──┐
    │   Combine Context + Generate Answer  │
    │         (Groq Llama 3.1 8B)          │
    └──────────────┬───────────────────────┘
                   │
            ┌──────▼───────┐
            │   Response   │
            └──────────────┘
```

### System Components

1. **Query Router**: Detects time-sensitive keywords to trigger immediate search
2. **RAG Pipeline**: Retrieves relevant chunks from vector database
3. **LLM Evaluator**: Scores RAG context relevance (1-10 scale)
4. **Search Fallback**: Executes Tavily search for low-relevance or time-sensitive queries
5. **Response Generator**: Synthesizes final answer from best available source

---

## 📚 Knowledge Base

### Sources Processed

| Source | Years | Pages | Description |
|--------|-------|-------|-------------|
| Berkshire Hathaway Annual Letters | 1977-2024 | ~1,900 | Complete archive of shareholder letters |
| Poor Charlie's Almanack | 2005 | ~150 | Charlie Munger's wisdom and speeches |
| Daily Journal Transcripts | 2018-2023 | ~40 | Q&A sessions from annual meetings |
| Key Speeches | Various | ~14 | Psychology of Human Misjudgment, etc. |

**Total**: 2,104 pages → 7,921 semantic chunks

### Processing Pipeline
```bash
# 1. Download documents
python download_data.py

# 2. Process and embed
python process_documents.py
```

The pipeline:
1. Loads PDFs from `knowledge_base/docs/`
2. Splits into 1000-char chunks (200-char overlap)
3. Generates embeddings via HuggingFace
4. Stores in Chroma vector database
5. Persists to `knowledge_base/vector_db/`

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- API Keys (free tiers available):
  - [Groq API Key](https://console.groq.com)
  - [Tavily API Key](https://tavily.com)

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/buffetts-brain.git
cd buffetts-brain

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# 5. Download and process documents
python download_data.py
python process_documents.py

# 6. Run the application
streamlit run src/app3.py
```

### Requirements.txt
```
streamlit>=1.28.0
langchain>=0.1.0
langchain-groq>=0.1.0
langchain-huggingface>=0.0.1
langchain-chroma>=0.1.0
langchain-tavily>=0.1.0
langchain-community>=0.0.20
python-dotenv>=1.0.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
pypdf>=3.17.0
```

---

## 💻 Usage

### Running the App
```bash
streamlit run src/app3.py
```

The app will open in your browser at `http://localhost:8501`

## 🚀 Running the Application

### Version 3.0 (Recommended - LLM-Based Routing)
```bash
streamlit run src/app3.py
```
Features intelligent relevance scoring and hybrid retrieval.

### Version 2.0 (Stable - Keyword-Based Routing)
```bash
streamlit run src/app2.py
```
Simpler keyword-based approach, useful for comparison.

### Example Queries

**Historical Knowledge (RAG):**
```
"What is Buffett's circle of competence principle?"
"Explain Charlie Munger's views on cryptocurrency"
"What are the key characteristics of an economic moat?"
```

**Time-Sensitive (Web Search):**
```
"What is the current stock price of Berkshire Hathaway?"
"Recent news about Berkshire Hathaway acquisitions"
"What happened at yesterday's shareholder meeting?"
```

**Hybrid Queries:**
```
"Should I invest in Apple today based on Buffett's criteria?"
"How would Munger evaluate Tesla's current valuation?"
```

---

## 🔬 Major Discoveries & Learnings

### 1. **The LLM Evaluator Challenge**

**Discovery**: Initial keyword-based routing was too brittle and missed nuanced queries.

**Solution**: Implemented LLM-based relevance scoring where a fast model (Llama 3.1 8B) evaluates if RAG context can answer the query on a 1-10 scale.

**Key Insight**: The evaluator needed to be VERY strict. Early versions scored ≥6 even when RAG had general company info but the query asked for specific current prices. We had to:
- Make scoring criteria extremely explicit
- Emphasize the difference between historical vs. current data
- Lower the threshold from 6 to 5
- Eventually add explicit time-sensitive keyword detection as a failsafe

**Code Evolution**:
```python
# V1: Brittle keyword matching
if 'price' in query: use_search = True

# V2: LLM-based evaluation (too generous)
score = evaluate_rag_relevance(query, context, llm)
if score >= 6: use_rag()

# V3: Stricter evaluation + hybrid approach
if is_time_sensitive(query): 
    use_search = True  # Skip RAG entirely
else:
    score = strict_evaluate(query, context, llm)
    if score >= 5: use_rag()
    else: use_search()
```

### 2. **Search Results ≠ Actual Data**

**Discovery**: Tavily Search returns **snippets and summaries** about web pages, not the actual page content. 

**Example**: When asking "What's Apple's stock price today?", Tavily returns:
```
"You can find Apple's stock price on Yahoo Finance at finance.yahoo.com"
```

But NOT:
```
"Apple's current stock price is $175.23"
```

**Why This Happens**: 
- Search engines return metadata and descriptions
- Stock prices are embedded in page elements (tables, charts)
- Would need to fetch and parse the actual HTML

**Attempted Solutions**:
1. ❌ Stronger prompting ("USE THE DATA PROVIDED") → LLM still gave generic advice
2. ❌ Better formatting of search results → Still no raw data to format
3. ✅ Acknowledged limitation in UI and prompt → Honest about capabilities

**Key Learning**: For production systems requiring real-time numerical data (stock prices, sports scores, etc.), **dedicated APIs are essential**. Web search is excellent for news, articles, and text-based content, but not structured data extraction.

### 3. **The Stubborn Toddler Problem**

**Discovery**: Even when providing search results in context, the LLM would sometimes "think it knows better" and give generic advice like "check Yahoo Finance" instead of using the search results we provided.

**Solution**: Extremely forceful prompting:
```python
prompt = """
IMPORTANT: 
1. If web search results are provided, USE THEM directly
2. DO NOT give generic advice to "check Yahoo Finance"
3. If search results lack specific data, acknowledge the limitation
"""
```

**Analogy**: Training an LLM is like training a 3-year-old who thinks they're smarter than you! 😂

### 4. **Dependency Hell with Google Gemini**

**Original Plan**: Use Google's Gemini API for the LLM.

**Reality**: Spent hours battling version conflicts:
```
langchain-google-genai 3.2.0 requires google-ai-generativelanguage>=0.9.0
but google-generativeai 0.8.5 requires google-ai-generativelanguage==0.6.15
ERROR: ResolutionImpossible
```

**Solution**: Pivoted to Groq API with Llama 3.1
- ✅ No dependency conflicts
- ✅ Faster inference (~200ms vs 2-5s for Gemini)
- ✅ Free tier is generous
- ✅ Better documentation

**Lesson**: Choose tools with stable, well-maintained packages. Sometimes the "hot new thing" isn't production-ready.

### 5. **Embeddings: Free > Paid**

**Discovery**: Switched from Google's `text-embedding-004` (quota issues) to HuggingFace's `all-MiniLM-L6-v2` (free, local).

**Result**: 
- ✅ No API calls for embeddings
- ✅ No quota limits
- ✅ Faster processing (local inference)
- ✅ Still excellent quality (384 dimensions)

**Tradeoff**: Initial model download (~90MB), but cached locally after first run.

---

## ⚠️ Limitations & Future Work

### Current Limitations

#### 1. **Real-Time Data Accuracy**
**Issue**: Web search returns references to data sources, not always the actual data itself.

**Example**: 
- ❌ Can't reliably return exact stock prices
- ✅ Can return news articles, analysis, and context

**Impact**: Time-sensitive queries about specific numbers may not get exact answers.

#### 2. **Search Result Quality**
**Issue**: Tavily search quality varies by query type.

**Works Well**: News, articles, company information, general facts

**Works Poorly**: Structured data (prices, scores), data behind authentication, very recent events (<1 hour)

#### 3. **LLM Hallucination Risk**
**Issue**: Even with RAG, LLMs can occasionally generate plausible-sounding but incorrect information.

**Mitigation**: 
- Relevance scoring helps (only uses RAG when score ≥5)
- Citations from sources included in responses
- User should verify critical information

#### 4. **Knowledge Base Freshness**
**Issue**: RAG knowledge only includes documents up to processing date (late 2024 in this version).

**Solution**: Re-run `download_data.py` and `process_documents.py` periodically to update.

---

### 🚀 Version 2.0 Roadmap

**FINE-TUNING** is now a major focus based on our discoveries:

#### Priority 1: Dedicated Financial API Integration
**Why**: Web search snippets don't contain raw numerical data

**Solution**: Integrate dedicated APIs:
- Alpha Vantage or Polygon.io for stock prices
- News API for financial news
- SEC EDGAR API for filings

**Benefit**: Get actual real-time data, not references to data sources

#### Priority 2: LLM Fine-Tuning for Financial Domain
**Why**: Generic LLMs sometimes provide unhelpful generic advice

**Approach**:
- Fine-tune Llama 3.1 8B on financial Q&A pairs
- Train on Buffett/Munger response patterns
- Improve citation habits and data usage

**Expected Improvement**: More "Buffett-like" responses, better use of provided context

#### Priority 3: Multi-Agent Architecture
**Current**: Single LLM handles routing, evaluation, and generation

**Proposed**: Specialized agents:
- **Router Agent**: Classifies query type
- **RAG Agent**: Handles knowledge base queries
- **Search Agent**: Handles current events
- **Financial Agent**: Specialized for numerical/analytical queries
- **Synthesizer Agent**: Combines outputs

**Benefit**: Each agent optimized for its specific task

#### Priority 4: Improved Relevance Evaluation
**Current**: Single LLM call to evaluate relevance

**Proposed**:
- Use semantic similarity scores from vector DB
- Combine with LLM evaluation
- Train a small classifier specifically for "can RAG answer this?" task
- A/B test different threshold values

#### Priority 5: Enhanced Search Strategy
**Options**:
- Try multiple search engines (Tavily + Google + Bing)
- Implement `web_fetch` to actually retrieve page content
- Parse structured data from known financial sites
- Cache recent search results to reduce API calls

#### Priority 6: User Feedback Loop
- Add thumbs up/down to responses
- Log queries where RAG/search selection was suboptimal
- Build training data for fine-tuning
- Continuously improve routing logic

---

## 🛠️ Technical Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq (Llama 3.1 8B Instant) | Ultra-fast inference, query evaluation, response generation |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Free, local, 384-dim semantic embeddings |
| **Vector DB** | Chroma | Lightweight, persistent vector storage |
| **Web Search** | Tavily Advanced | Deep web search with snippets |
| **Framework** | LangChain | RAG orchestration, chain building |
| **UI** | Streamlit | Interactive web interface |
| **Document Processing** | PyPDF, LangChain Loaders | PDF parsing and chunking |

### Why These Choices?

**Groq over OpenAI/Anthropic**:
- 10-100x faster inference
- Free tier is generous
- Cost-effective for demos

**HuggingFace Embeddings over OpenAI**:
- Completely free
- Runs locally (no API calls)
- No quota limits
- Excellent quality for general domain

**Chroma over Pinecone/Weaviate**:
- Lightweight, easy setup
- Persists locally (no cloud required)
- Perfect for demos and prototypes
- Can scale to production if needed

**Tavily over Google Search API**:
- Built for LLM applications
- Returns clean, structured results
- More generous free tier
- Better snippet quality

---

## 🎯 Demo Queries

### Category 1: Pure RAG (Historical Wisdom)
```
Q: "What is Buffett's circle of competence principle?"
Expected: Detailed explanation with citations from annual letters

Q: "What would Charlie Munger say about cryptocurrency?"
Expected: Synthesis of Munger's views on speculation, understanding, and mental models

Q: "Explain the concept of an economic moat"
Expected: Buffett's framework for sustainable competitive advantages
```

### Category 2: Web Search (Time-Sensitive)
```
Q: "What is Berkshire Hathaway's stock price today?"
Expected: Search results with links to financial sites (note: may not return exact price)

Q: "Recent news about Berkshire Hathaway"
Expected: Current news articles and summaries

Q: "What happened at yesterday's shareholder meeting?"
Expected: News coverage and summaries of recent events
```

### Category 3: Hybrid Queries
```
Q: "Should I invest in Apple today based on Buffett's criteria?"
Expected: RAG context about Buffett's investment criteria + search for Apple's current status

Q: "How does Amazon's moat compare to what Buffett looks for?"
Expected: RAG explanation of moat criteria + search for Amazon's current competitive position
```

### Category 4: Edge Cases (Good for Demo!)
```
Q: "What does Buffett think about Tesla?"
Expected: High relevance score if he's written about it, otherwise low score → search

Q: "Should I buy crypto?"
Expected: Munger's wisdom on speculation + acknowledgment of limitations

Q: "What's the weather in Omaha?"
Expected: Honest response that this is outside knowledge domain
```

---

## 📊 Project Journey

This project evolved through multiple iterations, each teaching valuable lessons about production ML systems:

### Phase 1: Initial RAG (v0.1-0.3)
- ✅ Built basic RAG pipeline
- ✅ Processed 2,100 pages of documents
- ❌ No real-time data capability
- ❌ Hallucinated answers for current events

**Learning**: Pure RAG has clear boundaries - excellent for historical knowledge, fails for current data

### Phase 2: Adding Web Search (v0.4-0.5)
- ✅ Integrated Tavily search
- ✅ Keyword-based routing
- ❌ Brittle routing logic
- ❌ Search not triggering reliably

**Learning**: Keyword matching is too simplistic for natural language queries

### Phase 3: LLM-Based Routing (v0.6-0.7)
- ✅ Implemented relevance scoring
- ✅ Groq integration (after Gemini issues)
- ❌ Evaluator too generous
- ❌ Search still not triggering

**Learning**: LLMs need very explicit, strict instructions. "Somewhat related" ≠ "can answer the question"

### Phase 4: Hybrid Approach (v0.8-1.0)
- ✅ Time-sensitive keyword detection
- ✅ Stricter relevance evaluation
- ✅ Improved prompting
- ✅ Discovered search snippet limitations
- ✅ Documented everything for V2

**Learning**: Production systems need multiple fallback strategies. Acknowledge limitations honestly.

---

## 🎓 What We're Proud Of

### Technical Achievements
✅ **Production-grade RAG system** with intelligent routing
✅ **7,921 semantic chunks** from 50 years of investment wisdom
✅ **Sub-second response times** thanks to Groq
✅ **Hybrid retrieval** strategy with graceful fallbacks
✅ **LLM-based evaluation** for dynamic routing decisions

### Engineering Process
✅ **Systematic debugging** through multiple iterations
✅ **Honest acknowledgment** of limitations
✅ **Clear documentation** of discoveries and lessons learned
✅ **Production mindset** from day one
✅ **V2 roadmap** based on real-world findings

### Domain Expertise
✅ **Comprehensive knowledge base** covering value investing fundamentals
✅ **50 years** of Buffett/Munger wisdom at our fingertips
✅ **Semantic search** across complex financial concepts
✅ **Citation-backed responses** maintaining accuracy

---

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome!

**Areas for Contribution:**
- Financial API integrations
- Fine-tuning datasets
- UI/UX improvements
- Additional document sources
- Testing and benchmarking

**How to Contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Warren Buffett & Charlie Munger** for 50 years of timeless investment wisdom
- **Groq** for blazing-fast LLM inference
- **HuggingFace** for free, high-quality embeddings
- **LangChain** for excellent RAG orchestration tools
- **Streamlit** for making beautiful UIs simple
- **The AI/ML community** for countless tutorials and examples

---

## 📧 Contact

**Cristian Perera** - https://www.linkedin.com/in/christianperera/

Project Link: [https://github.com/csperera/buffetts-brain](https://github.com/yourusername/buffetts-brain)

---

## 🎬 Demo Video

[COMING SOON!]

*Video demonstrates:*
- RAG query with Buffett's investment philosophy
- Time-sensitive query triggering web search
- Hybrid query combining both sources
- Architecture explanation and key learnings

---


**Built with ❤️ and lots of debugging by Cristian Perera**

*"The best investment you can make is in yourself." - Warren Buffett*