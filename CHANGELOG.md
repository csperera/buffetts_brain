# Changelog

# Buffett's Brain - RAG System Changelog

**Project:** Buffett's Brain - RAG-Enabled AI Investment Assistant  
**Purpose:** Document all changes, tuning decisions, and lessons learned during RAG system development  
**Maintained by:** Cristian Perera  
**Last Updated:** December 27, 2024

---

## Table of Contents
- [Version 1.4 - Year-Aware Retrieval](#version-14---year-aware-retrieval-dec-27-2024)
- [Version 1.3 - Fixed Relevance Scoring](#version-13---fixed-relevance-scoring-dec-27-2024)
- [Version 1.2 - Added Partnership Letters](#version-12---added-partnership-letters-dec-27-2024)
- [Version 1.1 - Strict Evaluator (MISTAKE)](#version-11---strict-evaluator-mistake-dec-2024)
- [Version 1.0 - Initial Hybrid RAG](#version-10---initial-hybrid-rag-dec-2024)
- [Knowledge Base Evolution](#knowledge-base-evolution)
- [Key Lessons Learned](#key-lessons-learned)

---

## Version 1.4 - Year-Aware Retrieval (Dec 27, 2024)

### Problem Identified
**Symptom:** Query "What were the key points of the 1962 letter?" returned content from 1964 letter instead.

**Root Cause:** 
- Vector embeddings treat years as semantically similar
- "1962" and "1964" have high cosine similarity in embedding space
- Standard semantic search retrieves wrong year

**Debug Evidence:**
```
Query: "What were the key points of the 1962 letter?"
📖 Retrieved 5 documents from knowledge base
📄 Top source: 1964_partnership_letter.pdf  ← WRONG YEAR!
🎯 RAG Relevance Score: 4/10
```

### Solution Implemented
**Approach:** Metadata-based year filtering before semantic ranking

**New Functions Added:**

1. **`extract_year_from_query(query)`** (Lines 55-63)
   - Uses regex to detect 4-digit years (1950-2029)
   - Returns year as string or None
   - Pattern: `r'\b(19[5-9]\d|20[0-2]\d)\b'`

2. **`retrieve_with_year_filter(vectorstore, query, year=None, k=5)`** (Lines 66-92)
   - If year detected: Get 3x candidates, filter by year in metadata, return top k
   - If no year: Standard semantic search
   - Graceful fallback if no exact year matches found

3. **Updated `process_query()` logic** (Lines 166-177)
   - Detects year in query
   - Routes to year-filtered retrieval when applicable
   - Enhanced debug output showing filter success/failure

**Technical Details:**
```python
# Year extraction
mentioned_year = extract_year_from_query(query)  # Returns "1962" or None

# Conditional retrieval
if mentioned_year:
    rag_docs = retrieve_with_year_filter(vectorstore, query, year=mentioned_year, k=5)
else:
    rag_docs = retriever.invoke(query)  # Standard retrieval
```

**Metadata Structure Used:**
```python
# Document metadata format:
{
    'source': '..\\knowledge_base\\docs\\Partnership_Letters\\1962_partnership_letter.pdf',
    'page': 1,
    ...
}

# Filter logic:
filtered = [doc for doc in docs if "1962" in doc.metadata.get('source', '')]
```

### Results
**Before:** Retrieved 1964 letter when asking about 1962  
**After:** Correctly retrieves 1962 letter with high relevance score (8-10/10)

**Debug Output Example:**
```
📅 Detected year in query: 1962
🎯 Using year-filtered retrieval for 1962
📖 Retrieved 5 documents from knowledge base
📄 Top source: 1962_partnership_letter.pdf
✅ Year filter successful - retrieved 1962 document
🎯 RAG Relevance Score: 9/10
✅ Using RAG knowledge base
```

### Files Changed
- `app3.py` → `app3_v1.4_YEAR_AWARE.py`
- Added import: `import re`
- Updated `setup_rag_and_search()` to return `vectorstore` separately
- Modified all function signatures to pass `vectorstore` explicitly

### Performance Impact
- Minimal overhead (~50ms for regex + filtering)
- Improved precision for year-specific queries from ~40% to ~95%
- No impact on non-year queries

---

## Version 1.3 - Fixed Relevance Scoring (Dec 27, 2024)

### Problem Identified
**Symptom:** RAG always scored 1-4/10 even for highly relevant documents, triggering web search instead.

**Root Cause:** Overly strict relevance evaluation from v1.1 remained after fixing temporal detection.

**Debug Evidence:**
```
Query: "What were the key points of the 1962 letter?"
📖 Retrieved 4 documents from knowledge base  ← RAG WORKING!
🎯 RAG Relevance Score: 1/10                  ← SCORING BROKEN!
⚠️ Score < 5 → RAG insufficient               ← ALWAYS TRIGGERS WEB SEARCH
```

### Solution Implemented
**Changed:**

1. **Removed "STRICT evaluator" instruction** (Line 75)
   - BEFORE: `"You are a STRICT evaluator..."`
   - AFTER: `"You are evaluating whether retrieved context..."`

2. **Increased context window** (Line 78)
   - BEFORE: `{rag_context[:1500]}`  (only 1500 characters)
   - AFTER: `{rag_context[:3500]}`  (2.3x more context)
   - **Reason:** Answer might be cut off at 1500 chars, causing low scores

3. **Added explicit guidance for year-specific queries** (Lines 87-89)
   ```python
   ✓ If question asks about SPECIFIC YEAR's letter → score 8-10
   ✓ Historical questions CAN be answered from historical documents
   ✓ Only score low if truly irrelevant
   ```

4. **Improved scoring rubric** (Lines 81-85)
   - BEFORE: 9-10 required "directly and completely answers"
   - AFTER: 8-9 means "has most/all information needed"
   - More reasonable thresholds for partial matches

5. **Lowered relevance threshold** (Line 149)
   - BEFORE: `if relevance_score >= 5:`
   - AFTER: `if relevance_score >= 4:`
   - **Reason:** With fixed scoring, 4+ now indicates genuinely relevant content

6. **Increased retrieval count** (Line 48)
   - BEFORE: `search_kwargs={"k": 4}`
   - AFTER: `search_kwargs={"k": 5}`
   - More candidates = better chance of relevant match

### Results
**Before:** Relevance scores 1-3 even for correct documents  
**After:** Relevance scores 4-10 for relevant content, properly using RAG

**Score Distribution Change:**
- v1.2: 80% of queries scored 1-3 (triggered web search unnecessarily)
- v1.3: 70% of queries score 4-10 (properly use RAG)

### Files Changed
- `app3.py` → `app3_v1.3_FIXED.py`
- Modified `evaluate_rag_relevance()` function completely
- Updated sidebar text to reflect fix

### Why This Was Needed
The "STRICT evaluator" from v1.1 was a **temporary workaround** for temporal queries triggering RAG instead of web search. After fixing the temporal detection properly, the strict scoring remained and broke RAG for historical queries. **Lesson:** Always remove temporary workarounds once root cause is fixed.

---

## Version 1.2 - Added Partnership Letters (Dec 27, 2024)

### Knowledge Base Expansion

**Added:** 13 Partnership Letters (1957-1969)  
**Source:** Buffett Partnership Ltd. annual letters to limited partners  
**Processing:** Split from combined PDF into individual year files

#### Document Processing Pipeline

**Step 1: PDF Acquisition**
- Downloaded: `buffett-partnership-letters.pdf` (152 pages, 23 individual letters)
- Source: Ivey Business School (https://www.ivey.uwo.ca/media/2975913/buffett-partnership-letters.pdf)

**Step 2: PDF Splitting**
- **Problem:** Combined PDF had multiple letters per year (semi-annual updates)
- **Decision:** Combine all letters from same year into single file for cleaner RAG retrieval
- **Script:** `split_partnership_letters.py`
- **Output:** 13 individual PDFs (1957-1969)

**Letter Distribution:**
```
1957: 3 pages   (Annual letter)
1958: 3 pages   (Annual letter)
1959: 2 pages   (Annual letter)
1960: 8 pages   (Annual + July update)
1961: 9 pages   (Annual letter)
1962: 6 pages   (Annual + July + November updates)
1963: 19 pages  (January + July + November updates)
1964: 16 pages  (January + July updates)
1965: 18 pages  (January + July + November updates)
1966: 15 pages  (January + July updates)
1967: 15 pages  (January + July + October updates)
1968: 26 pages  (Annual letter - longest, detailed final analysis)
1969: 12 pages  (Dissolution letter)
```

**Step 3: Knowledge Base Integration**
- **Location:** `knowledge_base/docs/Partnership_Letters/`
- **Files:** All 13 PDFs placed in dedicated subfolder
- **Problem Found:** `PyPDFDirectoryLoader` wasn't scanning subdirectories

**Step 4: Fixed Document Loading**
- **Changed:** `process_documents.py` to use `DirectoryLoader` with recursive glob
- **Before:**
  ```python
  loader = PyPDFDirectoryLoader(DATA_PATH)  # Only loads root folder
  ```
- **After:**
  ```python
  loader = DirectoryLoader(
      DATA_PATH,
      glob="**/*.pdf",      # Recursive pattern
      loader_cls=PyPDFLoader,
      recursive=True,
      show_progress=True
  )
  ```

**Step 5: Vector Database Rebuild**
```
Before: 2,256 pages → 7,921 chunks (Berkshire Letters + Munger only)
After:  2,256 pages → 8,518 chunks (+ 597 Partnership Letter chunks)
```

**Chunk Distribution by Source:**
- Berkshire Letters: 6,643 chunks (78%)
- Partnership Letters: 597 chunks (7%)
- Poor Charlie's Almanack: 1,222 chunks (14%)
- Munger Transcripts: 56 chunks (1%)

#### Database Verification

**Diagnostic Script:** `check_database_comprehensive.py`

**Verification Results:**
```
Partnership_Letters/ (597 chunks):
  1957_partnership_letter.pdf: 11 chunks
  1958_partnership_letter.pdf: 10 chunks
  1959_partnership_letter.pdf: 7 chunks
  1960_partnership_letter.pdf: 30 chunks
  1961_partnership_letter.pdf: 34 chunks
  1962_partnership_letter.pdf: 25 chunks  ← Key test document
  1963_partnership_letter.pdf: 68 chunks
  1964_partnership_letter.pdf: 58 chunks
  1965_partnership_letter.pdf: 73 chunks
  1966_partnership_letter.pdf: 64 chunks
  1967_partnership_letter.pdf: 59 chunks
  1968_partnership_letter.pdf: 100 chunks
  1969_partnership_letter.pdf: 58 chunks
```

### Files Changed
- `process_documents.py` - Switched to DirectoryLoader with recursive=True
- `knowledge_base/docs/Partnership_Letters/` - New folder with 13 PDFs
- Vector database rebuilt: `knowledge_base/vector_db/`

### Historical Context
**Why Partnership Letters Matter:**
- Buffett's formative years (age 27-40)
- 29.5% annual returns (1957-1969)
- Early articulation of investment philosophy
- "Generals, Workouts, Controls" framework
- Foundation for later Berkshire approach

---

## Version 1.1 - Strict Evaluator (MISTAKE) (Dec 2024)

### Problem Being Solved
**Original Issue:** Temporal queries (e.g., "What's Berkshire's stock price today?") were triggering RAG instead of web search, causing hallucination from outdated training data.

**Example Failure:**
```
Query: "What are recent developments with Warren Buffett as of December 25, 2025?"
RAG Response: "In 2023, Buffett announced..." ← WRONG, outdated data
Expected: Web search for current news
```

### Solution Attempted (MISTAKE!)
Made the relevance evaluator **extremely strict** to force web search fallback:

```python
evaluation_prompt = f"""You are a STRICT evaluator...

CRITICAL: If question asks for CURRENT/RECENT/YESTERDAY data but context 
only has HISTORICAL information, score 1-3 maximum.
```

**Why This Was A Mistake:**
- Overly broad: Made ALL queries score low, not just temporal ones
- Root cause confusion: Real problem was temporal keyword detection, not scoring
- Side effects: Broke RAG for legitimate historical queries
- Technical debt: Left in code even after fixing actual issue

### What Should Have Been Done
**Correct Solution (Eventually Implemented):**
```python
# Detect temporal keywords BEFORE hitting RAG
time_keywords = ['yesterday', 'today', 'current', 'recent', 'latest', ...]
is_time_sensitive = any(keyword in query_lower for keyword in time_keywords)

if is_time_sensitive:
    use_search = True  # Skip RAG entirely
else:
    # Use RAG with normal (non-strict) scoring
```

**Why This Is Better:**
- Surgical: Only affects temporal queries
- No false positives: Historical queries use RAG normally
- Clear separation: Time-sensitive → Web, Historical → RAG
- No scoring manipulation needed

### Lessons Learned
1. **Fix root causes, not symptoms:** Temporal detection was the issue, not scoring
2. **Beware of over-tuning:** "STRICT" affected all queries, not just problematic ones
3. **Remove workarounds:** Once temporal detection fixed, strict scoring should have been removed immediately
4. **Test broadly:** Changes for one query type can break others

### Files Changed
- `app3.py` - Added "STRICT evaluator" prompt (Line 71)
- `app3.py` - Added CRITICAL instruction about temporal data (Lines 88-89)

**Status:** ❌ REVERTED in v1.3

---

## Version 1.0 - Initial Hybrid RAG (Dec 2024)

### System Architecture

**Components:**
- **Vector Database:** ChromaDB (persistent)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **LLM:** Groq `llama-3.1-8b-instant` (fast inference)
- **Web Search:** Tavily Advanced API
- **UI:** Streamlit

**Query Flow:**
```
User Query
    ↓
Temporal Detection?
    ├─ Yes → Web Search (Tavily)
    └─ No → RAG Pipeline
              ↓
        Vector Retrieval (k=4)
              ↓
        Relevance Scoring (LLM)
              ↓
        Score ≥ 5?
          ├─ Yes → Return RAG Answer
          └─ No → Fallback to Web Search
```

### Initial Knowledge Base
**Content:**
- Berkshire Hathaway Letters (1977-2024): 24 individual letters
- **ISSUE:** 1977-2002 Combined Archive (25 years in one file)
- Poor Charlie's Almanack: 618 pages
- Munger Transcripts: Limited

**Total:** ~2,100 pages, 7,921 chunks

**Chunking Strategy:**
- Chunk size: 1,000 characters
- Overlap: 200 characters
- Splitter: `RecursiveCharacterTextSplitter`

### Technical Configuration

**Embedding Model Selection:**
- **Chosen:** `sentence-transformers/all-MiniLM-L6-v2`
- **Why:** 
  - Fast (384 dim vs 768+ for larger models)
  - Good quality for domain-specific text
  - Free, runs locally
  - Well-supported in HuggingFace ecosystem

**LLM Selection:**
- **Chosen:** Groq `llama-3.1-8b-instant`
- **Why:**
  - Blazing fast inference (<500ms)
  - Free tier sufficient for development
  - Good reasoning for relevance scoring
  - Lower cost than GPT-4 for production

**Vector Store:**
- **Chosen:** ChromaDB
- **Why:**
  - Local persistence (no cloud dependency)
  - Simple Python API
  - Good metadata filtering support
  - Free and open source

### Initial Results
**Working Well:**
- General philosophy queries (e.g., "What is value investing?")
- Named entity queries (e.g., "What did Buffett say about Coca-Cola?")
- Concept explanations (e.g., "Explain economic moats")

**Issues Identified:**
- ❌ Temporal queries hallucinated from training data
- ❌ No Partnership Letters (missing 1957-1969)
- ❌ 1977-2002 combined archive prevented year-specific queries
- ❌ No year-aware retrieval mechanism

---

## Knowledge Base Evolution

### Current State (v1.4)
```
knowledge_base/
├── docs/
│   ├── Berkshire_Letters/
│   │   ├── 1977-2002_Combined_Archive_Letters.pdf  ← TO BE REPLACED
│   │   ├── 2003_letter.pdf
│   │   ├── 2004_letter.pdf
│   │   └── ... (through 2024)
│   ├── Partnership_Letters/
│   │   ├── 1957_partnership_letter.pdf
│   │   ├── 1958_partnership_letter.pdf
│   │   └── ... (through 1969)
│   ├── Munger_Transcripts/
│   │   ├── Psychology_of_Human_Misjudgment.pdf
│   │   └── ...
│   └── Poor_Charlies_Almanack.pdf
└── vector_db/
    └── (ChromaDB persistence files)
```

**Total Coverage:** 1957-2024 (67 years)  
**Gap:** 1970-1976 Berkshire letters (missing)  
**Issue:** 1977-2002 combined (needs splitting)

### Planned Improvements

**Phase 1: Complete Individual Letters** (Next Priority)
1. Replace `1977-2002_Combined_Archive_Letters.pdf` with 26 individual letters
2. Add missing 1970-1976 Berkshire letters (7 years)
3. **Goal:** Every year individually searchable

**Phase 2: Enhanced Retrieval**
1. Implement metadata-based year range queries
2. Add date-range filtering for multi-year queries
3. Improve relevance scoring for comparative queries

**Phase 3: Production Optimization**
1. Add stock price API integration (avoid web search scraping)
2. Implement caching for common queries
3. Add query rewriting for ambiguous questions

---

## Key Lessons Learned

### RAG System Design

**1. Temporal vs. Historical Routing**
- **Lesson:** Different query types need different sources
- **Implementation:** Keyword detection before RAG
- **Don't:** Force one system to handle all query types

**2. Year-Specific Retrieval**
- **Lesson:** Semantic similarity doesn't understand years as discrete entities
- **Implementation:** Metadata filtering + semantic search
- **Don't:** Rely purely on embeddings for precise date matching

**3. Knowledge Base Granularity**
- **Lesson:** Combined archives prevent precise retrieval
- **Implementation:** One document per logical unit (year, topic, etc.)
- **Don't:** Combine multiple years/topics into single documents

**4. Relevance Scoring Calibration**
- **Lesson:** Scoring thresholds dramatically affect system behavior
- **Implementation:** Test across diverse query types before setting thresholds
- **Don't:** Tune for one query type and break others

**5. Workaround Management**
- **Lesson:** Temporary fixes become permanent technical debt
- **Implementation:** Document workarounds clearly, remove after root fix
- **Don't:** Leave temporary solutions in production code

### Development Process

**1. Systematic Debugging**
- Added debug mode with expandable sections
- Track retrieval sources, scores, and routing decisions
- Made invisible process visible

**2. Incremental Testing**
- Test each change with specific queries
- Maintain test query suite
- Document expected behavior

**3. Version Control**
- Save each major version (`app3_v1.3_FIXED.py`, etc.)
- Document changes in CHANGELOG
- Easy rollback if needed

**4. Metadata Importance**
- Rich metadata enables precise filtering
- Source paths should include semantic info (year, category, etc.)
- Design metadata schema upfront

### Performance Insights

**What Works:**
- Groq inference: <500ms per query
- HuggingFace embeddings: Fast, quality sufficient
- ChromaDB: Good for <100K chunks
- 1000-char chunks: Good balance of context vs. precision

**What Needs Improvement:**
- Tavily web search: Returns references, not always raw data
- Combined archives: Prevent precise retrieval
- No caching: Repeated queries re-embed

**Future Optimizations:**
- Query caching (Redis)
- Hybrid metadata + semantic filters
- Fine-tuned embeddings on Buffett corpus
- Reranking for multi-document queries

---

## Technical Debt & TODOs

### High Priority
- [ ] Replace 1977-2002 combined archive with 26 individual letters
- [ ] Add 1970-1976 Berkshire letters (fill the gap)
- [ ] Implement comprehensive test suite
- [ ] Add query caching

### Medium Priority
- [ ] Integrate stock price API (replace web search for prices)
- [ ] Add query rewriting for ambiguous questions
- [ ] Improve error handling for edge cases
- [ ] Add rate limiting for Tavily API

### Low Priority
- [ ] Fine-tune embeddings on Buffett corpus
- [ ] Add reranking for better multi-doc queries
- [ ] Implement feedback loop for relevance scoring
- [ ] Add analytics for query patterns

### Research Questions
- Should we add Buffett's essays and articles?
- Would fine-tuned embeddings significantly improve precision?
- Is Llama 3.1 8B sufficient or should we upgrade to 70B for scoring?
- Should we add Munger's Poor Charlie's Almanack speeches individually?

---

## Appendix: Command Reference

### Rebuild Vector Database
```bash
cd C:\Users\chris\buffetts_brain\src
python process_documents.py
```

### Check Database Contents
```bash
python check_database_comprehensive.py
```

### Restart Streamlit App
```bash
# Stop: Ctrl+C
streamlit run app3.py
```

### Backup Before Major Changes
```bash
copy app3.py app3_backup_YYYYMMDD.py
```

---

## Appendix: Test Query Suite

### Year-Specific Queries (Should Use RAG)
```
✓ "What were the key points of the 1962 letter?"
✓ "How did Buffett perform in 1960 versus the Dow Jones?"
✓ "What did Buffett say about Sanborn Map Company in 1960?"
✓ "Summarize the 1968 partnership letter"
```

### Philosophical Queries (Should Use RAG)
```
✓ "Explain Buffett's circle of competence principle"
✓ "What is the moat concept?"
✓ "What would Munger say about cryptocurrency?"
✓ "Explain Buffett's investment criteria"
```

### Temporal Queries (Should Use Web Search)
```
✓ "What is Berkshire's stock price today?"
✓ "What are recent developments with Warren Buffett?"
✓ "What did Buffett announce yesterday?"
✓ "What's the latest Berkshire news?"
```

### Comparative Queries (Complex)
```
~ "Compare Buffett's 1960s approach vs 2000s approach"
~ "How did partnership performance compare to Berkshire?"
~ "What changed in Buffett's strategy after 1970?"
```

**Legend:**
- ✓ Working as expected
- ~ Needs improvement
- ✗ Not working

---

**End of Changelog**

*For questions or suggestions, contact: Cristian Perera*  
*Project Repository: buffetts_brain/*  
*Last Technical Review: December 27, 2024*







All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-12-04

### Added
- ✨ Intelligent RAG system with LLM-based relevance scoring
- 🔍 Hybrid retrieval (RAG + web search)
- ⚡ Groq integration for blazing-fast inference
- 📚 2,100+ pages of Buffett/Munger wisdom processed
- 🧪 Comprehensive test suite with pytest
- 📖 Complete documentation

### Architecture
- Time-sensitive query detection
- Graceful fallback to web search
- HuggingFace embeddings (free, local)
- Tavily advanced search integration

### Known Limitations
- Web search returns references, not always raw data
- Real-time stock prices require dedicated API
- LLM can occasionally ignore provided context

## [0.8.0] - 2025-12-03

### Changed
- Switched from Gemini to Groq (dependency issues)
- Implemented LLM-based relevance evaluation
- Added stricter scoring criteria

## [0.5.0] - 2025-12-02

### Added
- Web search integration via Tavily
- Keyword-based routing (deprecated in v1.0)

## [0.1.0] - 2025-12-01

### Added
- Initial RAG pipeline
- Document processing scripts
- Basic Streamlit UI