# App4 Installation & Upgrade Guide

## 🚀 Upgrading from app3 to app4

### What's New in app4?
- **BM25 + Semantic Hybrid Search Engine**
- Better entity recognition (companies, people, investments)
- Reciprocal Rank Fusion (RRF) for intelligent ranking
- Year-aware hybrid filtering
- Production-grade retrieval system

---

## 📋 Installation Steps

### Option A: Fresh Installation

```powershell
# 1. Navigate to project directory
cd C:\Users\chris\buffetts_brain\src

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Ensure environment variables are set
# Make sure .env file contains:
# GROQ_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here

# 4. Run app4
streamlit run app4.py
```

---

### Option B: Upgrade from app3

```powershell
# 1. Navigate to project directory
cd C:\Users\chris\buffetts_brain\src

# 2. Install new dependency (rank-bm25)
pip install rank-bm25

# 3. Install NLTK (if not already installed)
pip install nltk

# 4. Backup your current app3.py
copy app3.py app3_backup.py

# 5. Copy app4.py to your src directory
# (Download from outputs and place in src/)

# 6. Launch app4
streamlit run app4.py
```

---

## ⏱️ First Launch Notes

**Expected on first launch:**
```
🔨 Building BM25 index... (first load only)
✅ BM25 index built: 8,518 documents
```

- First launch: 10-20 seconds (builds BM25 index)
- Subsequent launches: Instant (uses cached data)

---

## 🧪 Testing After Installation

### Test 1: Entity Recognition (Previously Failed)
```
"What did Buffett say about Sanborn Map Company?"
```
**Expected:** Should retrieve 1960 Partnership Letter

### Test 2: Year-Specific Query
```
"How did Buffett perform in 1960 versus the Dow Jones?"
```
**Expected:** Should retrieve 1960 letter with performance data

### Test 3: Philosophical Query
```
"Explain Buffett's circle of competence principle"
```
**Expected:** Should retrieve relevant Berkshire letters

### Test 4: Web Search Routing
```
"What is Berkshire's stock price today?"
```
**Expected:** Should skip RAG and use Tavily web search

---

## 🔧 Configuration Options

Edit these constants in app4.py to tune hybrid search:

```python
# Line 33-35: Hybrid search weights
BM25_WEIGHT = 0.4      # 40% keyword matching (increase for more exact matches)
SEMANTIC_WEIGHT = 0.6  # 60% semantic similarity (increase for more concepts)

# Line 36: Candidate pool size
HYBRID_K = 20  # Number of candidates from each method
```

**Tuning Guide:**
- More entity/company queries failing? → Increase BM25_WEIGHT to 0.5
- More concept queries failing? → Increase SEMANTIC_WEIGHT to 0.7
- Start with defaults (0.4/0.6) and adjust based on testing

---

## 📊 System Requirements

**Minimum:**
- Python 3.9+
- 8GB RAM
- 2GB disk space (for vector database + BM25 index)

**Recommended:**
- Python 3.10+
- 16GB RAM
- SSD storage

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'rank_bm25'"
**Solution:**
```powershell
pip install rank-bm25
```

### Issue: "NLTK data not found"
**Solution:**
```python
# App4 handles this automatically, but if it fails:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Issue: BM25 index build is slow
**Explanation:** Normal on first launch (8,518 documents to index)
**Solution:** Wait for completion. Subsequent launches use cached index.

### Issue: High memory usage
**Explanation:** BM25 index + Vector DB + Embeddings in memory
**Solution:** This is expected for production RAG system (3-4GB RAM usage)

---

## 📁 Project Structure After Upgrade

```
buffetts_brain/
├── src/
│   ├── app3.py              ← Backup (old version)
│   ├── app4.py              ← NEW: Hybrid search engine
│   ├── process_documents.py
│   ├── .env                 ← API keys
│   └── requirements.txt     ← NEW: Updated dependencies
├── knowledge_base/
│   ├── docs/
│   │   ├── Partnership_Letters/
│   │   ├── Berkshire_Letters/
│   │   └── Poor_Charlies_Almanack.pdf
│   └── vector_db/           ← Chroma persistence
└── CHANGELOG.md             ← Version history (pending)
```

---

## 🎯 Rollback Instructions

If you need to revert to app3:

```powershell
# Stop app4 (Ctrl+C)
streamlit run app3.py
```

Your vector database and documents are unchanged - app4 only adds BM25 indexing on top.

---

## ✅ Installation Checklist

- [ ] `pip install rank-bm25` completed
- [ ] `pip install nltk` completed
- [ ] app3.py backed up
- [ ] app4.py deployed to src/ directory
- [ ] .env file contains valid API keys
- [ ] First launch completed (BM25 index built)
- [ ] Test query successful: "What did Buffett say about Sanborn Map Company?"

---

**Questions or Issues?**
See CHANGELOG.md for version history and detailed technical documentation.