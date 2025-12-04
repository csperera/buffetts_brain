# Changelog

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