"""
Pytest configuration and shared fixtures
"""
import pytest
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


@pytest.fixture
def mock_groq_api_key():
    """Mock Groq API key for testing"""
    return os.getenv("GROQ_API_KEY", "test_groq_key")


@pytest.fixture
def mock_tavily_api_key():
    """Mock Tavily API key for testing"""
    return os.getenv("TAVILY_API_KEY", "test_tavily_key")


@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return {
        "rag_query": "What is Buffett's circle of competence principle?",
        "time_sensitive": "What is the stock price of BRK.A today?",
        "hybrid": "Should I invest in Apple based on Buffett's criteria?",
        "edge_case": "What's the weather in Omaha?"
    }


@pytest.fixture
def sample_rag_context():
    """Sample RAG context for testing relevance evaluation"""
    return """
    Warren Buffett emphasizes the importance of staying within one's circle of competence.
    He believes investors should only invest in businesses they truly understand.
    This principle has been a cornerstone of Berkshire Hathaway's investment strategy.
    """


@pytest.fixture
def embedding_model():
    """Initialize embedding model for testing"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@pytest.fixture
def mock_llm(mock_groq_api_key):
    """Mock LLM for testing (requires valid API key to run)"""
    try:
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=mock_groq_api_key,
            temperature=0.0,
            max_tokens=100
        )
    except Exception:
        pytest.skip("Groq API key not available")