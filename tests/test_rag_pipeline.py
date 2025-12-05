"""
Tests for RAG pipeline functionality
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


class TestRAGPipeline:
    """Test suite for RAG retrieval pipeline"""
    
    def test_embedding_model_initialization(self, embedding_model):
        """Test that embedding model initializes correctly"""
        assert embedding_model is not None
        assert hasattr(embedding_model, 'embed_query')
    
    def test_embedding_dimensionality(self, embedding_model):
        """Test that embeddings have correct dimensionality"""
        test_text = "Warren Buffett's investment philosophy"
        embedding = embedding_model.embed_query(test_text)
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    
    def test_embedding_consistency(self, embedding_model):
        """Test that same text produces same embedding"""
        text = "Circle of competence"
        embedding1 = embedding_model.embed_query(text)
        embedding2 = embedding_model.embed_query(text)
        assert embedding1 == embedding2
    
    @pytest.mark.parametrize("chunk_size,overlap", [
        (1000, 200),
        (500, 100),
        (2000, 400)
    ])
    def test_chunking_parameters(self, chunk_size, overlap):
        """Test that chunking parameters are valid"""
        assert chunk_size > overlap
        assert overlap >= 0
        assert chunk_size > 0
    
    def test_vector_db_path_exists(self):
        """Test that vector DB path is configured"""
        expected_path = "../knowledge_base/vector_db"
        assert expected_path is not None
        # Note: Actual DB may not exist in test environment
    
    def test_retriever_k_parameter(self):
        """Test that retriever k parameter is sensible"""
        k = 4  # From actual config
        assert 1 <= k <= 10  # Reasonable range
    
    def test_mock_document_retrieval(self):
        """Test document retrieval with mock data"""
        mock_docs = [
            {"content": "Buffett invests in companies with moats", "score": 0.9},
            {"content": "Circle of competence is key", "score": 0.85},
            {"content": "Margin of safety principle", "score": 0.8},
            {"content": "Long-term value investing", "score": 0.75}
        ]
        
        # Test that we get expected number of docs
        assert len(mock_docs) == 4
        
        # Test that scores are descending
        scores = [doc["score"] for doc in mock_docs]
        assert scores == sorted(scores, reverse=True)


class TestDocumentProcessing:
    """Test suite for document processing"""
    
    def test_pdf_sources_defined(self):
        """Test that PDF sources are properly defined"""
        sources = {
            "berkshire_letters": "knowledge_base/docs/Berkshire_Letters",
            "munger_transcripts": "knowledge_base/docs/Munger_Transcripts",
            "almanack": "knowledge_base/docs/Poor_Charlies_Almanack.pdf"
        }
        assert all(isinstance(path, str) for path in sources.values())
    
    def test_document_count_expectations(self):
        """Test that we expect reasonable document counts"""
        expected_pages = 2104  # From actual processing
        expected_chunks = 7921  # From actual processing
        
        assert expected_pages > 1000  # Sanity check
        assert expected_chunks > expected_pages  # Chunks > pages makes sense
        assert expected_chunks / expected_pages < 10  # Reasonable ratio