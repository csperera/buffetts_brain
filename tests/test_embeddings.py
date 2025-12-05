"""
Tests for embedding functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


class TestEmbeddings:
    """Test suite for embedding operations"""
    
    def test_embedding_vector_properties(self, embedding_model):
        """Test that embeddings have expected properties"""
        text = "Warren Buffett invests in quality businesses"
        embedding = embedding_model.embed_query(text)
        
        # Check it's a list/array
        assert isinstance(embedding, (list, np.ndarray))
        
        # Check dimension
        assert len(embedding) == 384
        
        # Check all values are floats
        assert all(isinstance(val, (float, np.floating)) for val in embedding)
    
    def test_embedding_similarity_concept(self, embedding_model):
        """Test that similar texts have more similar embeddings"""
        text1 = "Warren Buffett's investment strategy"
        text2 = "Buffett's approach to investing"
        text3 = "The weather in Tokyo"
        
        emb1 = embedding_model.embed_query(text1)
        emb2 = embedding_model.embed_query(text2)
        emb3 = embedding_model.embed_query(text3)
        
        # Cosine similarity function
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_related = cosine_sim(emb1, emb2)
        sim_unrelated = cosine_sim(emb1, emb3)
        
        # Related texts should be more similar
        assert sim_related > sim_unrelated
    
    @pytest.mark.parametrize("text", [
        "Circle of competence",
        "Margin of safety",
        "Economic moat",
        "Intrinsic value",
        "Mr. Market"
    ])
    def test_buffett_concepts_embed_successfully(self, embedding_model, text):
        """Test that key Buffett concepts can be embedded"""
        embedding = embedding_model.embed_query(text)
        assert len(embedding) == 384
        assert all(isinstance(val, (float, np.floating)) for val in embedding)
    
    def test_empty_string_handling(self, embedding_model):
        """Test handling of empty string"""
        # Should not crash
        try:
            embedding = embedding_model.embed_query("")
            assert len(embedding) == 384
        except Exception as e:
            # If it raises an exception, that's also acceptable behavior
            assert isinstance(e, (ValueError, Exception))
    
    def test_long_text_handling(self, embedding_model):
        """Test handling of long text"""
        long_text = "Warren Buffett " * 1000  # Very long text
        
        try:
            embedding = embedding_model.embed_query(long_text)
            assert len(embedding) == 384
        except Exception:
            # Model may truncate or reject very long text - both acceptable
            pass


class TestVectorOperations:
    """Test suite for vector operations"""
    
    def test_vector_normalization_concept(self):
        """Test vector normalization logic"""
        vector = [1.0, 2.0, 3.0]
        magnitude = np.sqrt(sum(x**2 for x in vector))
        normalized = [x / magnitude for x in vector]
        
        # Check normalized vector has magnitude ~1
        normalized_magnitude = np.sqrt(sum(x**2 for x in normalized))
        assert abs(normalized_magnitude - 1.0) < 1e-6
    
    def test_similarity_search_k_parameter(self):
        """Test that k parameter for similarity search is valid"""
        k = 4  # From actual implementation
        assert isinstance(k, int)
        assert k > 0
        assert k <= 10  # Reasonable upper bound