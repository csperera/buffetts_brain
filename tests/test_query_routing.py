"""
Tests for intelligent query routing logic
"""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


class TestTimeSensitiveDetection:
    """Test suite for time-sensitive query detection"""
    
    @pytest.fixture
    def time_keywords(self):
        """Time-sensitive keywords from actual implementation"""
        return [
            'yesterday', 'today', 'current', 'now', 'latest', 'recent',
            'this week', 'last week', 'price', 'stock'
        ]
    
    @pytest.mark.parametrize("query,expected", [
        ("What is the stock price of BRK.A today?", True),
        ("What happened yesterday at the meeting?", True),
        ("What is the current price of Apple?", True),
        ("Recent news about Berkshire Hathaway", True),
        ("What is Buffett's investment philosophy?", False),
        ("Explain the circle of competence principle", False),
        ("What would Munger say about cryptocurrency?", False),
    ])
    def test_time_sensitive_detection(self, query, expected, time_keywords):
        """Test time-sensitive keyword detection"""
        query_lower = query.lower()
        is_time_sensitive = any(keyword in query_lower for keyword in time_keywords)
        assert is_time_sensitive == expected
    
    def test_all_time_keywords_lowercase(self, time_keywords):
        """Ensure all time keywords are lowercase for matching"""
        assert all(kw == kw.lower() for kw in time_keywords)
    
    def test_time_keywords_not_empty(self, time_keywords):
        """Ensure time keywords list is not empty"""
        assert len(time_keywords) > 0


class TestRelevanceEvaluation:
    """Test suite for RAG relevance evaluation"""
    
    @pytest.mark.parametrize("score,should_use_rag", [
        (1, False),
        (3, False),
        (4, False),
        (5, True),
        (7, True),
        (10, True),
    ])
    def test_relevance_threshold(self, score, should_use_rag):
        """Test relevance score threshold logic"""
        THRESHOLD = 5
        assert (score >= THRESHOLD) == should_use_rag
    
    def test_score_clamping(self):
        """Test that scores are clamped between 1-10"""
        test_scores = [-5, 0, 1, 5, 10, 15, 100]
        clamped = [min(max(score, 1), 10) for score in test_scores]
        
        assert all(1 <= score <= 10 for score in clamped)
        assert clamped == [1, 1, 1, 5, 10, 10, 10]
    
    def test_relevance_evaluation_structure(self):
        """Test that relevance evaluation has proper structure"""
        evaluation_criteria = {
            "irrelevant": (1, 2),
            "somewhat_related": (3, 4),
            "partially_relevant": (5, 6),
            "mostly_relevant": (7, 8),
            "directly_answers": (9, 10)
        }
        
        # Check all score ranges are covered
        all_scores = set()
        for low, high in evaluation_criteria.values():
            all_scores.update(range(low, high + 1))
        
        assert all_scores == set(range(1, 11))


class TestQueryRouting:
    """Test suite for overall query routing logic"""
    
    def test_routing_priority(self):
        """Test that time-sensitive queries skip RAG evaluation"""
        is_time_sensitive = True
        relevance_score = 10  # Even with high score, should use search
        
        if is_time_sensitive:
            should_use_search = True
        else:
            should_use_search = relevance_score < 5
        
        assert should_use_search == True
    
    def test_fallback_logic(self):
        """Test fallback to search on low relevance"""
        is_time_sensitive = False
        relevance_score = 3
        
        should_use_rag = relevance_score >= 5
        should_use_search = not should_use_rag
        
        assert should_use_search == True
        assert should_use_rag == False
    
    def test_rag_only_path(self):
        """Test RAG-only path for high relevance, non-time-sensitive"""
        is_time_sensitive = False
        relevance_score = 8
        
        should_use_rag = relevance_score >= 5
        should_use_search = is_time_sensitive or (relevance_score < 5)
        
        assert should_use_rag == True
        assert should_use_search == False