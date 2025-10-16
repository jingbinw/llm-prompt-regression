"""
Unit tests for metrics calculation utilities.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.utils.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Test cases for MetricsCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
    
    def test_calculate_lexical_similarity_identical(self):
        """Test lexical similarity for identical texts."""
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "The quick brown fox jumps over the lazy dog."
        
        similarity = self.calculator.calculate_lexical_similarity(text1, text2)
        assert similarity == 1.0
    
    def test_calculate_lexical_similarity_different(self):
        """Test lexical similarity for completely different texts."""
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "Artificial intelligence is transforming the world."
        
        similarity = self.calculator.calculate_lexical_similarity(text1, text2)
        assert similarity < 1.0
        assert similarity >= 0.0
    
    def test_calculate_token_similarity(self):
        """Test token-level similarity calculation."""
        text1 = "Hello world"
        text2 = "Hello world"
        
        similarity = self.calculator.calculate_token_similarity(text1, text2)
        assert similarity == 1.0
    
    def test_calculate_length_metrics(self):
        """Test length metrics calculation."""
        text1 = "Short text"
        text2 = "This is a much longer text with more words"
        
        metrics = self.calculator.calculate_length_metrics(text1, text2)
        
        assert "char_length_diff" in metrics
        assert "char_length_ratio" in metrics
        assert "token_length_diff" in metrics
        assert "token_length_ratio" in metrics
        assert "word_count_diff" in metrics
        
        assert metrics["char_length_diff"] > 0
        assert 0 <= metrics["char_length_ratio"] <= 1
    
    def test_calculate_structural_similarity(self):
        """Test structural similarity calculation."""
        text1 = "First sentence. Second sentence!"
        text2 = "Only one sentence."
        
        metrics = self.calculator.calculate_structural_similarity(text1, text2)
        
        assert "sentence_count_diff" in metrics
        assert "paragraph_count_diff" in metrics
        assert "question_count_diff" in metrics
        assert "exclamation_count_diff" in metrics
        assert "has_same_ending" in metrics
        
        assert metrics["sentence_count_diff"] == 1  # 2 - 1 = 1
        assert metrics["exclamation_count_diff"] == 1  # 1 - 0 = 1
    
    def test_calculate_content_overlap(self):
        """Test content overlap calculation."""
        text1 = "The cat sat on the mat"
        text2 = "The dog sat on the mat"
        
        metrics = self.calculator.calculate_content_overlap(text1, text2)
        
        assert "word_overlap_ratio" in metrics
        assert "unique_words_text1" in metrics
        assert "unique_words_text2" in metrics
        assert "key_phrase_overlap" in metrics
        assert "key_phrase_ratio" in metrics
        
        assert 0 <= metrics["word_overlap_ratio"] <= 1
        assert metrics["unique_words_text1"] >= 0
        assert metrics["unique_words_text2"] >= 0
    
    @pytest.mark.asyncio
    async def test_calculate_semantic_similarity(self):
        """Test semantic similarity calculation."""
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "A fast brown fox leaps over a sleepy dog."
        
        similarity = await self.calculator.calculate_semantic_similarity(text1, text2)
        
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be semantically similar
    
    @pytest.mark.asyncio
    async def test_calculate_comprehensive_similarity(self):
        """Test comprehensive similarity calculation."""
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "ML is part of AI technology."
        
        metrics = await self.calculator.calculate_comprehensive_similarity(text1, text2)
        
        assert "semantic_similarity" in metrics
        assert "lexical_similarity" in metrics
        assert "token_similarity" in metrics
        assert "length_metrics" in metrics
        assert "structural_similarity" in metrics
        assert "content_overlap" in metrics
        
        assert 0 <= metrics["semantic_similarity"] <= 1
        assert 0 <= metrics["lexical_similarity"] <= 1
        assert 0 <= metrics["token_similarity"] <= 1
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  This   is   a   test!!!  "
        cleaned = self.calculator._clean_text(dirty_text)
        
        assert cleaned == "This is a test!!!"
    
    def test_count_sentences(self):
        """Test sentence counting."""
        text = "First sentence. Second sentence! Third sentence?"
        count = self.calculator._count_sentences(text)
        assert count == 3
    
    def test_count_paragraphs(self):
        """Test paragraph counting."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        count = self.calculator._count_paragraphs(text)
        assert count == 3
    
    def test_count_questions(self):
        """Test question counting."""
        text = "What is this? How does it work? This is not a question."
        count = self.calculator._count_questions(text)
        assert count == 2
    
    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        text = "The OpenAI GPT model is amazing. API calls are simple."
        phrases = self.calculator._extract_key_phrases(text)
        
        assert "OpenAI" in phrases
        assert "GPT" in phrases
        assert "API" in phrases
