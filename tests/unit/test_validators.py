"""
Unit tests for response validation utilities.
"""

import pytest
from unittest.mock import Mock, patch

from src.utils.validators import ResponseValidator


class TestResponseValidator:
    """Test cases for ResponseValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ResponseValidator()
    
    @pytest.mark.asyncio
    async def test_validate_coherence_good_text(self):
        """Test coherence validation for well-structured text."""
        text = "This is a well-structured sentence. It has proper capitalization. The flow is logical and coherent."
        
        score = await self.validator.validate_coherence(text)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be reasonably coherent
    
    @pytest.mark.asyncio
    async def test_validate_coherence_poor_text(self):
        """Test coherence validation for poorly structured text."""
        text = "this is bad text no capitals repetitive repetitive repetitive words words words"
        
        score = await self.validator.validate_coherence(text)
        
        assert 0 <= score <= 1
        assert score < 0.8  # Should be less coherent
    
    @pytest.mark.asyncio
    async def test_validate_relevance_good_match(self):
        """Test relevance validation for relevant response."""
        prompt = "Explain machine learning"
        response = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        
        score = await self.validator.validate_relevance(prompt, response)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be relevant
    
    @pytest.mark.asyncio
    async def test_validate_relevance_poor_match(self):
        """Test relevance validation for irrelevant response."""
        prompt = "Explain machine learning"
        response = "The weather is nice today. I like pizza. Cats are cute animals."
        
        score = await self.validator.validate_relevance(prompt, response)
        
        assert 0 <= score <= 1
        assert score < 0.5  # Should be less relevant
    
    @pytest.mark.asyncio
    async def test_validate_completeness_complete(self):
        """Test completeness validation for complete response."""
        prompt = "Write a short story"
        response = "Once upon a time, there was a brave knight who saved the kingdom."
        
        score = await self.validator.validate_completeness(prompt, response)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be reasonably complete
    
    @pytest.mark.asyncio
    async def test_validate_completeness_incomplete(self):
        """Test completeness validation for incomplete response."""
        prompt = "Write a detailed explanation of photosynthesis"
        response = "Photosynthesis is a process"
        
        score = await self.validator.validate_completeness(prompt, response)
        
        assert 0 <= score <= 1
        assert score < 0.8  # Should be less complete
    
    @pytest.mark.asyncio
    async def test_validate_safety_safe_text(self):
        """Test safety validation for safe text."""
        text = "This is a normal, safe text about technology and innovation."
        
        score, warnings = await self.validator.validate_safety(text)
        
        assert 0 <= score <= 1
        assert score > 0.8  # Should be safe
        assert len(warnings) == 0
    
    @pytest.mark.asyncio
    async def test_validate_safety_repetitive_text(self):
        """Test safety validation for repetitive text."""
        text = "spam spam spam spam spam spam spam spam spam spam spam spam spam"
        
        score, warnings = await self.validator.validate_safety(text)
        
        assert 0 <= score <= 1
        assert score < 1.0  # Should be penalized for repetition
        assert len(warnings) > 0
        assert "repetition" in warnings[0].lower()
    
    @pytest.mark.asyncio
    async def test_validate_quality(self):
        """Test comprehensive quality validation."""
        text = "This is a well-written response. It has proper grammar and structure. The content is relevant and coherent."
        
        quality = await self.validator.validate_quality(text)
        
        assert "coherence" in quality
        assert "safety_score" in quality
        assert "safety_warnings" in quality
        assert "readability" in quality
        assert "grammar_score" in quality
        assert "structure_score" in quality
        
        assert 0 <= quality["coherence"] <= 1
        assert 0 <= quality["safety_score"] <= 1
        assert 0 <= quality["readability"] <= 1
        assert 0 <= quality["grammar_score"] <= 1
        assert 0 <= quality["structure_score"] <= 1
    
    def test_split_sentences(self):
        """Test sentence splitting functionality."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.validator._split_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
    
    def test_extract_key_terms(self):
        """Test key term extraction."""
        text = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        terms = self.validator._extract_key_terms(text)
        
        assert "machine" in terms
        assert "learning" in terms
        assert "artificial" in terms
        assert "intelligence" in terms
        assert "computers" in terms
        assert "data" in terms
    
    def test_is_question(self):
        """Test question detection."""
        assert self.validator._is_question("What is machine learning?")
        assert self.validator._is_question("How does it work?")
        assert self.validator._is_question("Why is this important?")
        assert not self.validator._is_question("This is a statement.")
        assert not self.validator._is_question("Machine learning is cool.")
    
    def test_is_instruction(self):
        """Test instruction detection."""
        assert self.validator._is_instruction("Write a story about AI")
        assert self.validator._is_instruction("Explain quantum computing")
        assert self.validator._is_instruction("Create a diagram")
        assert not self.validator._is_instruction("What is AI?")
        assert not self.validator._is_instruction("I like technology.")
    
    def test_calculate_readability(self):
        """Test readability calculation."""
        simple_text = "The cat sat. The dog ran. The bird flew."
        complex_text = "The extraordinarily sophisticated computational methodology encompasses multifaceted algorithmic paradigms."
        
        simple_score = self.validator._calculate_readability(simple_text)
        complex_score = self.validator._calculate_readability(complex_text)
        
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1
        assert simple_score > complex_score  # Simple text should be more readable
    
    def test_calculate_grammar_score(self):
        """Test grammar score calculation."""
        good_grammar = "This is a well-written sentence. It has proper capitalization!"
        bad_grammar = "this is bad grammar no caps and no periods"
        
        good_score = self.validator._calculate_grammar_score(good_grammar)
        bad_score = self.validator._calculate_grammar_score(bad_grammar)
        
        assert 0 <= good_score <= 1
        assert 0 <= bad_score <= 1
        assert good_score > bad_score  # Good grammar should score higher
    
    def test_calculate_structure_score(self):
        """Test structure score calculation."""
        well_structured = "First paragraph with multiple sentences.\n\nSecond paragraph with more content."
        poorly_structured = "One long sentence without proper structure or organization."
        
        good_score = self.validator._calculate_structure_score(well_structured)
        bad_score = self.validator._calculate_structure_score(poorly_structured)
        
        assert 0 <= good_score <= 1
        assert 0 <= bad_score <= 1
        assert good_score >= bad_score  # Well-structured text should score at least as well
