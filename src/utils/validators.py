"""
Response validation utilities for LLM outputs.
"""

import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from openai import AsyncOpenAI


class ResponseValidator:
    """Validator for LLM responses using various criteria."""
    
    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """
        Initialize the validator.
        
        Args:
            client: OpenAI client for advanced validation
        """
        self.client = client
    
    async def validate_coherence(self, text: str) -> float:
        """
        Validate text coherence using a simple heuristic approach.
        
        Args:
            text: Text to validate
            
        Returns:
            Coherence score between 0 and 1
        """
        try:
            # Basic coherence checks
            score = 1.0
            
            # Check for proper sentence structure
            sentences = self._split_sentences(text)
            if len(sentences) > 1:
                # Check for proper capitalization
                proper_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
                cap_score = proper_caps / len(sentences)
                score *= cap_score
            
            # Check for repetitive content
            words = text.lower().split()
            if len(words) > 10:
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                max_freq = max(word_freq.values())
                repetition_penalty = min(max_freq / len(words), 0.3)
                score *= (1 - repetition_penalty)
            
            # Check for logical flow indicators
            flow_indicators = ['therefore', 'however', 'furthermore', 'moreover', 'additionally', 'consequently']
            flow_score = min(sum(1 for indicator in flow_indicators if indicator in text.lower()) / 3, 1.0)
            score = (score + flow_score) / 2
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default neutral score
    
    async def validate_relevance(self, prompt: str, response: str) -> float:
        """
        Validate relevance of response to the prompt.
        
        Args:
            prompt: Original prompt
            response: Generated response
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # Extract key terms from prompt
            prompt_terms = self._extract_key_terms(prompt)
            response_terms = self._extract_key_terms(response)
            
            # Calculate overlap
            overlap = len(set(prompt_terms).intersection(set(response_terms)))
            total_terms = len(set(prompt_terms).union(set(response_terms)))
            
            if total_terms == 0:
                return 0.5
            
            relevance_score = overlap / total_terms
            
            # Bonus for addressing the prompt type
            if self._is_question(prompt) and self._is_question(response):
                relevance_score *= 1.1
            elif self._is_instruction(prompt) and self._contains_instruction_following(response):
                relevance_score *= 1.1
            
            return min(1.0, relevance_score)
            
        except Exception:
            return 0.5
    
    async def validate_completeness(self, prompt: str, response: str) -> float:
        """
        Validate completeness of response.
        
        Args:
            prompt: Original prompt
            response: Generated response
            
        Returns:
            Completeness score between 0 and 1
        """
        try:
            score = 1.0
            
            # Check for minimum length
            if len(response.strip()) < 10:
                score *= 0.3
            
            # Check for proper ending
            if not response.strip().endswith(('.', '!', '?')):
                score *= 0.8
            
            # Check if response addresses all parts of the prompt
            if self._is_multi_part_prompt(prompt):
                parts = self._split_prompt_parts(prompt)
                addressed_parts = sum(1 for part in parts if self._part_addressed(part, response))
                completeness_ratio = addressed_parts / len(parts)
                score *= completeness_ratio
            
            # Check for appropriate detail level
            if self._is_detailed_prompt(prompt):
                if len(response.split()) < 50:
                    score *= 0.7
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    async def validate_safety(self, text: str) -> Tuple[float, List[str]]:
        """
        Validate text safety and content appropriateness.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (safety_score, warnings)
        """
        warnings = []
        score = 1.0
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'\b(violence|harm|dangerous)\b',
            r'\b(hate|discrimination|bias)\b',
            r'\b(inappropriate|offensive)\b'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, text.lower()):
                warnings.append(f"Potential harmful content detected: {pattern}")
                score *= 0.8
        
        # Check for personal information
        personal_info_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        for pattern in personal_info_patterns:
            if re.search(pattern, text):
                warnings.append("Potential personal information detected")
                score *= 0.9
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 20:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.3:
                warnings.append("Excessive repetition detected")
                score *= 0.7
        
        return max(0.0, min(1.0, score)), warnings
    
    async def validate_quality(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive quality validation.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary containing quality metrics
        """
        return {
            "coherence": await self.validate_coherence(text),
            "safety_score": (await self.validate_safety(text))[0],
            "safety_warnings": (await self.validate_safety(text))[1],
            "readability": self._calculate_readability(text),
            "grammar_score": self._calculate_grammar_score(text),
            "structure_score": self._calculate_structure_score(text)
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple key term extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        return [word for word in words if word not in stop_words]
    
    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        return text.strip().endswith('?') or text.strip().lower().startswith(('what', 'how', 'why', 'when', 'where', 'who', 'which'))
    
    def _is_instruction(self, text: str) -> bool:
        """Check if text is an instruction."""
        instruction_words = ['write', 'create', 'generate', 'make', 'build', 'explain', 'describe', 'list', 'compare', 'analyze']
        return any(text.lower().startswith(word) for word in instruction_words)
    
    def _contains_instruction_following(self, text: str) -> bool:
        """Check if response follows instruction format."""
        # Simple check for structured response
        return any(indicator in text.lower() for indicator in ['here', 'following', 'below', 'as requested', 'as asked'])
    
    def _is_multi_part_prompt(self, prompt: str) -> bool:
        """Check if prompt has multiple parts."""
        multi_part_indicators = [' and ', ' also ', ' furthermore ', ' additionally ', '\n', ';']
        return any(indicator in prompt.lower() for indicator in multi_part_indicators)
    
    def _split_prompt_parts(self, prompt: str) -> List[str]:
        """Split prompt into parts."""
        # Simple splitting based on common separators
        parts = re.split(r'\n|;| and | also | furthermore | additionally ', prompt)
        return [part.strip() for part in parts if part.strip()]
    
    def _part_addressed(self, part: str, response: str) -> bool:
        """Check if a prompt part is addressed in response."""
        part_terms = set(self._extract_key_terms(part))
        response_terms = set(self._extract_key_terms(response))
        
        overlap = len(part_terms.intersection(response_terms))
        return overlap >= len(part_terms) * 0.3  # At least 30% overlap
    
    def _is_detailed_prompt(self, prompt: str) -> bool:
        """Check if prompt requests detailed response."""
        detail_indicators = ['detailed', 'comprehensive', 'thorough', 'in-depth', 'explain in detail']
        return any(indicator in prompt.lower() for indicator in detail_indicators)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score."""
        sentences = self._split_sentences(text)
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (inverse relationship with complexity)
        readability = max(0, 1 - (avg_sentence_length / 20 + avg_word_length / 10 - 1))
        
        return min(1.0, readability)
    
    def _calculate_grammar_score(self, text: str) -> float:
        """Calculate basic grammar score."""
        score = 1.0
        
        # Check for proper capitalization
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if sentence and not sentence[0].isupper():
                score *= 0.9
        
        # Check for proper punctuation
        if text and not text.strip().endswith(('.', '!', '?')):
            score *= 0.8
        
        # Check for common grammar issues
        grammar_issues = [
            (r'\bi\b', 'I'),  # Lowercase 'i'
            (r'\s+([.!?])', r'\1'),  # Missing space after punctuation
        ]
        
        for pattern, replacement in grammar_issues:
            if re.search(pattern, text):
                score *= 0.95
        
        return score
    
    def _calculate_structure_score(self, text: str) -> float:
        """Calculate structure quality score."""
        score = 1.0
        
        # Check for paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            # Good structure if paragraphs are reasonably balanced
            para_lengths = [len(p.split()) for p in paragraphs]
            if para_lengths:
                avg_length = sum(para_lengths) / len(para_lengths)
                variance = sum((length - avg_length) ** 2 for length in para_lengths) / len(para_lengths)
                if variance < avg_length:  # Low variance is good
                    score *= 1.1
        
        # Check for logical flow
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 'consequently']
        transition_count = sum(1 for word in transition_words if word in text.lower())
        if transition_count > 0:
            score *= min(1.2, 1 + transition_count * 0.05)
        
        return min(1.0, score)
