"""
Metrics calculation utilities for comparing LLM responses.

Supports two paths for semantic similarity:
- Default: TF-IDF + cosine similarity (no token cost)
- Optional: LangChain OpenAIEmbeddings + cosine similarity (token cost)

Enable embeddings via constructor (use_embeddings=True) or env USE_EMBEDDINGS=true.
If embeddings are unavailable or fail, it silently falls back to TF-IDF.
"""

import asyncio
import os
import re
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# Optional LangChain embeddings (used only if enabled and installed)
try:  # pragma: no cover - import availability depends on environment
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore


class MetricsCalculator:
    """Calculator for various metrics to compare LLM responses.

    Args:
        use_embeddings: If True, try to use LangChain OpenAIEmbeddings for
            semantic similarity (token cost). If False, use TF-IDF only.
            If None, read from env USE_EMBEDDINGS (default False).
    """

    def __init__(self, use_embeddings: bool | None = None):
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )

        # Determine whether to enable embeddings
        if use_embeddings is None:
            use_embeddings = os.getenv("USE_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}

        # Initialize embeddings client if requested and available
        self._embeddings = None
        if use_embeddings and OpenAIEmbeddings is not None:
            try:  # pragma: no cover - external service init
                self._embeddings = OpenAIEmbeddings()
            except Exception:
                # Fallback will be TF-IDF if initialization fails
                self._embeddings = None
    
    async def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Preference order:
        1) If LangChain OpenAIEmbeddings is enabled and available, use embeddings
           with cosine similarity (token cost).
        2) Otherwise, use TF-IDF + cosine similarity (no token cost).
        3) On failure, fall back to simple word overlap ratio.

        Returns a score between 0 and 1.
        """
        # Attempt embeddings-based similarity if configured
        if self._embeddings is not None:
            try:  # pragma: no cover - external call
                clean1 = self._clean_text(text1)
                clean2 = self._clean_text(text2)
                v1, v2 = await asyncio.gather(
                    asyncio.to_thread(self._embeddings.embed_query, clean1),
                    asyncio.to_thread(self._embeddings.embed_query, clean2),
                )
                v1 = np.array(v1).reshape(1, -1)
                v2 = np.array(v2).reshape(1, -1)
                similarity = float(cosine_similarity(v1, v2)[0][0])
                # Normalize to [0,1] bounds just in case
                return max(0.0, min(1.0, similarity))
            except Exception:
                # If embeddings path fails, fall through to TF-IDF
                pass

        # TF-IDF similarity (default path)
        try:
            texts = [self._clean_text(text1), self._clean_text(text2)]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            # Fallback to simple word overlap
            return self._calculate_word_overlap(text1, text2)
    
    def calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate lexical similarity using Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        words1 = set(self._clean_text(text1).lower().split())
        words2 = set(self._clean_text(text2).lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_token_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate token-level similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Token similarity score between 0 and 1
        """
        tokens1 = set(self.token_encoder.encode(text1))
        tokens2 = set(self.token_encoder.encode(text2))
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_length_metrics(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Calculate various length-related metrics.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary containing length metrics
        """
        len1, len2 = len(text1), len(text2)
        tokens1 = len(self.token_encoder.encode(text1))
        tokens2 = len(self.token_encoder.encode(text2))
        
        return {
            "char_length_diff": abs(len1 - len2),
            "char_length_ratio": min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0,
            "token_length_diff": abs(tokens1 - tokens2),
            "token_length_ratio": min(tokens1, tokens2) / max(tokens1, tokens2) if max(tokens1, tokens2) > 0 else 1.0,
            "word_count_diff": abs(len(text1.split()) - len(text2.split()))
        }
    
    def calculate_structural_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Calculate structural similarity metrics.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary containing structural metrics
        """
        return {
            "sentence_count_diff": abs(self._count_sentences(text1) - self._count_sentences(text2)),
            "paragraph_count_diff": abs(self._count_paragraphs(text1) - self._count_paragraphs(text2)),
            "question_count_diff": abs(self._count_questions(text1) - self._count_questions(text2)),
            "exclamation_count_diff": abs(text1.count('!') - text2.count('!')),
            "has_same_ending": text1.strip().endswith(text2.strip()[-10:]) if len(text2.strip()) >= 10 else False
        }
    
    def calculate_content_overlap(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Calculate content overlap metrics.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary containing overlap metrics
        """
        words1 = set(self._clean_text(text1).lower().split())
        words2 = set(self._clean_text(text2).lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Extract named entities or key phrases
        key_phrases1 = self._extract_key_phrases(text1)
        key_phrases2 = self._extract_key_phrases(text2)
        
        return {
            "word_overlap_ratio": len(intersection) / len(union) if union else 0.0,
            "unique_words_text1": len(words1 - words2),
            "unique_words_text2": len(words2 - words1),
            "key_phrase_overlap": len(set(key_phrases1).intersection(set(key_phrases2))),
            "key_phrase_ratio": len(set(key_phrases1).intersection(set(key_phrases2))) / 
                               max(len(set(key_phrases1)), len(set(key_phrases2))) 
                               if max(len(set(key_phrases1)), len(set(key_phrases2))) > 0 else 0.0
        }
    
    async def calculate_comprehensive_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Calculate comprehensive similarity metrics.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary containing all similarity metrics
        """
        return {
            "semantic_similarity": await self.calculate_semantic_similarity(text1, text2),
            "lexical_similarity": self.calculate_lexical_similarity(text1, text2),
            "token_similarity": self.calculate_token_similarity(text1, text2),
            "length_metrics": self.calculate_length_metrics(text1, text2),
            "structural_similarity": self.calculate_structural_similarity(text1, text2),
            "content_overlap": self.calculate_content_overlap(text1, text2)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Fallback word overlap calculation."""
        words1 = set(self._clean_text(text1).lower().split())
        words2 = set(self._clean_text(text2).lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        return len(re.findall(r'[.!?]+', text))
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text."""
        return len([p for p in text.split('\n\n') if p.strip()])
    
    def _count_questions(self, text: str) -> int:
        """Count questions in text."""
        return text.count('?')
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text (simple implementation)."""
        # Simple key phrase extraction based on capitalized words and common patterns
        words = text.split()
        key_phrases = []
        
        for i, word in enumerate(words):
            # Capitalized words (potential proper nouns)
            if word[0].isupper() and len(word) > 2:
                key_phrases.append(word)
            
            # Common technical terms
            if word.lower() in ['api', 'ai', 'ml', 'llm', 'gpt', 'openai']:
                key_phrases.append(word.lower())
        
        return key_phrases
