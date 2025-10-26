"""
Core test runner for executing LLM prompt regression tests.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import tiktoken

from ..models.config_schemas import TestConfig, ModelConfig
from ..models.result_schemas import (
    TestResult, ComparisonResult, ModelResponse, 
    TestStatus, ComparisonMetric
)
from ..utils.metrics import MetricsCalculator
from ..utils.validators import ResponseValidator


logger = logging.getLogger(__name__)


class TestRunner:
    """Main test runner for LLM prompt regression testing."""
    
    def __init__(self, api_key: str, max_retries: int = 3, timeout: int = 30):
        """
        Initialize the test runner.
        
        Args:
            api_key: OpenAI API key
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.max_retries = max_retries
        self.timeout = timeout
        self.metrics_calculator = MetricsCalculator()
        self.validator = ResponseValidator()
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        
    async def run_test(self, config: TestConfig) -> TestResult:
        """
        Run a single test configuration.
        
        Args:
            config: Test configuration
            
        Returns:
            Test result containing all comparison results
        """
        logger.info(f"Starting test: {config.test_name}")
        start_time = datetime.now()
        
        test_result = TestResult(
            test_name=config.test_name,
            status=TestStatus.RUNNING,
            start_time=start_time,
            total_prompts=len(config.prompts),
            total_comparisons=0
        )
        
        try:
            # Generate all model responses
            all_responses = await self._generate_responses(config)
            
            # Perform comparisons
            comparison_results = []
            for prompt in config.prompts:
                prompt_responses = [r for r in all_responses if r.prompt == prompt]
                
                # Compare each pair of models
                for i in range(len(prompt_responses)):
                    for j in range(i + 1, len(prompt_responses)):
                        comparison = await self._compare_responses(
                            prompt_responses[i], 
                            prompt_responses[j],
                            config
                        )
                        comparison_results.append(comparison)
            
            test_result.comparison_results = comparison_results
            test_result.total_comparisons = len(comparison_results)
            test_result.status = TestStatus.COMPLETED
            test_result.summary = self._generate_test_summary(comparison_results)
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            test_result.status = TestStatus.FAILED
            test_result.errors.append(str(e))
        
        finally:
            test_result.end_time = datetime.now()
            if test_result.start_time:
                test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
        
        logger.info(f"Test completed: {config.test_name}, Status: {test_result.status}")
        return test_result
    
    async def _generate_responses(self, config: TestConfig) -> List[ModelResponse]:
        """Generate responses from all models for all prompts."""
        all_responses = []
        
        # Create tasks for all combinations
        tasks = []
        for model in config.models:
            for prompt in config.prompts:
                # Base parameters
                base_params = model.parameters.dict()
                
                # Add parameter variations
                if config.parameter_variations:
                    for variation in config.parameter_variations:
                        params = {**base_params, **variation.dict()}
                        task = self._generate_single_response(
                            model.name, prompt, params, config
                        )
                        tasks.append(task)
                else:
                    task = self._generate_single_response(
                        model.name, prompt, base_params, config
                    )
                    tasks.append(task)
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and collect valid responses
        for response in responses:
            if isinstance(response, ModelResponse):
                all_responses.append(response)
            elif isinstance(response, Exception):
                logger.error(f"Failed to generate response: {str(response)}")
        
        return all_responses
    
    async def _generate_single_response(
        self, 
        model_name: str, 
        prompt: str, 
        parameters: Dict[str, Any],
        config: TestConfig
    ) -> ModelResponse:
        """Generate a single response from a model."""
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=parameters.get("temperature", 0.7),
                    top_p=parameters.get("top_p", 1.0),
                    max_tokens=parameters.get("max_tokens", 1000),
                    frequency_penalty=parameters.get("frequency_penalty", 0.0),
                    presence_penalty=parameters.get("presence_penalty", 0.0),
                    timeout=self.timeout
                )
                
                response_text = response.choices[0].message.content or ""
                response_time = time.time() - start_time
                token_count = len(self.token_encoder.encode(response_text))
                
                return ModelResponse(
                    model_name=model_name,
                    prompt=prompt,
                    response=response_text,
                    parameters=parameters,
                    response_time=response_time,
                    token_count=token_count,
                    timestamp=datetime.now(),
                    metadata={
                        "attempt": attempt + 1,
                        "usage": response.usage.dict() if response.usage else None
                    }
                )
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed to generate response after {self.max_retries} attempts")
    
    async def _compare_responses(
        self, 
        response1: ModelResponse, 
        response2: ModelResponse,
        config: TestConfig
    ) -> ComparisonResult:
        """Compare two model responses and calculate metrics."""
        
        # Calculate all available metrics
        metrics = {}
        
        # Exact match
        metrics[ComparisonMetric.EXACT_MATCH] = response1.response == response2.response
        
        # Semantic similarity
        similarity = await self.metrics_calculator.calculate_semantic_similarity(
            response1.response, response2.response
        )
        metrics[ComparisonMetric.SEMANTIC_SIMILARITY] = similarity
        
        # Token count difference
        token_diff = abs(response1.token_count - response2.token_count)
        metrics[ComparisonMetric.TOKEN_COUNT] = token_diff
        
        # Response time difference
        time_diff = abs(response1.response_time - response2.response_time)
        metrics[ComparisonMetric.RESPONSE_TIME] = time_diff
        
        # Coherence scores (if available)
        try:
            coherence1 = await self.validator.validate_coherence(response1.response)
            coherence2 = await self.validator.validate_coherence(response2.response)
            metrics[ComparisonMetric.COHERENCE_SCORE] = abs(coherence1 - coherence2)
        except Exception as e:
            logger.warning(f"Could not calculate coherence scores: {str(e)}")
            metrics[ComparisonMetric.COHERENCE_SCORE] = None
        
        # Determine if drift is detected
        drift_detected, severity, explanation = self._analyze_drift(metrics)
        
        return ComparisonResult(
            prompt=response1.prompt,
            model_1_response=response1,
            model_2_response=response2,
            metrics=metrics,
            drift_detected=drift_detected,
            drift_severity=severity,
            drift_explanation=explanation,
            timestamp=datetime.now()
        )
    
    def _analyze_drift(self, metrics: Dict[ComparisonMetric, Any]) -> Tuple[bool, str, str]:
        """Analyze metrics to determine if drift is detected."""
        drift_detected = False
        severity = "low"
        explanations = []
        
        # Check exact match
        if metrics.get(ComparisonMetric.EXACT_MATCH) is False:
            drift_detected = True
            explanations.append("Responses are not identical")
        
        # Check semantic similarity
        similarity = metrics.get(ComparisonMetric.SEMANTIC_SIMILARITY, 1.0)
        if similarity < 0.8:
            drift_detected = True
            if similarity < 0.6:
                severity = "high"
                explanations.append("Low semantic similarity")
            else:
                severity = "medium"
                explanations.append("Moderate semantic similarity")
        
        # Check token count difference
        token_diff = metrics.get(ComparisonMetric.TOKEN_COUNT, 0)
        if token_diff > 50:  # Significant length difference
            drift_detected = True
            if token_diff > 100:
                severity = "high"
            explanations.append(f"Significant token count difference: {token_diff}")
        
        # Check response time difference
        time_diff = metrics.get(ComparisonMetric.RESPONSE_TIME, 0)
        if time_diff > 2.0:  # More than 2 seconds difference
            drift_detected = True
            explanations.append(f"Significant response time difference: {time_diff:.2f}s")
        
        explanation = "; ".join(explanations) if explanations else "No significant drift detected"
        
        return drift_detected, severity, explanation
    
    def _generate_test_summary(self, comparisons: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate summary statistics for the test."""
        total_comparisons = len(comparisons)
        drift_detected_count = sum(1 for c in comparisons if c.drift_detected)
        
        # Calculate average similarity
        similarities = [
            c.metrics.get(ComparisonMetric.SEMANTIC_SIMILARITY, 1.0) 
            for c in comparisons 
            if ComparisonMetric.SEMANTIC_SIMILARITY in c.metrics
        ]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        
        # Calculate drift severity distribution
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        for comparison in comparisons:
            if comparison.drift_detected:
                severity_counts[comparison.drift_severity] += 1
        
        return {
            "total_comparisons": total_comparisons,
            "drift_detected_count": drift_detected_count,
            "drift_rate": drift_detected_count / total_comparisons if total_comparisons > 0 else 0,
            "average_similarity": avg_similarity,
            "severity_distribution": severity_counts,
            "success_rate": 1.0 - (drift_detected_count / total_comparisons) if total_comparisons > 0 else 1.0
        }
