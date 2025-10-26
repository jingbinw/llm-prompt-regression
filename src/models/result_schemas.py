"""
Test result models for storing and analyzing test outcomes.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ComparisonMetric(str, Enum):
    """Available comparison metrics."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TOKEN_COUNT = "token_count"
    RESPONSE_TIME = "response_time"
    COHERENCE_SCORE = "coherence_score"
    RELEVANCE_SCORE = "relevance_score"


class ModelResponse(BaseModel):
    """Response from a single model."""
    model_name: str = Field(description="Name of the model that generated the response")
    prompt: str = Field(description="Input prompt")
    response: str = Field(description="Model's response")
    parameters: Dict[str, Any] = Field(description="Parameters used for generation")
    response_time: float = Field(description="Time taken to generate response in seconds")
    token_count: int = Field(description="Number of tokens in the response")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "gpt-3.5-turbo",
                "prompt": "Explain quantum computing",
                "response": "Quantum computing is a type of computation...",
                "parameters": {"temperature": 0.7, "max_tokens": 100},
                "response_time": 1.234,
                "token_count": 45,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
    )


class ComparisonResult(BaseModel):
    """Result of comparing two model responses."""
    prompt: str = Field(description="Input prompt")
    model_1_response: ModelResponse = Field(description="Response from first model")
    model_2_response: ModelResponse = Field(description="Response from second model")
    metrics: Dict[ComparisonMetric, Union[float, bool]] = Field(
        description="Comparison metrics between the responses"
    )
    drift_detected: bool = Field(description="Whether significant drift was detected")
    drift_severity: str = Field(description="Severity of drift: low, medium, high")
    drift_explanation: Optional[str] = Field(default=None, description="Explanation of the drift")
    timestamp: datetime = Field(default_factory=datetime.now, description="When comparison was made")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Explain quantum computing",
                "model_1_response": {...},
                "model_2_response": {...},
                "metrics": {
                    "exact_match": False,
                    "semantic_similarity": 0.85,
                    "token_count": 5
                },
                "drift_detected": True,
                "drift_severity": "medium",
                "drift_explanation": "Significant difference in response length"
            }
        }
    )


class TestResult(BaseModel):
    """Result of a single test execution."""
    test_name: str = Field(description="Name of the test")
    status: TestStatus = Field(description="Status of the test execution")
    start_time: datetime = Field(description="When the test started")
    end_time: Optional[datetime] = Field(default=None, description="When the test ended")
    duration: Optional[float] = Field(default=None, description="Test duration in seconds")
    total_prompts: int = Field(description="Total number of prompts tested")
    total_comparisons: int = Field(description="Total number of comparisons made")
    comparison_results: List[ComparisonResult] = Field(
        default_factory=list, 
        description="Results of individual comparisons"
    )
    summary: Dict[str, Any] = Field(default_factory=dict, description="Test summary statistics")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional test metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "test_name": "Basic Model Comparison",
                "status": "completed",
                "start_time": "2024-01-01T12:00:00Z",
                "end_time": "2024-01-01T12:05:00Z",
                "duration": 300.0,
                "total_prompts": 5,
                "total_comparisons": 10,
                "comparison_results": [...],
                "summary": {
                    "drift_detected_count": 3,
                    "average_similarity": 0.78,
                    "success_rate": 0.9
                }
            }
        }
    )


class TestSuiteResult(BaseModel):
    """Result of a complete test suite execution."""
    suite_name: str = Field(description="Name of the test suite")
    start_time: datetime = Field(description="When the suite started")
    end_time: Optional[datetime] = Field(default=None, description="When the suite ended")
    duration: Optional[float] = Field(default=None, description="Suite duration in seconds")
    test_results: List[TestResult] = Field(description="Results of individual tests")
    overall_status: TestStatus = Field(description="Overall status of the suite")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Suite summary statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional suite metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suite_name": "Comprehensive Model Comparison",
                "start_time": "2024-01-01T12:00:00Z",
                "end_time": "2024-01-01T12:30:00Z",
                "duration": 1800.0,
                "test_results": [...],
                "overall_status": "completed",
                "summary": {
                    "total_tests": 3,
                    "passed_tests": 2,
                    "failed_tests": 1,
                    "total_drift_detected": 5
                }
            }
        }
    )


class DriftReport(BaseModel):
    """Detailed drift analysis report."""
    report_id: str = Field(description="Unique identifier for the report")
    generated_at: datetime = Field(default_factory=datetime.now, description="When report was generated")
    test_suite_result: TestSuiteResult = Field(description="Test suite result being analyzed")
    drift_summary: Dict[str, Any] = Field(description="Summary of drift findings")
    detailed_analysis: List[Dict[str, Any]] = Field(description="Detailed drift analysis")
    recommendations: List[str] = Field(description="Recommendations based on findings")
    charts_data: Dict[str, Any] = Field(default_factory=dict, description="Data for generating charts")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "report_id": "report_2024_01_01_001",
                "generated_at": "2024-01-01T12:30:00Z",
                "test_suite_result": {...},
                "drift_summary": {
                    "total_drift_instances": 5,
                    "high_severity_drift": 1,
                    "medium_severity_drift": 2,
                    "low_severity_drift": 2
                },
                "detailed_analysis": [...],
                "recommendations": [
                    "Monitor model consistency more closely",
                    "Consider adjusting temperature parameters"
                ]
            }
        }
    )
