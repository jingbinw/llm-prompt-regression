"""
Test configuration models for LLM prompt regression testing.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"


class ParameterConfig(BaseModel):
    """Configuration for model parameters."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens to generate")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 0.7,
                "top_p": 1.0,
                "max_tokens": 1000,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        }


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str = Field(description="Model name (e.g., gpt-3.5-turbo)")
    model_type: ModelType = Field(description="Type of the model")
    parameters: ParameterConfig = Field(default_factory=ParameterConfig, description="Model parameters")
    description: Optional[str] = Field(default=None, description="Optional description of the model")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "gpt-3.5-turbo",
                "model_type": "gpt-3.5-turbo",
                "parameters": {
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "max_tokens": 1000
                },
                "description": "GPT-3.5 Turbo model for comparison testing"
            }
        }


class TestConfig(BaseModel):
    """Main test configuration."""
    test_name: str = Field(description="Name of the test suite")
    prompts: List[str] = Field(description="List of prompts to test")
    models: List[ModelConfig] = Field(description="List of models to compare")
    parameter_variations: List[ParameterConfig] = Field(
        default_factory=list, 
        description="Additional parameter variations to test"
    )
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum number of retries for failed requests")
    request_timeout: int = Field(default=30, ge=5, le=120, description="Request timeout in seconds")
    batch_size: int = Field(default=5, ge=1, le=20, description="Batch size for concurrent requests")
    output_dir: str = Field(default="./reports", description="Output directory for reports")
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_name": "Basic Prompt Regression Test",
                "prompts": [
                    "Explain quantum computing in simple terms.",
                    "Write a haiku about artificial intelligence."
                ],
                "models": [
                    {
                        "name": "gpt-3.5-turbo",
                        "model_type": "gpt-3.5-turbo",
                        "parameters": {"temperature": 0.7}
                    },
                    {
                        "name": "gpt-4",
                        "model_type": "gpt-4", 
                        "parameters": {"temperature": 0.7}
                    }
                ],
                "parameter_variations": [
                    {"temperature": 0.0},
                    {"temperature": 1.0}
                ]
            }
        }


class TestSuiteConfig(BaseModel):
    """Configuration for a test suite containing multiple test configurations."""
    suite_name: str = Field(description="Name of the test suite")
    description: Optional[str] = Field(default=None, description="Description of the test suite")
    tests: List[TestConfig] = Field(description="List of test configurations")
    global_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global settings applied to all tests"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "suite_name": "Comprehensive Model Comparison",
                "description": "Test suite for comparing different OpenAI models",
                "tests": [
                    {
                        "test_name": "Basic Functionality",
                        "prompts": ["Hello, how are you?"],
                        "models": [
                            {"name": "gpt-3.5-turbo", "model_type": "gpt-3.5-turbo"},
                            {"name": "gpt-4", "model_type": "gpt-4"}
                        ]
                    }
                ]
            }
        }
