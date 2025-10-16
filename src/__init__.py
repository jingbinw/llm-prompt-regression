"""
LLM Prompt Regression Testing Framework

A comprehensive framework for testing LLM output consistency across different
model versions and parameter configurations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.test_runner import TestRunner
from .models.test_config import TestConfig, ModelConfig, ParameterConfig
from .models.test_result import TestResult, ComparisonResult
from .core.report_generator import ReportGenerator

__all__ = [
    "TestRunner",
    "TestConfig", 
    "ModelConfig",
    "ParameterConfig",
    "TestResult",
    "ComparisonResult",
    "ReportGenerator",
]
