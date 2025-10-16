"""
End-to-end integration tests for the LLM prompt regression testing framework.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from src.core.test_runner import TestRunner
from src.core.report_generator import ReportGenerator
from src.utils.config_loader import ConfigLoader
from src.models.test_result import TestSuiteResult, TestStatus


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.report_dir = Path(self.temp_dir) / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        # Mock API key for testing
        self.api_key = "test-api-key"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_basic_test_execution(self, mock_openai):
        """Test basic test execution with mocked OpenAI API."""
        # Setup mock responses
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "This is a test response from GPT-3.5-turbo."
        mock_response.usage = AsyncMock()
        mock_response.usage.dict.return_value = {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create test configuration
        config_loader = ConfigLoader()
        config = config_loader.create_default_config()
        config.prompts = ["Test prompt"]
        config.models = config.models[:1]  # Use only one model for simplicity
        config.parameter_variations = []  # No variations for basic test
        
        # Run test
        runner = TestRunner(self.api_key)
        result = await runner.run_test(config)
        
        # Verify results
        assert result.status == TestStatus.COMPLETED
        assert result.test_name == config.test_name
        assert len(result.comparison_results) == 0  # No comparisons with single model
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_model_comparison_execution(self, mock_openai):
        """Test model comparison with mocked responses."""
        # Setup different mock responses for different models
        def create_mock_response(content):
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.content = content
            mock_response.usage = AsyncMock()
            mock_response.usage.dict.return_value = {"total_tokens": len(content.split()), "prompt_tokens": 2, "completion_tokens": len(content.split()) - 2}
            return mock_response
        
        mock_client = AsyncMock()
        
        # Setup responses for different models
        responses = {
            "gpt-3.5-turbo": create_mock_response("GPT-3.5 response about machine learning."),
            "gpt-4": create_mock_response("GPT-4 response about machine learning with more detail.")
        }
        
        async def mock_create(**kwargs):
            model = kwargs.get('model')
            return responses.get(model, responses["gpt-3.5-turbo"])
        
        mock_client.chat.completions.create.side_effect = mock_create
        mock_openai.return_value = mock_client
        
        # Create test configuration with two models
        config_loader = ConfigLoader()
        config = config_loader.create_default_config()
        config.prompts = ["Explain machine learning"]
        config.models = config.models[:2]  # Use both models
        config.parameter_variations = []  # No variations for simplicity
        
        # Run test
        runner = TestRunner(self.api_key)
        result = await runner.run_test(config)
        
        # Verify results
        assert result.status == TestStatus.COMPLETED
        assert len(result.comparison_results) == 1  # One comparison between two models
        assert result.comparison_results[0].prompt == "Explain machine learning"
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_parameter_variation_execution(self, mock_openai):
        """Test parameter variation execution."""
        # Setup mock responses
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Response with different parameters."
        mock_response.usage = AsyncMock()
        mock_response.usage.dict.return_value = {"total_tokens": 6, "prompt_tokens": 2, "completion_tokens": 4}
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create test configuration with parameter variations
        config_loader = ConfigLoader()
        config = config_loader.create_parameter_variation_config()
        config.prompts = ["Test prompt"]
        config.parameter_variations = config.parameter_variations[:3]  # Limit for testing
        
        # Run test
        runner = TestRunner(self.api_key)
        result = await runner.run_test(config)
        
        # Verify results
        assert result.status == TestStatus.COMPLETED
        assert len(result.comparison_results) > 0  # Should have comparisons between variations
    
    def test_report_generation(self):
        """Test report generation functionality."""
        # Create a mock test suite result
        from src.llm_prompt_regression.models.test_result import TestResult, ComparisonResult, ModelResponse
        from datetime import datetime
        
        # Create mock model responses
        response1 = ModelResponse(
            model_name="gpt-3.5-turbo",
            prompt="Test prompt",
            response="Response from GPT-3.5",
            parameters={"temperature": 0.7},
            response_time=1.0,
            token_count=10,
            timestamp=datetime.now()
        )
        
        response2 = ModelResponse(
            model_name="gpt-4",
            prompt="Test prompt",
            response="Response from GPT-4",
            parameters={"temperature": 0.7},
            response_time=1.5,
            token_count=12,
            timestamp=datetime.now()
        )
        
        # Create comparison result
        comparison = ComparisonResult(
            prompt="Test prompt",
            model_1_response=response1,
            model_2_response=response2,
            metrics={
                "exact_match": False,
                "semantic_similarity": 0.85,
                "token_count": 2,
                "response_time": 0.5
            },
            drift_detected=True,
            drift_severity="medium",
            drift_explanation="Moderate differences detected"
        )
        
        # Create test result
        test_result = TestResult(
            test_name="Test",
            status=TestStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_prompts=1,
            total_comparisons=1,
            comparison_results=[comparison]
        )
        
        # Create test suite result
        suite_result = TestSuiteResult(
            suite_name="Test Suite",
            start_time=datetime.now(),
            end_time=datetime.now(),
            test_results=[test_result],
            overall_status=TestStatus.COMPLETED
        )
        
        # Generate report
        report_generator = ReportGenerator(output_dir=str(self.report_dir))
        drift_report = report_generator.generate_drift_report(suite_result)
        
        # Verify report
        assert drift_report.report_id is not None
        assert drift_report.drift_summary["total_comparisons"] == 1
        assert drift_report.drift_summary["drift_detected_count"] == 1
        assert len(drift_report.recommendations) > 0
        
        # Test HTML report generation
        html_file = report_generator.generate_html_report(drift_report)
        assert Path(html_file).exists()
        
        # Test CSV report generation
        csv_file = report_generator.generate_csv_report(suite_result)
        assert Path(csv_file).exists()
    
    def test_config_loading_and_validation(self):
        """Test configuration loading and validation."""
        # Create a temporary config file
        config_content = """
test_name: "Integration Test Config"
prompts:
  - "Integration test prompt"
models:
  - name: "gpt-3.5-turbo"
    model_type: "gpt-3.5-turbo"
    parameters:
      temperature: 0.7
      max_tokens: 100
parameter_variations:
  - temperature: 0.0
  - temperature: 1.0
max_retries: 3
request_timeout: 30
batch_size: 5
output_dir: "./reports"
"""
        
        config_file = Path(self.temp_dir) / "integration_test.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Load configuration
        config_loader = ConfigLoader(config_dir=str(self.temp_dir))
        config = config_loader.load_test_config("integration_test.yaml")
        
        # Verify configuration
        assert config.test_name == "Integration Test Config"
        assert len(config.prompts) == 1
        assert len(config.models) == 1
        assert len(config.parameter_variations) == 2
        assert config.models[0].name == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with invalid API key
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client
            
            config_loader = ConfigLoader()
            config = config_loader.create_default_config()
            config.prompts = ["Test prompt"]
            config.models = config.models[:1]
            config.parameter_variations = []
            
            runner = TestRunner("invalid-key")
            result = await runner.run_test(config)
            
            # Should handle error gracefully
            assert result.status == TestStatus.FAILED
            assert len(result.errors) > 0
    
    def test_file_outputs(self):
        """Test that all expected output files are created."""
        # Create mock data
        from src.llm_prompt_regression.models.test_result import TestResult, ComparisonResult, ModelResponse
        from datetime import datetime
        
        response1 = ModelResponse(
            model_name="gpt-3.5-turbo",
            prompt="Test prompt",
            response="Test response",
            parameters={"temperature": 0.7},
            response_time=1.0,
            token_count=5,
            timestamp=datetime.now()
        )
        
        response2 = ModelResponse(
            model_name="gpt-4",
            prompt="Test prompt",
            response="Test response",
            parameters={"temperature": 0.7},
            response_time=1.0,
            token_count=5,
            timestamp=datetime.now()
        )
        
        comparison = ComparisonResult(
            prompt="Test prompt",
            model_1_response=response1,
            model_2_response=response2,
            metrics={"exact_match": True, "semantic_similarity": 1.0},
            drift_detected=False,
            drift_severity="low"
        )
        
        test_result = TestResult(
            test_name="File Output Test",
            status=TestStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_prompts=1,
            total_comparisons=1,
            comparison_results=[comparison]
        )
        
        suite_result = TestSuiteResult(
            suite_name="File Output Test Suite",
            start_time=datetime.now(),
            end_time=datetime.now(),
            test_results=[test_result],
            overall_status=TestStatus.COMPLETED
        )
        
        # Generate reports
        report_generator = ReportGenerator(output_dir=str(self.report_dir))
        drift_report = report_generator.generate_drift_report(suite_result)
        
        # Check that files are created
        report_files = list(self.report_dir.glob("*"))
        assert len(report_files) > 0
        
        # Check for specific file types
        json_files = list(self.report_dir.glob("*.json"))
        html_files = list(self.report_dir.glob("*.html"))
        csv_files = list(self.report_dir.glob("*.csv"))
        
        assert len(json_files) > 0  # Should have drift report JSON
        assert len(html_files) > 0  # Should have HTML report
        assert len(csv_files) > 0   # Should have CSV report
