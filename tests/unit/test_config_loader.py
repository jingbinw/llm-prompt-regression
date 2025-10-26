"""
Unit tests for configuration loader utilities.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.utils.config_loader import ConfigLoader
from src.models.config_schemas import TestConfig, ModelType


class TestConfigLoader:
    """Test cases for ConfigLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ConfigLoader(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = self.loader.create_default_config()
        
        assert isinstance(config, TestConfig)
        assert config.test_name == "Default Regression Test"
        assert len(config.prompts) > 0
        assert len(config.models) == 2
        assert config.models[0].model_type == ModelType.GPT_35_TURBO
        assert config.models[1].model_type == ModelType.GPT_4
        assert len(config.parameter_variations) > 0
    
    def test_create_parameter_variation_config(self):
        """Test parameter variation configuration creation."""
        config = self.loader.create_parameter_variation_config()
        
        assert isinstance(config, TestConfig)
        assert config.test_name == "Parameter Variation Test"
        assert len(config.models) == 1
        assert len(config.parameter_variations) > 10  # Should have many variations
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration."""
        yaml_content = """
test_name: "Test Configuration"
prompts:
  - "Test prompt 1"
  - "Test prompt 2"
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
        
        config_file = Path(self.temp_dir) / "test_config.yaml"
        with open(config_file, 'w') as f:
            f.write(yaml_content)
        
        config = self.loader.load_test_config("test_config.yaml")
        
        assert isinstance(config, TestConfig)
        assert config.test_name == "Test Configuration"
        assert len(config.prompts) == 2
        assert len(config.models) == 1
        assert config.models[0].name == "gpt-3.5-turbo"
        assert len(config.parameter_variations) == 2
    
    def test_load_json_config(self):
        """Test loading JSON configuration."""
        json_content = """
{
  "test_name": "JSON Test Configuration",
  "prompts": ["JSON test prompt"],
  "models": [
    {
      "name": "gpt-4",
      "model_type": "gpt-4",
      "parameters": {
        "temperature": 0.8,
        "max_tokens": 200
      }
    }
  ],
  "parameter_variations": [
    {"temperature": 0.0},
    {"temperature": 1.0}
  ],
  "max_retries": 5,
  "request_timeout": 45,
  "batch_size": 3,
  "output_dir": "./json_reports"
}
"""
        
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            f.write(json_content)
        
        config = self.loader.load_test_config("test_config.json")
        
        assert isinstance(config, TestConfig)
        assert config.test_name == "JSON Test Configuration"
        assert len(config.prompts) == 1
        assert len(config.models) == 1
        assert config.models[0].name == "gpt-4"
        assert config.max_retries == 5
        assert config.request_timeout == 45
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_test_config("nonexistent.yaml")
    
    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        config_file = Path(self.temp_dir) / "test_config.txt"
        with open(config_file, 'w') as f:
            f.write("This is not a valid config file")
        
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            self.loader.load_test_config("test_config.txt")
    
    def test_save_config_yaml(self):
        """Test saving configuration to YAML."""
        config = self.loader.create_default_config()
        config.test_name = "Saved Test Config"
        
        output_file = Path(self.temp_dir) / "saved_config.yaml"
        self.loader.save_config(config, str(output_file), format='yaml')
        
        assert output_file.exists()
        
        # Load it back and verify
        loaded_config = self.loader.load_test_config("saved_config.yaml")
        assert loaded_config.test_name == "Saved Test Config"
    
    def test_save_config_json(self):
        """Test saving configuration to JSON."""
        config = self.loader.create_default_config()
        config.test_name = "Saved JSON Config"
        
        output_file = Path(self.temp_dir) / "saved_config.json"
        self.loader.save_config(config, str(output_file), format='json')
        
        assert output_file.exists()
        
        # Load it back and verify
        loaded_config = self.loader.load_test_config("saved_config.json")
        assert loaded_config.test_name == "Saved JSON Config"
    
    def test_get_environment_config(self):
        """Test getting configuration from environment variables."""
        # Set some test environment variables
        os.environ['TEST_OPENAI_API_KEY'] = 'test-key-123'
        os.environ['TEST_DEFAULT_MODEL_1'] = 'gpt-3.5-turbo'
        os.environ['TEST_MAX_RETRIES'] = '5'
        
        # Create a loader with modified environment variable names
        loader = ConfigLoader()
        
        # Mock the environment variables for this test
        with pytest.MonkeyPatch().context() as m:
            m.setenv('OPENAI_API_KEY', 'test-key-123')
            m.setenv('DEFAULT_MODEL_1', 'gpt-3.5-turbo')
            m.setenv('MAX_RETRIES', '5')
            
            env_config = loader.get_environment_config()
            
            assert env_config['api_key'] == 'test-key-123'
            assert env_config['default_model_1'] == 'gpt-3.5-turbo'
            assert env_config['max_retries'] == 5
    
    def test_load_test_suite_config(self):
        """Test loading test suite configuration."""
        yaml_content = """
suite_name: "Test Suite"
description: "A test suite"
tests:
  - test_name: "Test 1"
    prompts: ["Prompt 1"]
    models:
      - name: "gpt-3.5-turbo"
        model_type: "gpt-3.5-turbo"
        parameters:
          temperature: 0.7
    parameter_variations: []
    max_retries: 3
    request_timeout: 30
    batch_size: 5
global_settings:
  output_dir: "./suite_reports"
"""
        
        config_file = Path(self.temp_dir) / "test_suite.yaml"
        with open(config_file, 'w') as f:
            f.write(yaml_content)
        
        suite_config = self.loader.load_test_suite_config("test_suite.yaml")
        
        assert suite_config.suite_name == "Test Suite"
        assert len(suite_config.tests) == 1
        assert suite_config.tests[0].test_name == "Test 1"
        assert suite_config.global_settings['output_dir'] == "./suite_reports"
