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
    
    def test_environment_loading_from_env_example(self):
        """Test loading environment variables from env.example (safe for CI)."""
        # Create a fresh loader to load environment
        loader = ConfigLoader()
        
        # Test that config uses expected default values
        # (either from env.example or fallback defaults in create_default_config)
        config = loader.create_default_config()
        assert config.max_retries == 3
        assert config.request_timeout == 30
        assert config.batch_size == 5
        assert config.output_dir == './reports'
