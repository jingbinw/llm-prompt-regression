"""
Configuration loader utilities.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from ..models.test_config import TestConfig, TestSuiteConfig, ModelConfig, ParameterConfig, ModelType


class ConfigLoader:
    """Loader for test configurations from various sources."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./config")
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables."""
        # Try to load from .env file
        env_files = [".env", "env.example"]
        for env_file in env_files:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                break
    
    def load_test_config(self, config_file: str) -> TestConfig:
        """
        Load test configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Test configuration
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            return self._load_yaml_config(config_path)
        elif config_path.suffix.lower() == '.json':
            return self._load_json_config(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def load_test_suite_config(self, config_file: str) -> TestSuiteConfig:
        """
        Load test suite configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Test suite configuration
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            return self._load_yaml_suite_config(config_path)
        elif config_path.suffix.lower() == '.json':
            return self._load_json_suite_config(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def create_default_config(self) -> TestConfig:
        """
        Create a default test configuration.
        
        Returns:
            Default test configuration
        """
        return TestConfig(
            test_name="Default Regression Test",
            prompts=[
                "Explain quantum computing in simple terms.",
                "Write a haiku about artificial intelligence.",
                "What are the benefits of renewable energy?"
            ],
            models=[
                ModelConfig(
                    name="gpt-3.5-turbo",
                    model_type=ModelType.GPT_35_TURBO,
                    parameters=ParameterConfig(temperature=0.7, max_tokens=200),
                    description="GPT-3.5 Turbo for baseline comparison"
                ),
                ModelConfig(
                    name="gpt-4",
                    model_type=ModelType.GPT_4,
                    parameters=ParameterConfig(temperature=0.7, max_tokens=200),
                    description="GPT-4 for comparison testing"
                )
            ],
            parameter_variations=[
                ParameterConfig(temperature=0.0, description="Deterministic"),
                ParameterConfig(temperature=1.0, description="Creative"),
                ParameterConfig(top_p=0.5, description="Focused sampling"),
                ParameterConfig(max_tokens=100, description="Short response"),
                ParameterConfig(max_tokens=300, description="Long response")
            ],
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            batch_size=int(os.getenv("BATCH_SIZE", "5")),
            output_dir=os.getenv("REPORT_OUTPUT_DIR", "./reports")
        )
    
    def create_parameter_variation_config(self) -> TestConfig:
        """
        Create a configuration focused on parameter variations.
        
        Returns:
            Parameter variation test configuration
        """
        base_model = ModelConfig(
            name="gpt-3.5-turbo",
            model_type=ModelType.GPT_35_TURBO,
            parameters=ParameterConfig(),
            description="Base model for parameter testing"
        )
        
        return TestConfig(
            test_name="Parameter Variation Test",
            prompts=[
                "Write a creative story about a robot learning to paint.",
                "Explain the concept of machine learning to a 10-year-old.",
                "Describe the future of artificial intelligence in healthcare."
            ],
            models=[base_model],
            parameter_variations=[
                # Temperature variations
                ParameterConfig(temperature=0.0, description="Temperature 0.0 - Deterministic"),
                ParameterConfig(temperature=0.3, description="Temperature 0.3 - Conservative"),
                ParameterConfig(temperature=0.7, description="Temperature 0.7 - Balanced"),
                ParameterConfig(temperature=1.0, description="Temperature 1.0 - Creative"),
                ParameterConfig(temperature=1.5, description="Temperature 1.5 - Very Creative"),
                
                # Top-p variations
                ParameterConfig(top_p=0.1, description="Top-p 0.1 - Very Focused"),
                ParameterConfig(top_p=0.5, description="Top-p 0.5 - Focused"),
                ParameterConfig(top_p=0.9, description="Top-p 0.9 - Diverse"),
                ParameterConfig(top_p=1.0, description="Top-p 1.0 - Maximum Diversity"),
                
                # Max tokens variations
                ParameterConfig(max_tokens=50, description="Max tokens 50 - Very Short"),
                ParameterConfig(max_tokens=100, description="Max tokens 100 - Short"),
                ParameterConfig(max_tokens=200, description="Max tokens 200 - Medium"),
                ParameterConfig(max_tokens=500, description="Max tokens 500 - Long"),
                
                # Combined variations
                ParameterConfig(temperature=0.0, top_p=0.5, max_tokens=100, description="Conservative + Focused + Short"),
                ParameterConfig(temperature=1.0, top_p=1.0, max_tokens=300, description="Creative + Diverse + Long"),
            ],
            output_dir=os.getenv("REPORT_OUTPUT_DIR", "./reports")
        )
    
    def _load_yaml_config(self, config_path: Path) -> TestConfig:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return self._parse_config_data(config_data)
    
    def _load_json_config(self, config_path: Path) -> TestConfig:
        """Load configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return self._parse_config_data(config_data)
    
    def _load_yaml_suite_config(self, config_path: Path) -> TestSuiteConfig:
        """Load test suite configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return self._parse_suite_config_data(config_data)
    
    def _load_json_suite_config(self, config_path: Path) -> TestSuiteConfig:
        """Load test suite configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return self._parse_suite_config_data(config_data)
    
    def _parse_config_data(self, config_data: Dict[str, Any]) -> TestConfig:
        """Parse configuration data into TestConfig object."""
        # Parse models
        models = []
        for model_data in config_data.get('models', []):
            model = ModelConfig(
                name=model_data['name'],
                model_type=ModelType(model_data['model_type']),
                parameters=ParameterConfig(**model_data.get('parameters', {})),
                description=model_data.get('description')
            )
            models.append(model)
        
        # Parse parameter variations
        parameter_variations = []
        for var_data in config_data.get('parameter_variations', []):
            variation = ParameterConfig(**var_data)
            parameter_variations.append(variation)
        
        return TestConfig(
            test_name=config_data['test_name'],
            prompts=config_data['prompts'],
            models=models,
            parameter_variations=parameter_variations,
            max_retries=config_data.get('max_retries', 3),
            request_timeout=config_data.get('request_timeout', 30),
            batch_size=config_data.get('batch_size', 5),
            output_dir=config_data.get('output_dir', './reports')
        )
    
    def _parse_suite_config_data(self, config_data: Dict[str, Any]) -> TestSuiteConfig:
        """Parse configuration data into TestSuiteConfig object."""
        tests = []
        for test_data in config_data.get('tests', []):
            test = self._parse_config_data(test_data)
            tests.append(test)
        
        return TestSuiteConfig(
            suite_name=config_data['suite_name'],
            description=config_data.get('description'),
            tests=tests,
            global_settings=config_data.get('global_settings', {})
        )
    
    def save_config(self, config: TestConfig, filepath: str, format: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            filepath: Path to save file
            format: File format ('yaml' or 'json')
        """
        config_path = Path(filepath)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.dict()
        
        if format.lower() == 'yaml':
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get configuration from environment variables.
        
        Returns:
            Dictionary of environment-based configuration
        """
        return {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'default_model_1': os.getenv('DEFAULT_MODEL_1', 'gpt-3.5-turbo'),
            'default_model_2': os.getenv('DEFAULT_MODEL_2', 'gpt-4'),
            'default_temperature_1': float(os.getenv('DEFAULT_TEMPERATURE_1', '0.0')),
            'default_temperature_2': float(os.getenv('DEFAULT_TEMPERATURE_2', '1.0')),
            'default_top_p_1': float(os.getenv('DEFAULT_TOP_P_1', '0.5')),
            'default_top_p_2': float(os.getenv('DEFAULT_TOP_P_2', '1.0')),
            'default_max_tokens_1': int(os.getenv('DEFAULT_MAX_TOKENS_1', '100')),
            'default_max_tokens_2': int(os.getenv('DEFAULT_MAX_TOKENS_2', '200')),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '30')),
            'batch_size': int(os.getenv('BATCH_SIZE', '5')),
            'output_dir': os.getenv('REPORT_OUTPUT_DIR', './reports'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'ci_mode': os.getenv('CI_MODE', 'false').lower() == 'true'
        }
