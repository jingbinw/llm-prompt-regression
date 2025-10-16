# LLM Prompt Regression Testing Framework

A comprehensive framework for testing LLM output consistency across different model versions and parameter configurations. This framework helps detect drift and inconsistencies in LLM responses, ensuring reliable AI applications.

## 🎯 Features

- **Model Comparison**: Compare responses between different LLM models (GPT-3.5-turbo, GPT-4, etc.)
- **Parameter Variation Testing**: Test how different parameters (temperature, top_p, max_tokens) affect output consistency
- **Drift Detection**: Automated detection of response drift and inconsistencies
- **Comprehensive Reporting**: Generate detailed HTML, CSV, and JSON reports with visualizations
- **CI/CD Integration**: GitHub Actions workflow for automated regression testing
- **Docker Support**: Containerized testing environment
- **Flexible Configuration**: YAML/JSON configuration files for easy test customization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-prompt-regression.git
   cd llm-prompt-regression
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Run a basic test**
   ```bash
   python examples/run_basic_test.py
   ```

## 📖 Usage

### Command Line Interface

The framework provides a comprehensive CLI for running tests:

```bash
# Run default test configuration
python -m src.llm_prompt_regression.cli run

# Run specific configuration file
python -m src.llm_prompt_regression.cli run --config config/basic_test.yaml

# Run with custom parameters
python -m src.llm_prompt_regression.cli run \
  --model1 gpt-3.5-turbo \
  --model2 gpt-4 \
  --prompts "Explain AI" "What is machine learning?" \
  --temperature1 0.0 \
  --temperature2 1.0

# Run test suite
python -m src.llm_prompt_regression.cli run-suite \
  --config config/comprehensive_test_suite.yaml

# Generate reports from existing results
python -m src.llm_prompt_regression.cli report --input reports/results.json

# Create configuration files
python -m src.llm_prompt_regression.cli config --create-default
```

### Configuration Files

Create YAML or JSON configuration files to define your tests:

```yaml
# config/my_test.yaml
test_name: "My Custom Test"
prompts:
  - "Explain quantum computing in simple terms."
  - "Write a haiku about artificial intelligence."

models:
  - name: "gpt-3.5-turbo"
    model_type: "gpt-3.5-turbo"
    parameters:
      temperature: 0.7
      max_tokens: 200
  
  - name: "gpt-4"
    model_type: "gpt-4"
    parameters:
      temperature: 0.7
      max_tokens: 200

parameter_variations:
  - temperature: 0.0
    description: "Deterministic"
  - temperature: 1.0
    description: "Creative"

max_retries: 3
request_timeout: 30
batch_size: 5
output_dir: "./reports"
```

### Programmatic Usage

```python
import asyncio
from llm_prompt_regression import TestRunner, ConfigLoader, ReportGenerator

async def run_custom_test():
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.create_default_config()
    
    # Customize configuration
    config.prompts = ["Your custom prompt here"]
    
    # Run tests
    runner = TestRunner(api_key="your-api-key")
    result = await runner.run_test(config)
    
    # Generate reports
    report_generator = ReportGenerator()
    report = report_generator.generate_drift_report(result)
    
    return report

# Run the test
asyncio.run(run_custom_test())
```

## 🐳 Docker Usage

### Build and Run

```bash
# Build the Docker image
docker build -t llm-prompt-regression .

# Run tests in Docker
docker run --rm \
  -e OPENAI_API_KEY="your-api-key" \
  -v $(pwd)/reports:/app/reports \
  llm-prompt-regression

# Using Docker Compose
docker-compose up llm-prompt-regression
```

### Docker Compose Services

```bash
# Run tests
docker-compose up llm-prompt-regression

# Run specific test suite
docker-compose run test-runner

# Generate reports
docker-compose run report-generator
```

## 🔧 Configuration Options

### Model Configuration

```yaml
models:
  - name: "gpt-3.5-turbo"
    model_type: "gpt-3.5-turbo"
    parameters:
      temperature: 0.7      # 0.0 to 2.0
      top_p: 1.0           # 0.0 to 1.0
      max_tokens: 200      # > 0
      frequency_penalty: 0.0  # -2.0 to 2.0
      presence_penalty: 0.0   # -2.0 to 2.0
    description: "Model description"
```

### Parameter Variations

Test how different parameters affect output consistency:

```yaml
parameter_variations:
  # Temperature variations
  - temperature: 0.0
    description: "Deterministic"
  - temperature: 1.0
    description: "Creative"
  
  # Top-p variations
  - top_p: 0.5
    description: "Focused sampling"
  - top_p: 1.0
    description: "Maximum diversity"
  
  # Max tokens variations
  - max_tokens: 100
    description: "Short response"
  - max_tokens: 500
    description: "Long response"
  
  # Combined variations
  - temperature: 0.0
    top_p: 0.5
    max_tokens: 100
    description: "Conservative + Focused + Short"
```

## 📊 Reports and Metrics

The framework generates comprehensive reports including:

### Drift Metrics
- **Exact Match**: Whether responses are identical
- **Semantic Similarity**: TF-IDF based similarity score
- **Token Count Difference**: Length variation between responses
- **Response Time Difference**: Performance comparison
- **Coherence Score**: Response quality assessment

### Report Formats
- **HTML Report**: Interactive dashboard with visualizations
- **CSV Report**: Detailed data for analysis
- **JSON Report**: Machine-readable results
- **Charts**: Visual representations of drift patterns

### Example Report Structure
```
reports/
├── report_20240101_120000.json      # Detailed results
├── report_20240101_120000.html      # HTML dashboard
├── detailed_results_20240101_120000.csv  # CSV data
├── report_20240101_120000_drift_distribution.png
├── report_20240101_120000_similarity_heatmap.png
├── report_20240101_120000_token_comparison.png
└── report_20240101_120000_response_time.png
```

## 🔄 CI/CD Integration

### GitHub Actions

The framework includes a comprehensive CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline
on:
  push:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
```

**Pipeline Features:**
- Code quality checks (linting, formatting, type checking)
- Unit and integration tests
- LLM regression testing
- Security scanning
- Docker image building
- Automated reporting

### Setting up GitHub Actions

1. **Add secrets to your repository:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `DOCKER_USERNAME`: Docker Hub username (optional)
   - `DOCKER_PASSWORD`: Docker Hub password (optional)

2. **Configure workflow triggers:**
   - Push to main/develop branches
   - Pull requests
   - Scheduled daily runs
   - Manual triggers

3. **Monitor results:**
   - Check GitHub Actions tab for test results
   - Review generated reports in artifacts
   - Set up notifications for failures

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=src/llm_prompt_regression --cov-report=html
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Full workflow testing
- **Mock Tests**: API-free testing with mocked responses

## 📁 Project Structure

```
llm-prompt-regression/
├── src/llm_prompt_regression/          # Main source code
│   ├── core/                          # Core testing logic
│   │   ├── test_runner.py             # Main test runner
│   │   └── report_generator.py        # Report generation
│   ├── models/                        # Data models
│   │   ├── test_config.py             # Configuration models
│   │   └── test_result.py             # Result models
│   ├── utils/                         # Utility functions
│   │   ├── metrics.py                 # Metrics calculation
│   │   ├── validators.py              # Response validation
│   │   ├── config_loader.py           # Configuration loading
│   │   └── logger_setup.py            # Logging setup
│   ├── cli.py                         # Command-line interface
│   └── __init__.py
├── tests/                             # Test files
│   ├── unit/                          # Unit tests
│   └── integration/                   # Integration tests
├── config/                            # Configuration files
│   ├── basic_test.yaml                # Basic test configuration
│   ├── parameter_variation_test.yaml  # Parameter variation tests
│   └── comprehensive_test_suite.yaml  # Full test suite
├── examples/                          # Example scripts
│   ├── run_basic_test.py              # Basic test example
│   └── run_parameter_variation_test.py # Parameter test example
├── .github/workflows/                 # GitHub Actions
│   ├── ci.yml                         # CI/CD pipeline
│   └── release.yml                    # Release workflow
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Docker Compose setup
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
└── README.md                          # This file
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests**: `pytest`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/llm-prompt-regression.git
cd llm-prompt-regression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- The open-source community for various Python packages
- Contributors and testers who helped improve this framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llm-prompt-regression/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llm-prompt-regression/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/llm-prompt-regression/wiki)

## 🔮 Roadmap

- [ ] Support for more LLM providers (Anthropic, Cohere, etc.)
- [ ] Advanced drift detection algorithms
- [ ] Real-time monitoring dashboard
- [ ] Integration with popular ML frameworks
- [ ] Performance optimization for large-scale testing
- [ ] Advanced visualization options
- [ ] Plugin system for custom metrics

---

**Made with ❤️ for the AI community**
