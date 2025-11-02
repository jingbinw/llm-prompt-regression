# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-11-02

### Added
- **Complete LLM Regression Testing Framework**
  - Multi-model comparison (GPT-3.5-turbo vs GPT-4)
  - Parameter variation analysis (temperature, top_p, max_tokens)
  - Automated drift detection with severity classification
  - Comprehensive HTML reports with side-by-side response comparison
  - CSV and JSON export capabilities
  - Interactive visualizations (drift distribution, similarity heatmap, performance charts)

- **Low-Cost CI/CD Integration**
  - GitHub Actions workflow for automated testing
  - Optimized token usage (~320 tokens per CI run, <$0.01)
  - Artifact upload for reports and visualizations
  - Skip options for LLM tests during development

- **Developer Experience**
  - Example scripts with minimal token usage
  - Docker containerization support
  - Comprehensive test suite (20 tests, zero token cost)
  - Environment configuration with .env support
  - CLI interface for advanced usage

- **Production Ready Features**
  - Async API calls with retry logic and error handling
  - Configurable similarity metrics (TF-IDF default, OpenAI embeddings optional)
  - Data-driven recommendations based on drift analysis
  - Professional HTML reports with CSS styling
  - Extensible configuration via YAML/JSON files

### Technical Stack
- Python 3.8+ with async/await support
- OpenAI API integration with cost optimization
- LangChain (optional) for embeddings-based similarity
- pytest testing framework with full coverage
- Docker and GitHub Actions for CI/CD
- Visualization libraries: matplotlib, seaborn, plotly

### Documentation
- Comprehensive README with installation and usage guides
- Example configurations and use cases
- API documentation and best practices
- Cost analysis and optimization tips