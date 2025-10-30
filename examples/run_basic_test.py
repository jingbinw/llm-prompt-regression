"""
Example script for running basic LLM prompt regression tests.
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import the 'src' package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.test_runner import TestRunner
from src.core.report_generator import ReportGenerator
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logging
from src.models.result_schemas import TestSuiteResult, TestStatus
from src.models.config_schemas import ParameterConfig


async def main():
    """Run a basic regression test example."""
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting basic regression test example")
    
    # Create config loader first to load environment variables
    config_loader = ConfigLoader()
    
    # Check for API key (after environment is loaded)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Create test configuration
        config = config_loader.create_default_config()
        
        # Customize for this example (low-token mode)
        config.test_name = "Multi-Model Regression Test with Parameter Variations"
        # Use a prompt that allows for variation but stays concise
        config.prompts = [
            "Explain 'regression testing' in one sentence."
        ]

        # Use only two models for simplicity
        config.models = config.models[:2]

        # Force small outputs and consistent settings to minimize tokens
        for m in config.models:
            m.parameters.max_tokens = min(getattr(m.parameters, "max_tokens", 200) or 200, 32)
            m.parameters.temperature = 0.2
            # Optionally tighten sampling
            m.parameters.top_p = 0.8

        # Minimal parameter variations: smaller range to reduce drift
        # Total calls = prompts (1) * models (2) * variations (2) = 4
        config.parameter_variations = [
            ParameterConfig(temperature=0.3, top_p=0.7, max_tokens=24, description="Low temp, focused"),
            ParameterConfig(temperature=0.7, top_p=0.9, max_tokens=24, description="Moderate temp, diverse"),
        ]
        # If you prefer full 2x2 orthogonal coverage (4 variations -> 8 calls), use below instead:
        # config.parameter_variations = [
        #     ParameterConfig(temperature=0.0, top_p=0.7, max_tokens=24, description="T0.0, p0.7"),
        #     ParameterConfig(temperature=1.0, top_p=0.7, max_tokens=24, description="T1.0, p0.7"),
        #     ParameterConfig(temperature=0.0, top_p=1.0, max_tokens=24, description="T0.0, p1.0"),
        #     ParameterConfig(temperature=1.0, top_p=1.0, max_tokens=24, description="T1.0, p1.0"),
        # ]
        
        logger.info(f"Running test: {config.test_name}")
        logger.info(f"Testing {len(config.prompts)} prompts with {len(config.models)} models")
        logger.info(f"Parameter variations: {len(config.parameter_variations)}")
        
        # Run the test
        runner = TestRunner(api_key, max_retries=3, timeout=30)
        test_result = await runner.run_test(config)
        
        logger.info(f"Test completed with status: {test_result.status}")
        logger.info(f"Duration: {test_result.duration:.2f} seconds")
        logger.info(f"Total comparisons: {test_result.total_comparisons}")
        
        if test_result.comparison_results:
            drift_count = sum(1 for c in test_result.comparison_results if c.drift_detected)
            logger.info(f"Drift detected in {drift_count} comparisons")
            
            # Show some example results
            for i, comparison in enumerate(test_result.comparison_results[:3]):  # Show first 3
                logger.info(f"Comparison {i+1}:")
                logger.info(f"  Prompt: {comparison.prompt[:50]}...")
                logger.info(f"  Models: {comparison.model_1_response.model_name} vs {comparison.model_2_response.model_name}")
                logger.info(f"  Drift detected: {comparison.drift_detected}")
                logger.info(f"  Severity: {comparison.drift_severity}")
                logger.info(f"  Similarity: {comparison.metrics.get('semantic_similarity', 'N/A'):.2f}")
        
        # Generate reports
        logger.info("Generating reports...")
        report_generator = ReportGenerator(output_dir="./reports")
        
        # Create test suite result
        suite_result = TestSuiteResult(
            suite_name="Basic Example Suite",
            start_time=test_result.start_time,
            end_time=test_result.end_time,
            test_results=[test_result],
            overall_status=test_result.status
        )
        
        # Generate drift report
        drift_report = report_generator.generate_drift_report(suite_result)
        logger.info(f"Generated drift report: {drift_report.report_id}")
        
        # Generate HTML report
        html_file = report_generator.generate_html_report(drift_report)
        logger.info(f"Generated HTML report: {html_file}")
        
        # Generate CSV report
        csv_file = report_generator.generate_csv_report(suite_result)
        logger.info(f"Generated CSV report: {csv_file}")
        
        # Generate visualizations
        charts = report_generator.generate_visualizations(drift_report)
        logger.info(f"Generated {len(charts)} visualization charts")
        
        # Print summary
        print("\n" + "="*60)
        print("BASIC REGRESSION TEST SUMMARY")
        print("="*60)
        print(f"Test Name: {test_result.test_name}")
        print(f"Status: {test_result.status}")
        print(f"Duration: {test_result.duration:.2f} seconds")
        print(f"Total Comparisons: {test_result.total_comparisons}")
        print(f"Drift Detected: {test_result.summary.get('drift_detected_count', 0)}")
        print(f"Drift Rate: {test_result.summary.get('drift_rate', 0):.1%}")
        print(f"Average Similarity: {test_result.summary.get('average_similarity', 0):.2f}")
        print("\nGenerated Reports:")
        print(f"  - HTML Report: {html_file}")
        print(f"  - CSV Report: {csv_file}")
        print(f"  - JSON Report: {drift_report.report_id}.json")
        print(f"  - Charts: {len(charts)} visualization(s)")
        
        # Show recommendations
        if drift_report.recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(drift_report.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
