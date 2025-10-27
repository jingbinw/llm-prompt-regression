"""
Example script for running parameter variation tests.
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


async def main():
    """Run parameter variation test example."""
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting parameter variation test example")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Create parameter variation configuration
        config_loader = ConfigLoader()
        config = config_loader.create_parameter_variation_config()
        
        # Customize for this example (low-token mode)
        config.test_name = "Parameter Variation Example (Low Token)"
        # Use single, shorter prompt to minimize token usage
        config.prompts = [
            "What is 5+3? Answer with just the number."
        ]
        
        # Limit parameter variations and cap max_tokens (to reduce API costs)
        config.parameter_variations = config.parameter_variations[:4]  # First 4 variations only
        
        # Force all variations to use small max_tokens
        for variation in config.parameter_variations:
            variation.max_tokens = min(variation.max_tokens or 50, 30)
        
        logger.info(f"Running test: {config.test_name}")
        logger.info(f"Testing {len(config.prompts)} prompts with {len(config.models)} model(s)")
        logger.info(f"Parameter variations: {len(config.parameter_variations)}")
        
        # Show parameter variations
        logger.info("Parameter variations to test:")
        for i, variation in enumerate(config.parameter_variations, 1):
            params = []
            if variation.temperature is not None:
                params.append(f"temp={variation.temperature}")
            if variation.top_p is not None:
                params.append(f"top_p={variation.top_p}")
            if variation.max_tokens is not None:
                params.append(f"max_tokens={variation.max_tokens}")
            
            logger.info(f"  {i}. {', '.join(params)} - {variation.description}")
        
        # Run the test
        runner = TestRunner(api_key, max_retries=3, timeout=30)
        test_result = await runner.run_test(config)
        
        logger.info(f"Test completed with status: {test_result.status}")
        logger.info(f"Duration: {test_result.duration:.2f} seconds")
        logger.info(f"Total comparisons: {test_result.total_comparisons}")
        
        # Analyze parameter effects
        if test_result.comparison_results:
            logger.info("\nAnalyzing parameter effects...")
            
            # Group results by parameter type
            temp_results = {}
            top_p_results = {}
            max_tokens_results = {}
            
            for comparison in test_result.comparison_results:
                params1 = comparison.model_1_response.parameters
                params2 = comparison.model_2_response.parameters
                
                # Check temperature variations
                if params1.get('temperature') != params2.get('temperature'):
                    temp_diff = abs(params1.get('temperature', 0.7) - params2.get('temperature', 0.7))
                    if temp_diff not in temp_results:
                        temp_results[temp_diff] = []
                    temp_results[temp_diff].append(comparison)
                
                # Check top_p variations
                if params1.get('top_p') != params2.get('top_p'):
                    top_p_diff = abs(params1.get('top_p', 1.0) - params2.get('top_p', 1.0))
                    if top_p_diff not in top_p_results:
                        top_p_results[top_p_diff] = []
                    top_p_results[top_p_diff].append(comparison)
                
                # Check max_tokens variations
                if params1.get('max_tokens') != params2.get('max_tokens'):
                    tokens_diff = abs(params1.get('max_tokens', 200) - params2.get('max_tokens', 200))
                    if tokens_diff not in max_tokens_results:
                        max_tokens_results[tokens_diff] = []
                    max_tokens_results[tokens_diff].append(comparison)
            
            # Report parameter effects
            if temp_results:
                logger.info("\nTemperature Effects:")
                for temp_diff, comparisons in temp_results.items():
                    drift_count = sum(1 for c in comparisons if c.drift_detected)
                    avg_similarity = sum(c.metrics.get('semantic_similarity', 1.0) for c in comparisons) / len(comparisons)
                    logger.info(f"  Temperature difference {temp_diff}: {drift_count}/{len(comparisons)} drift, avg similarity: {avg_similarity:.2f}")
            
            if top_p_results:
                logger.info("\nTop-p Effects:")
                for top_p_diff, comparisons in top_p_results.items():
                    drift_count = sum(1 for c in comparisons if c.drift_detected)
                    avg_similarity = sum(c.metrics.get('semantic_similarity', 1.0) for c in comparisons) / len(comparisons)
                    logger.info(f"  Top-p difference {top_p_diff}: {drift_count}/{len(comparisons)} drift, avg similarity: {avg_similarity:.2f}")
            
            if max_tokens_results:
                logger.info("\nMax Tokens Effects:")
                for tokens_diff, comparisons in max_tokens_results.items():
                    drift_count = sum(1 for c in comparisons if c.drift_detected)
                    avg_similarity = sum(c.metrics.get('semantic_similarity', 1.0) for c in comparisons) / len(comparisons)
                    logger.info(f"  Max tokens difference {tokens_diff}: {drift_count}/{len(comparisons)} drift, avg similarity: {avg_similarity:.2f}")
        
        # Generate reports
        logger.info("\nGenerating reports...")
        report_generator = ReportGenerator(output_dir="./reports")
        
        # Create test suite result
        suite_result = TestSuiteResult(
            suite_name="Parameter Variation Example Suite",
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
        print("PARAMETER VARIATION TEST SUMMARY")
        print("="*60)
        print(f"Test Name: {test_result.test_name}")
        print(f"Status: {test_result.status}")
        print(f"Duration: {test_result.duration:.2f} seconds")
        print(f"Total Comparisons: {test_result.total_comparisons}")
        print(f"Drift Detected: {test_result.summary.get('drift_detected_count', 0)}")
        print(f"Drift Rate: {test_result.summary.get('drift_rate', 0):.1%}")
        print(f"Average Similarity: {test_result.summary.get('average_similarity', 0):.2f}")
        
        # Show parameter-specific insights
        if temp_results:
            print(f"\nTemperature Variations Tested: {len(temp_results)}")
        if top_p_results:
            print(f"Top-p Variations Tested: {len(top_p_results)}")
        if max_tokens_results:
            print(f"Max Tokens Variations Tested: {len(max_tokens_results)}")
        
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
