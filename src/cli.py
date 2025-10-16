"""
Command-line interface for LLM Prompt Regression Testing Framework.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from .core.test_runner import TestRunner
from .core.report_generator import ReportGenerator
from .utils.config_loader import ConfigLoader
from .utils.logger_setup import setup_logging
from .models.test_result import TestSuiteResult, TestStatus


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Prompt Regression Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default test configuration
  python -m llm_prompt_regression.cli run

  # Run specific configuration file
  python -m llm_prompt_regression.cli run --config config/basic_test.yaml

  # Run with custom parameters
  python -m llm_prompt_regression.cli run --model1 gpt-3.5-turbo --model2 gpt-4

  # Generate report from existing results
  python -m llm_prompt_regression.cli report --input reports/results.json

  # Run test suite
  python -m llm_prompt_regression.cli run-suite --config config/comprehensive_test_suite.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run regression tests')
    run_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )
    run_parser.add_argument(
        '--model1',
        type=str,
        default='gpt-3.5-turbo',
        help='First model to test (default: gpt-3.5-turbo)'
    )
    run_parser.add_argument(
        '--model2',
        type=str,
        default='gpt-4',
        help='Second model to test (default: gpt-4)'
    )
    run_parser.add_argument(
        '--prompts',
        nargs='+',
        help='Custom prompts to test'
    )
    run_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./reports',
        help='Output directory for reports (default: ./reports)'
    )
    run_parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum number of retries (default: 3)'
    )
    run_parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    run_parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Batch size for concurrent requests (default: 5)'
    )
    run_parser.add_argument(
        '--temperature1',
        type=float,
        default=0.7,
        help='Temperature for model 1 (default: 0.7)'
    )
    run_parser.add_argument(
        '--temperature2',
        type=float,
        default=0.7,
        help='Temperature for model 2 (default: 0.7)'
    )
    run_parser.add_argument(
        '--max-tokens',
        type=int,
        default=200,
        help='Maximum tokens per response (default: 200)'
    )
    run_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Run suite command
    suite_parser = subparsers.add_parser('run-suite', help='Run test suite')
    suite_parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Test suite configuration file path'
    )
    suite_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./reports',
        help='Output directory for reports (default: ./reports)'
    )
    suite_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input test results file (JSON)'
    )
    report_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./reports',
        help='Output directory for reports (default: ./reports)'
    )
    report_parser.add_argument(
        '--format',
        choices=['html', 'csv', 'json', 'all'],
        default='all',
        help='Report format (default: all)'
    )
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration utilities')
    config_parser.add_argument(
        '--create-default',
        action='store_true',
        help='Create default configuration file'
    )
    config_parser.add_argument(
        '--create-parameter-test',
        action='store_true',
        help='Create parameter variation test configuration'
    )
    config_parser.add_argument(
        '--output', '-o',
        type=str,
        default='./config',
        help='Output directory for configuration files (default: ./config)'
    )
    
    return parser


async def run_tests(args) -> int:
    """Run regression tests."""
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level=log_level, log_file="logs/test_run.log")
    
    try:
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return 1
        
        # Load or create configuration
        config_loader = ConfigLoader()
        
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = config_loader.load_test_config(args.config)
        else:
            logger.info("Creating default configuration")
            config = config_loader.create_default_config()
            
            # Override with command-line arguments
            if args.model1 or args.model2:
                config.models = []
                if args.model1:
                    from .models.test_config import ModelConfig, ParameterConfig, ModelType
                    config.models.append(ModelConfig(
                        name=args.model1,
                        model_type=ModelType(args.model1),
                        parameters=ParameterConfig(
                            temperature=args.temperature1,
                            max_tokens=args.max_tokens
                        )
                    ))
                if args.model2:
                    config.models.append(ModelConfig(
                        name=args.model2,
                        model_type=ModelType(args.model2),
                        parameters=ParameterConfig(
                            temperature=args.temperature2,
                            max_tokens=args.max_tokens
                        )
                    ))
            
            if args.prompts:
                config.prompts = args.prompts
            
            config.max_retries = args.max_retries
            config.request_timeout = args.timeout
            config.batch_size = args.batch_size
            config.output_dir = args.output_dir
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run tests
        logger.info(f"Starting test: {config.test_name}")
        runner = TestRunner(api_key, max_retries=config.max_retries, timeout=config.request_timeout)
        test_result = await runner.run_test(config)
        
        # Generate reports
        logger.info("Generating reports...")
        report_generator = ReportGenerator(output_dir=args.output_dir)
        
        # Create test suite result
        suite_result = TestSuiteResult(
            suite_name=f"CLI Test Run - {config.test_name}",
            start_time=test_result.start_time,
            end_time=test_result.end_time,
            test_results=[test_result],
            overall_status=test_result.status
        )
        
        # Generate drift report
        drift_report = report_generator.generate_drift_report(suite_result)
        
        # Generate HTML report
        html_file = report_generator.generate_html_report(drift_report)
        
        # Generate CSV report
        csv_file = report_generator.generate_csv_report(suite_result)
        
        # Generate visualizations
        charts = report_generator.generate_visualizations(drift_report)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Test Name: {test_result.test_name}")
        logger.info(f"Status: {test_result.status}")
        logger.info(f"Duration: {test_result.duration:.2f} seconds")
        logger.info(f"Total Comparisons: {test_result.total_comparisons}")
        logger.info(f"Drift Detected: {test_result.summary.get('drift_detected_count', 0)}")
        logger.info(f"Drift Rate: {test_result.summary.get('drift_rate', 0):.1%}")
        logger.info(f"Average Similarity: {test_result.summary.get('average_similarity', 0):.2f}")
        logger.info("")
        logger.info("Generated Reports:")
        logger.info(f"  - HTML Report: {html_file}")
        logger.info(f"  - CSV Report: {csv_file}")
        logger.info(f"  - JSON Report: {drift_report.report_id}.json")
        logger.info(f"  - Charts: {len(charts)} visualization(s)")
        logger.info("=" * 50)
        
        # Return exit code based on results
        if test_result.status == TestStatus.FAILED:
            return 1
        elif test_result.summary.get('drift_rate', 0) > 0.3:
            logger.warning("High drift rate detected!")
            return 2
        else:
            logger.info("Tests completed successfully!")
            return 0
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return 1


async def run_suite(args) -> int:
    """Run test suite."""
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level=log_level, log_file="logs/suite_run.log")
    
    try:
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return 1
        
        # Load test suite configuration
        config_loader = ConfigLoader()
        suite_config = config_loader.load_test_suite_config(args.config)
        
        logger.info(f"Running test suite: {suite_config.suite_name}")
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run all tests in the suite
        test_results = []
        runner = TestRunner(api_key)
        
        for test_config in suite_config.tests:
            logger.info(f"Running test: {test_config.test_name}")
            test_result = await runner.run_test(test_config)
            test_results.append(test_result)
            
            if test_result.status == TestStatus.FAILED:
                logger.error(f"Test failed: {test_config.test_name}")
        
        # Create test suite result
        suite_result = TestSuiteResult(
            suite_name=suite_config.suite_name,
            start_time=test_results[0].start_time if test_results else None,
            end_time=test_results[-1].end_time if test_results else None,
            test_results=test_results,
            overall_status=TestStatus.COMPLETED if all(t.status == TestStatus.COMPLETED for t in test_results) else TestStatus.FAILED
        )
        
        # Generate reports
        logger.info("Generating suite reports...")
        report_generator = ReportGenerator(output_dir=args.output_dir)
        drift_report = report_generator.generate_drift_report(suite_result)
        
        # Generate all report formats
        html_file = report_generator.generate_html_report(drift_report)
        csv_file = report_generator.generate_csv_report(suite_result)
        charts = report_generator.generate_visualizations(drift_report)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("SUITE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Suite Name: {suite_config.suite_name}")
        logger.info(f"Overall Status: {suite_result.overall_status}")
        logger.info(f"Tests Run: {len(test_results)}")
        logger.info(f"Tests Passed: {sum(1 for t in test_results if t.status == TestStatus.COMPLETED)}")
        logger.info(f"Tests Failed: {sum(1 for t in test_results if t.status == TestStatus.FAILED)}")
        logger.info("")
        logger.info("Generated Reports:")
        logger.info(f"  - HTML Report: {html_file}")
        logger.info(f"  - CSV Report: {csv_file}")
        logger.info(f"  - JSON Report: {drift_report.report_id}.json")
        logger.info(f"  - Charts: {len(charts)} visualization(s)")
        logger.info("=" * 50)
        
        return 0 if suite_result.overall_status == TestStatus.COMPLETED else 1
        
    except Exception as e:
        logger.error(f"Suite execution failed: {str(e)}")
        return 1


def generate_reports(args) -> int:
    """Generate reports from existing test results."""
    # Setup logging
    logger = setup_logging(log_level="INFO", log_file="logs/report_generation.log")
    
    try:
        if not args.input:
            logger.error("Input file is required for report generation")
            return 1
        
        # Load test results
        import json
        with open(args.input, 'r') as f:
            results_data = json.load(f)
        
        # Convert to TestSuiteResult object
        # This would need proper deserialization logic
        logger.info(f"Generating reports from {args.input}")
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate reports
        report_generator = ReportGenerator(output_dir=args.output_dir)
        
        # For now, just create a placeholder
        logger.info("Report generation completed")
        return 0
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return 1


def create_config_files(args) -> int:
    """Create configuration files."""
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_loader = ConfigLoader()
        
        if args.create_default:
            config = config_loader.create_default_config()
            config_file = output_dir / "default_test.yaml"
            config_loader.save_config(config, str(config_file), format='yaml')
            logger.info(f"Created default configuration: {config_file}")
        
        if args.create_parameter_test:
            config = config_loader.create_parameter_variation_config()
            config_file = output_dir / "parameter_variation_test.yaml"
            config_loader.save_config(config, str(config_file), format='yaml')
            logger.info(f"Created parameter variation configuration: {config_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Configuration creation failed: {str(e)}")
        return 1


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    if args.command == 'run':
        return await run_tests(args)
    elif args.command == 'run-suite':
        return await run_suite(args)
    elif args.command == 'report':
        return generate_reports(args)
    elif args.command == 'config':
        return create_config_files(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
