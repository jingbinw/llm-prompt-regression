"""
Report generation utilities for LLM prompt regression testing.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from ..models.result_schemas import TestSuiteResult, DriftReport, ComparisonResult
from ..utils.logger_setup import get_logger


logger = get_logger(__name__)


class ReportGenerator:
    """Generator for comprehensive test reports."""
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_drift_report(self, test_suite_result: TestSuiteResult) -> DriftReport:
        """
        Generate a comprehensive drift analysis report.
        
        Args:
            test_suite_result: Results from test suite execution
            
        Returns:
            Generated drift report
        """
        logger.info("Generating drift analysis report...")
        
        # Collect all comparison results
        all_comparisons = []
        for test_result in test_suite_result.test_results:
            all_comparisons.extend(test_result.comparison_results)
        
        # Generate drift summary
        drift_summary = self._generate_drift_summary(all_comparisons)
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(all_comparisons)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(drift_summary, detailed_analysis)
        
        # Generate charts data
        charts_data = self._generate_charts_data(all_comparisons)
        
        # Create drift report
        report = DriftReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_suite_result=test_suite_result,
            drift_summary=drift_summary,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            charts_data=charts_data
        )
        
        # Save report
        self._save_drift_report(report)
        
        logger.info(f"Drift report generated: {report.report_id}")
        return report
    
    def generate_html_report(self, drift_report: DriftReport) -> str:
        """
        Generate HTML report from drift report.
        
        Args:
            drift_report: Drift report to convert
            
        Returns:
            Path to generated HTML report
        """
        logger.info("Generating HTML report...")
        
        html_content = self._generate_html_content(drift_report)
        
        html_file = self.output_dir / f"{drift_report.report_id}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved: {html_file}")
        return str(html_file)
    
    def generate_csv_report(self, test_suite_result: TestSuiteResult) -> str:
        """
        Generate CSV report with detailed results.
        
        Args:
            test_suite_result: Test suite result
            
        Returns:
            Path to generated CSV report
        """
        logger.info("Generating CSV report...")
        
        # Collect all data
        rows = []
        for test_result in test_suite_result.test_results:
            for comparison in test_result.comparison_results:
                row = {
                    'test_name': test_result.test_name,
                    'prompt': comparison.prompt,
                    'model_1': comparison.model_1_response.model_name,
                    'model_2': comparison.model_2_response.model_name,
                    'model_1_response': comparison.model_1_response.response,
                    'model_2_response': comparison.model_2_response.response,
                    'model_1_tokens': comparison.model_1_response.token_count,
                    'model_2_tokens': comparison.model_2_response.token_count,
                    'exact_match': comparison.metrics.get('exact_match', False),
                    'semantic_similarity': comparison.metrics.get('semantic_similarity', 0.0),
                    'token_count_diff': comparison.metrics.get('token_count', 0),
                    'response_time_diff': comparison.metrics.get('response_time', 0.0),
                    'drift_detected': comparison.drift_detected,
                    'drift_severity': comparison.drift_severity,
                    'drift_explanation': comparison.drift_explanation,
                    'timestamp': comparison.timestamp
                }
                rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        csv_file = self.output_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"CSV report saved: {csv_file}")
        return str(csv_file)
    
    def generate_visualizations(self, drift_report: DriftReport) -> Dict[str, str]:
        """
        Generate visualization charts.
        
        Args:
            drift_report: Drift report containing data
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        logger.info("Generating visualizations...")
        
        charts = {}
        
        # Drift distribution chart
        charts['drift_distribution'] = self._create_drift_distribution_chart(drift_report)
        
        # Similarity heatmap
        charts['similarity_heatmap'] = self._create_similarity_heatmap(drift_report)
        
        # Token count comparison
        charts['token_comparison'] = self._create_token_comparison_chart(drift_report)
        
        # Response time analysis
        charts['response_time_analysis'] = self._create_response_time_chart(drift_report)
        
        # Model performance comparison
        charts['model_performance'] = self._create_model_performance_chart(drift_report)
        
        logger.info(f"Generated {len(charts)} visualizations")
        return charts
    
    def _generate_drift_summary(self, comparisons: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate summary of drift findings."""
        total_comparisons = len(comparisons)
        drift_detected = sum(1 for c in comparisons if c.drift_detected)
        
        # Severity distribution
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        for comparison in comparisons:
            if comparison.drift_detected:
                severity_counts[comparison.drift_severity] += 1
        
        # Model-specific drift rates
        model_drift_rates = {}
        for comparison in comparisons:
            model_pair = f"{comparison.model_1_response.model_name} vs {comparison.model_2_response.model_name}"
            if model_pair not in model_drift_rates:
                model_drift_rates[model_pair] = {"total": 0, "drift": 0}
            
            model_drift_rates[model_pair]["total"] += 1
            if comparison.drift_detected:
                model_drift_rates[model_pair]["drift"] += 1
        
        # Calculate drift rates
        for model_pair in model_drift_rates:
            total = model_drift_rates[model_pair]["total"]
            drift = model_drift_rates[model_pair]["drift"]
            model_drift_rates[model_pair]["rate"] = drift / total if total > 0 else 0
        
        # Average similarity
        similarities = [
            c.metrics.get('semantic_similarity', 1.0) 
            for c in comparisons 
            if 'semantic_similarity' in c.metrics
        ]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        
        return {
            "total_comparisons": total_comparisons,
            "drift_detected_count": drift_detected,
            "drift_rate": drift_detected / total_comparisons if total_comparisons > 0 else 0,
            "severity_distribution": severity_counts,
            "model_drift_rates": model_drift_rates,
            "average_similarity": avg_similarity,
            "high_severity_rate": severity_counts["high"] / total_comparisons if total_comparisons > 0 else 0
        }
    
    def _generate_detailed_analysis(self, comparisons: List[ComparisonResult]) -> List[Dict[str, Any]]:
        """Generate detailed drift analysis."""
        analysis = []
        
        # Group by prompt
        prompt_groups = {}
        for comparison in comparisons:
            prompt = comparison.prompt
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(comparison)
        
        # Analyze each prompt
        for prompt, prompt_comparisons in prompt_groups.items():
            prompt_analysis = {
                "prompt": prompt,
                "total_comparisons": len(prompt_comparisons),
                "drift_count": sum(1 for c in prompt_comparisons if c.drift_detected),
                "avg_similarity": sum(
                    c.metrics.get('semantic_similarity', 1.0) 
                    for c in prompt_comparisons 
                    if 'semantic_similarity' in c.metrics
                ) / len(prompt_comparisons) if prompt_comparisons else 1.0,
                "severity_breakdown": {"low": 0, "medium": 0, "high": 0},
                "common_issues": [],
                "comparisons": []  # Add detailed comparison info
            }
            
            # Count severities and collect comparison details
            for comparison in prompt_comparisons:
                if comparison.drift_detected:
                    prompt_analysis["severity_breakdown"][comparison.drift_severity] += 1
                
                # Add comparison details with parameters
                prompt_analysis["comparisons"].append({
                    "model_1": comparison.model_1_response.model_name,
                    "model_2": comparison.model_2_response.model_name,
                    "params_1": comparison.model_1_response.parameters,
                    "params_2": comparison.model_2_response.parameters,
                    "drift": comparison.drift_detected,
                    "severity": comparison.drift_severity if comparison.drift_detected else "N/A",
                    "similarity": comparison.metrics.get('semantic_similarity', 'N/A'),
                    "explanation": comparison.drift_explanation
                })
            
            # Identify common issues
            explanations = [c.drift_explanation for c in prompt_comparisons if c.drift_detected]
            issue_counts = {}
            for explanation in explanations:
                # Extract key issues from explanations
                if "semantic similarity" in explanation.lower():
                    issue_counts["Low semantic similarity"] = issue_counts.get("Low semantic similarity", 0) + 1
                if "token count" in explanation.lower():
                    issue_counts["Token count differences"] = issue_counts.get("Token count differences", 0) + 1
                if "response time" in explanation.lower():
                    issue_counts["Response time differences"] = issue_counts.get("Response time differences", 0) + 1
            
            prompt_analysis["common_issues"] = [
                {"issue": issue, "count": count} 
                for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            
            analysis.append(prompt_analysis)
        
        return analysis
    
    def _generate_recommendations(self, drift_summary: Dict[str, Any], detailed_analysis: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        overall_drift_rate = drift_summary.get("drift_rate", 0)
        high_severity_rate = drift_summary.get("high_severity_rate", 0)
        avg_similarity = drift_summary.get("average_similarity", 1.0)
        
        # Overall recommendations
        if overall_drift_rate > 0.3:
            recommendations.append(
                f"High overall drift rate detected ({overall_drift_rate:.1%}). "
                "Consider implementing stricter consistency checks and model fine-tuning."
            )
        
        if high_severity_rate > 0.1:
            recommendations.append(
                f"Significant high-severity drift detected ({high_severity_rate:.1%}). "
                "Immediate attention required for model consistency."
            )
        
        if avg_similarity < 0.8:
            recommendations.append(
                f"Low average semantic similarity ({avg_similarity:.2f}). "
                "Consider adjusting model parameters or implementing semantic consistency checks."
            )
        
        # Model-specific recommendations
        model_drift_rates = drift_summary.get("model_drift_rates", {})
        for model_pair, stats in model_drift_rates.items():
            if stats.get("rate", 0) > 0.5:
                recommendations.append(
                    f"High drift rate between {model_pair} ({stats['rate']:.1%}). "
                    f"Consider separate testing strategies for these models."
                )
        
        # Prompt-specific recommendations
        problematic_prompts = [
            analysis for analysis in detailed_analysis 
            if analysis["drift_count"] / analysis["total_comparisons"] > 0.5
        ]
        
        if problematic_prompts:
            recommendations.append(
                f"Found {len(problematic_prompts)} prompts with high drift rates. "
                "Review and potentially revise these prompts for better consistency."
            )
        
        # General recommendation if no specific issues found
        if not recommendations:
            recommendations.append(
                "No significant drift detected. Model outputs are consistent across test cases. "
                "Continue monitoring for future changes."
            )
        
        return recommendations
    
    def _generate_charts_data(self, comparisons: List[ComparisonResult]) -> Dict[str, Any]:
        """Generate data for charts."""
        return {
            "similarity_scores": [
                c.metrics.get('semantic_similarity', 1.0) 
                for c in comparisons 
                if 'semantic_similarity' in c.metrics
            ],
            "token_differences": [
                c.metrics.get('token_count', 0) 
                for c in comparisons 
                if 'token_count' in c.metrics
            ],
            "response_times_1": [c.model_1_response.response_time for c in comparisons],
            "response_times_2": [c.model_2_response.response_time for c in comparisons],
            "drift_severities": [c.drift_severity for c in comparisons if c.drift_detected],
            "model_pairs": [
                f"{c.model_1_response.model_name} vs {c.model_2_response.model_name}" 
                for c in comparisons
            ]
        }
    
    def _save_drift_report(self, report: DriftReport):
        """Save drift report to file."""
        report_file = self.output_dir / f"{report.report_id}.json"
        
        # Convert to dict and handle datetime serialization
        report_dict = report.model_dump(mode='json')
        report_dict['generated_at'] = report_dict['generated_at'].isoformat() if isinstance(report_dict['generated_at'], datetime) else report_dict['generated_at']
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
    
    def _generate_html_content(self, drift_report: DriftReport) -> str:
        """Generate HTML content for the report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Prompt Drift Report - {report_id}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .test-name {{ font-size: 1.2em; color: #1976d2; margin-top: 10px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
                .drift-high {{ color: #d32f2f; }}
                .drift-medium {{ color: #f57c00; }}
                .drift-low {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LLM Prompt Drift Report</h1>
                <p class="test-name"><strong>Test Name:</strong> {test_name}</p>
                <p><strong>Report ID:</strong> {report_id}</p>
                <p><strong>Generated:</strong> {generated_at}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Total Comparisons:</strong> {total_comparisons}
                </div>
                <div class="metric">
                    <strong>Drift Detected:</strong> {drift_count}
                </div>
                <div class="metric">
                    <strong>Drift Rate:</strong> {drift_rate:.1%}
                </div>
                <div class="metric">
                    <strong>Average Similarity:</strong> {avg_similarity:.2f}
                </div>
            </div>
            
            <div class="section">
                <h2>Severity Distribution</h2>
                <div class="metric drift-high">
                    <strong>High Severity:</strong> {high_count}
                </div>
                <div class="metric drift-medium">
                    <strong>Medium Severity:</strong> {medium_count}
                </div>
                <div class="metric drift-low">
                    <strong>Low Severity:</strong> {low_count}
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {recommendations_html}
                </ul>
            </div>
            
            <div class="section">
                <h2>Detailed Analysis</h2>
                {detailed_analysis_html}
            </div>
        </body>
        </html>
        """
        
        # Prepare data
        summary = drift_report.drift_summary
        recommendations_html = "".join(f"<li>{rec}</li>" for rec in drift_report.recommendations)
        
        # Generate detailed analysis HTML
        detailed_analysis_html = ""
        for analysis in drift_report.detailed_analysis:
            prompt_text = analysis['prompt']
            # Show full prompt but make it clear this is the prompt being analyzed
            detailed_analysis_html += f"""
            <h3>Prompt: "{prompt_text}"</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Comparisons</td><td>{analysis['total_comparisons']}</td></tr>
                <tr><td>Drift Count</td><td>{analysis['drift_count']}</td></tr>
                <tr><td>Average Similarity</td><td>{analysis['avg_similarity']:.2f}</td></tr>
            </table>
            <h4>Comparison Details</h4>
            <table>
                <tr>
                    <th>Model 1</th>
                    <th>Temp 1</th>
                    <th>Top-p 1</th>
                    <th>Model 2</th>
                    <th>Temp 2</th>
                    <th>Top-p 2</th>
                    <th>Similarity</th>
                    <th>Drift</th>
                    <th>Severity</th>
                </tr>
            """
            
            # Add rows for each comparison
            for comp in analysis.get('comparisons', []):
                temp1 = comp['params_1'].get('temperature', 'N/A')
                temp2 = comp['params_2'].get('temperature', 'N/A')
                topp1 = comp['params_1'].get('top_p', 'N/A')
                topp2 = comp['params_2'].get('top_p', 'N/A')
                sim = comp['similarity']
                sim_str = f"{sim:.2f}" if isinstance(sim, (int, float)) else str(sim)
                drift_class = f"drift-{comp['severity']}" if comp['drift'] else ""
                
                detailed_analysis_html += f"""
                <tr class="{drift_class}">
                    <td>{comp['model_1']}</td>
                    <td>{temp1}</td>
                    <td>{topp1}</td>
                    <td>{comp['model_2']}</td>
                    <td>{temp2}</td>
                    <td>{topp2}</td>
                    <td>{sim_str}</td>
                    <td>{"Yes" if comp['drift'] else "No"}</td>
                    <td>{comp['severity']}</td>
                </tr>
                """
            
            detailed_analysis_html += "</table>"
        
        # Collect test names from all test results
        test_names = [test_result.test_name for test_result in drift_report.test_suite_result.test_results]
        test_name_display = ", ".join(test_names) if test_names else "N/A"
        
        return html_template.format(
            report_id=drift_report.report_id,
            test_name=test_name_display,
            generated_at=drift_report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            total_comparisons=summary['total_comparisons'],
            drift_count=summary['drift_detected_count'],
            drift_rate=summary['drift_rate'],
            avg_similarity=summary['average_similarity'],
            high_count=summary['severity_distribution']['high'],
            medium_count=summary['severity_distribution']['medium'],
            low_count=summary['severity_distribution']['low'],
            recommendations_html=recommendations_html,
            detailed_analysis_html=detailed_analysis_html
        )
    
    def _create_drift_distribution_chart(self, drift_report: DriftReport) -> str:
        """Create drift distribution chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        severity_dist = drift_report.drift_summary['severity_distribution']
        labels = list(severity_dist.keys())
        values = list(severity_dist.values())
        colors = ['#388e3c', '#f57c00', '#d32f2f']
        
        ax.bar(labels, values, color=colors)
        ax.set_title('Drift Severity Distribution')
        ax.set_ylabel('Count')
        
        chart_file = self.output_dir / f"{drift_report.report_id}_drift_distribution.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
    
    def _create_similarity_heatmap(self, drift_report: DriftReport) -> str:
        """Create similarity heatmap."""
        # This would require more complex data processing
        # For now, create a simple similarity distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        similarities = drift_report.charts_data.get('similarity_scores', [])
        ax.hist(similarities, bins=20, alpha=0.7, color='skyblue')
        ax.set_title('Semantic Similarity Distribution')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        
        chart_file = self.output_dir / f"{drift_report.report_id}_similarity_heatmap.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
    
    def _create_token_comparison_chart(self, drift_report: DriftReport) -> str:
        """Create token count comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        token_diffs = drift_report.charts_data.get('token_differences', [])
        ax.hist(token_diffs, bins=20, alpha=0.7, color='lightcoral')
        ax.set_title('Token Count Differences Distribution')
        ax.set_xlabel('Token Count Difference')
        ax.set_ylabel('Frequency')
        
        chart_file = self.output_dir / f"{drift_report.report_id}_token_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
    
    def _create_response_time_chart(self, drift_report: DriftReport) -> str:
        """Create response time analysis chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        times_1 = drift_report.charts_data.get('response_times_1', [])
        times_2 = drift_report.charts_data.get('response_times_2', [])
        
        ax1.hist(times_1, bins=20, alpha=0.7, label='Model 1', color='lightblue')
        ax1.hist(times_2, bins=20, alpha=0.7, label='Model 2', color='lightgreen')
        ax1.set_title('Response Time Distribution')
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Scatter plot
        ax2.scatter(times_1, times_2, alpha=0.6)
        ax2.set_title('Response Time Correlation')
        ax2.set_xlabel('Model 1 Response Time')
        ax2.set_ylabel('Model 2 Response Time')
        
        chart_file = self.output_dir / f"{drift_report.report_id}_response_time.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
    
    def _create_model_performance_chart(self, drift_report: DriftReport) -> str:
        """Create model performance comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        model_drift_rates = drift_report.drift_summary.get('model_drift_rates', {})
        if model_drift_rates:
            models = list(model_drift_rates.keys())
            rates = [model_drift_rates[model]['rate'] for model in models]
            
            bars = ax.bar(models, rates, color='lightsteelblue')
            ax.set_title('Drift Rate by Model Pair')
            ax.set_ylabel('Drift Rate')
            ax.set_xlabel('Model Pair')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
        
        chart_file = self.output_dir / f"{drift_report.report_id}_model_performance.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_file)
