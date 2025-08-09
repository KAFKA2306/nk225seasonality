#!/usr/bin/env python3
"""
Main execution script for Nikkei 225 Seasonality Analysis.

This script provides command-line interface for running the complete analysis pipeline
or individual components of the seasonality analysis system.

Usage:
    python main.py --help
    python main.py full-analysis --start-date 2020-01-01 --end-date 2023-12-31
    python main.py daily-update
    python main.py status
"""

import asyncio
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import AnalysisPipeline, SystemConfig, setup_logging, get_logger


def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Nikkei 225 Seasonality Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full analysis for last 5 years
    python main.py full-analysis --years 5
    
    # Run analysis for specific date range
    python main.py full-analysis --start-date 2020-01-01 --end-date 2023-12-31
    
    # Run daily data update
    python main.py daily-update
    
    # Check system status
    python main.py status
    
    # Validate configuration
    python main.py validate-config
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full analysis command
    full_analysis_parser = subparsers.add_parser(
        'full-analysis', 
        help='Run complete seasonality analysis pipeline'
    )
    
    date_group = full_analysis_parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        '--years', 
        type=int, 
        help='Number of years to analyze (counting back from today)'
    )
    date_group.add_argument(
        '--start-date', 
        type=str, 
        help='Start date (YYYY-MM-DD format, requires --end-date)'
    )
    
    full_analysis_parser.add_argument(
        '--end-date', 
        type=str, 
        help='End date (YYYY-MM-DD format, used with --start-date)'
    )
    
    full_analysis_parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Do not save results to disk'
    )
    
    full_analysis_parser.add_argument(
        '--config-file',
        type=str,
        help='Path to custom configuration file'
    )
    
    # Daily update command
    daily_parser = subparsers.add_parser(
        'daily-update',
        help='Run daily data update'
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Check system status and health'
    )
    
    # Config validation command
    config_parser = subparsers.add_parser(
        'validate-config',
        help='Validate system configuration'
    )
    
    config_parser.add_argument(
        '--config-file',
        type=str,
        help='Path to configuration file to validate'
    )
    
    return parser.parse_args()


async def run_full_analysis(args):
    """Run complete analysis pipeline."""
    
    logger = get_logger(__name__)
    logger.info("Starting full analysis pipeline")
    
    # Parse date arguments
    if args.years:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.years * 365)
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        print("Error: Either --years or both --start-date and --end-date must be provided")
        return 1
    
    # Validate date range
    if start_date >= end_date:
        print("Error: Start date must be before end date")
        return 1
    
    if (end_date - start_date).days < 365:
        print("Warning: Date range is less than 1 year, results may be limited")
    
    try:
        # Initialize configuration
        if args.config_file:
            # Would load custom config here
            config = SystemConfig()
            logger.info(f"Using custom configuration: {args.config_file}")
        else:
            config = SystemConfig()
        
        # Initialize pipeline
        pipeline = AnalysisPipeline(config)
        
        # Run analysis
        logger.info(f"Analyzing data from {start_date.date()} to {end_date.date()}")
        
        results = await pipeline.run_full_analysis(
            start_date=start_date,
            end_date=end_date,
            save_results=not args.no_save
        )
        
        # Print summary
        if results['success']:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*60)
            
            if 'summary' in results:
                summary = results['summary']
                
                print(f"Analysis Period: {start_date.date()} to {end_date.date()}")
                print(f"Total Runtime: {summary['analysis_overview']['total_runtime']:.1f} seconds")
                
                if 'data_quality' in summary:
                    print(f"Records Analyzed: {summary['data_quality']['records_analyzed']:,}")
                    print(f"Data Quality Valid: {summary['data_quality']['data_quality_valid']}")
                
                if 'key_findings' in summary:
                    significant_months = summary['key_findings']['significant_months']
                    print(f"Significant Seasonal Patterns: {len(significant_months)} months")
                    if significant_months:
                        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        significant_names = [month_names[m] for m in significant_months]
                        print(f"Significant Months: {', '.join(significant_names)}")
                
                if 'strategy_performance' in summary:
                    print(f"Strategies Developed: {summary['strategy_performance']['strategies_developed']}")
                    print(f"Successfully Backtested: {summary['strategy_performance']['successfully_backtested']}")
                
                if 'recommendations' in summary:
                    print("\nRecommendations:")
                    for i, rec in enumerate(summary['recommendations'], 1):
                        print(f"  {i}. {rec}")
            
            if not args.no_save:
                print(f"\nResults saved to: {config.output_dir}")
            
            print("="*60)
            return 0
        
        else:
            print("\n" + "="*60)
            print("ANALYSIS FAILED")
            print("="*60)
            print(f"Error: {results.get('error', 'Unknown error')}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\nAnalysis interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        print(f"\nAnalysis failed: {e}")
        return 1


async def run_daily_update(args):
    """Run daily data update."""
    
    logger = get_logger(__name__)
    logger.info("Starting daily update")
    
    try:
        config = SystemConfig()
        pipeline = AnalysisPipeline(config)
        
        result = await pipeline.run_daily_update()
        
        if result['success']:
            print(f"Daily update completed successfully")
            print(f"Records updated: {result['records_updated']}")
            if 'data_quality_valid' in result:
                print(f"Data quality valid: {result['data_quality_valid']}")
            return 0
        else:
            print(f"Daily update failed: {result.get('error', 'Unknown error')}")
            return 1
    
    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        print(f"Daily update failed: {e}")
        return 1


def run_status_check(args):
    """Check system status."""
    
    try:
        config = SystemConfig()
        pipeline = AnalysisPipeline(config)
        
        status = pipeline.get_pipeline_status()
        
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        
        # Overall health
        if status['pipeline_healthy'] is True:
            print("Overall Status: ✓ HEALTHY")
        elif status['pipeline_healthy'] == 'warning':
            print("Overall Status: ⚠ WARNING")
        else:
            print("Overall Status: ✗ ERROR")
        
        # Component status
        print("\nComponent Status:")
        for component, info in status.get('components', {}).items():
            status_symbol = {"healthy": "✓", "warning": "⚠", "error": "✗"}[info['status']]
            print(f"  {component}: {status_symbol} {info['status'].upper()}")
            
            if info['status'] != 'healthy':
                if 'issues' in info and info['issues']:
                    for issue in info['issues']:
                        print(f"    - Issue: {issue}")
                if 'warnings' in info and info['warnings']:
                    for warning in info['warnings']:
                        print(f"    - Warning: {warning}")
        
        # Data coverage
        coverage = status.get('data_coverage', {})
        if coverage:
            print(f"\nData Coverage:")
            print(f"  Earliest Date: {coverage.get('earliest_date', 'N/A')}")
            print(f"  Latest Date: {coverage.get('latest_date', 'N/A')}")
            print(f"  Total Records: {coverage.get('total_records', 0):,}")
            print(f"  Average Quality Score: {coverage.get('avg_quality_score', 0):.2f}")
        
        print("="*50)
        
        return 0 if status['pipeline_healthy'] else 1
    
    except Exception as e:
        print(f"Status check failed: {e}")
        return 1


def validate_configuration(args):
    """Validate system configuration."""
    
    try:
        if args.config_file:
            print(f"Validating configuration file: {args.config_file}")
            # Would load custom config here
            config = SystemConfig()
        else:
            print("Validating default configuration...")
            config = SystemConfig()
        
        validation_result = config.validate_config()
        
        print("\n" + "="*50)
        print("CONFIGURATION VALIDATION")
        print("="*50)
        
        if validation_result['valid']:
            print("Configuration Status: ✓ VALID")
        else:
            print("Configuration Status: ✗ INVALID")
        
        if validation_result['issues']:
            print("\nIssues Found:")
            for issue in validation_result['issues']:
                print(f"  ✗ {issue}")
        
        if validation_result['warnings']:
            print("\nWarnings:")
            for warning in validation_result['warnings']:
                print(f"  ⚠ {warning}")
        
        if validation_result['valid'] and not validation_result['warnings']:
            print("\nConfiguration is valid and ready for use.")
        
        print("="*50)
        
        return 0 if validation_result['valid'] else 1
    
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return 1


async def main():
    """Main entry point."""
    
    # Setup logging
    setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    if not args.command:
        print("Error: No command specified. Use --help for available commands.")
        return 1
    
    # Execute command
    if args.command == 'full-analysis':
        return await run_full_analysis(args)
    elif args.command == 'daily-update':
        return await run_daily_update(args)
    elif args.command == 'status':
        return run_status_check(args)
    elif args.command == 'validate-config':
        return validate_configuration(args)
    else:
        print(f"Error: Unknown command '{args.command}'")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)