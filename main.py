import argparse
import asyncio
from datetime import datetime, timedelta

from src import AnalysisPipeline, SystemConfig, run_analysis_report
from src.analysis.valuation import run_time_series_report


async def run_seasonality(args):
    print("SEASONALITY ANALYSIS")
    config = SystemConfig()
    pipeline = AnalysisPipeline(config)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)

    print(f"Analyzing {args.years} years: {start_date.date()} to {end_date.date()}")

    results = await pipeline.run_full_analysis(start_date, end_date, save_results=True)

    if results["success"]:
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        if "summary" in results and "key_findings" in results["summary"]:
            months = results["summary"]["key_findings"].get("significant_months", [])
            print(f"Significant Months: {months}")
        print(f"Report saved to: {config.output_dir}")
    else:
        print(f"Analysis Failed: {results.get('error')}")


def run():
    parser = argparse.ArgumentParser(description="Nikkei 225 Analysis System")
    subparsers = parser.add_subparsers(dest="command", help="Select analysis mode")

    # Valuation Command
    val = subparsers.add_parser("valuation", help="Market Valuation (Yield Gap) Analysis")
    val.add_argument("--current-per", type=float, default=16.0, help="Current Market PER")
    # jgb-yield is now dynamic
    val.add_argument("--risk-premium", type=float, default=3.5, help="Equity Risk Premium %")

    # Time Series Valuation Command
    valts = subparsers.add_parser("valuation-ts", help="Time Series Valuation Analysis")
    valts.add_argument("--years", type=int, default=10, help="Years of history")
    # jgb-yield is now dynamic
    valts.add_argument("--risk-premium", type=float, default=3.5, help="Risk Premium %")

    # Seasonality Command
    seas = subparsers.add_parser("seasonality", help="Full Seasonality & Strategy Analysis")
    seas.add_argument("--years", type=int, default=5, help="Years of history to analyze")

    args = parser.parse_args()

    if args.command == "valuation":
        # Updated to remove static yield
        from src import SystemConfig
        from src.analysis.valuation import fetch_current_jgb_yield
        
        config = SystemConfig()
        current_yield = fetch_current_jgb_yield(config.valuation.jgb_ticker)
        print(f"Fetched JGB Yield: {current_yield}%")
        
        run_analysis_report(current_yield, args.current_per, args.risk_premium)
    elif args.command == "valuation-ts":
        run_time_series_report(args.years)
    elif args.command == "seasonality":
        asyncio.run(run_seasonality(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    run()
