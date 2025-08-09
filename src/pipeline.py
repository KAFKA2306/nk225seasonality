"""
Main analysis pipeline orchestrator for Nikkei 225 seasonality analysis.

This module orchestrates the complete analysis workflow including data collection,
validation, statistical analysis, strategy generation, and reporting.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

from .config import SystemConfig, get_logger, setup_logging
from .data import DataIngestionPipeline, DataValidator, MarketDataRepository
from .analysis import SeasonalityAnalyzer, SeasonalRegressionModel, MechanismAnalyzer
from .options import SeasonalOptionsStrategy, StrategyBacktester, OptionsCalculator
from .risk import MonteCarloEngine, VaRCalculator, ProcessParameters, StochasticProcess
from .visualization import SeasonalityVisualizer, OptionsVisualizer, RiskVisualizer


class AnalysisPipeline:
    """Main orchestrator for the complete analysis pipeline."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            config: System configuration (creates default if None)
        """
        self.config = config or SystemConfig()
        
        # Setup logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.data_ingestion = DataIngestionPipeline(self.config)
        self.data_validator = DataValidator()
        self.data_repository = MarketDataRepository(self.config)
        self.options_calculator = OptionsCalculator()
        self.mc_engine = MonteCarloEngine(self.config.analysis.monte_carlo_simulations)
        self.var_calculator = VaRCalculator()
        
        # Initialize visualizers
        self.seasonality_viz = SeasonalityVisualizer(self.config.output_dir)
        self.options_viz = OptionsVisualizer(self.config.output_dir)
        self.risk_viz = RiskVisualizer(self.config.output_dir)
        
        self.logger.info("Analysis pipeline initialized successfully")
    
    async def run_full_analysis(self, 
                               start_date: datetime,
                               end_date: datetime,
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            save_results: Whether to save results to disk
            
        Returns:
            Complete analysis results
        """
        
        self.logger.info(f"Starting full analysis from {start_date} to {end_date}")
        
        pipeline_results = {
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'analysis_timestamp': datetime.now(),
                'config_used': self.config.export_config()
            }
        }
        
        try:
            # Phase 1: Data Collection and Validation
            self.logger.info("Phase 1: Data Collection and Validation")
            data_results = await self._execute_data_phase(start_date, end_date)
            pipeline_results['data_phase'] = data_results
            
            if not data_results['success']:
                raise RuntimeError("Data collection failed")
            
            # Phase 2: Statistical Analysis
            self.logger.info("Phase 2: Statistical Analysis")
            analysis_results = await self._execute_analysis_phase(data_results['validated_data'])
            pipeline_results['analysis_phase'] = analysis_results
            
            # Phase 3: Strategy Development
            self.logger.info("Phase 3: Strategy Development")
            strategy_results = await self._execute_strategy_phase(
                data_results['validated_data'], 
                analysis_results['seasonality_results']
            )
            pipeline_results['strategy_phase'] = strategy_results
            
            # Phase 4: Risk Assessment
            self.logger.info("Phase 4: Risk Assessment")
            risk_results = await self._execute_risk_phase(
                data_results['validated_data'],
                strategy_results['strategies']
            )
            pipeline_results['risk_phase'] = risk_results
            
            # Phase 5: Visualization and Reporting
            self.logger.info("Phase 5: Visualization and Reporting")
            viz_results = await self._execute_visualization_phase(
                data_results['validated_data'],
                analysis_results,
                strategy_results,
                risk_results
            )
            pipeline_results['visualization_phase'] = viz_results
            
            # Phase 6: Generate Summary Report
            self.logger.info("Phase 6: Generating Summary Report")
            summary = self._generate_summary_report(pipeline_results)
            pipeline_results['summary'] = summary
            
            # Save results if requested
            if save_results:
                await self._save_pipeline_results(pipeline_results)
            
            self.logger.info("Full analysis pipeline completed successfully")
            pipeline_results['success'] = True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
            raise
        
        return pipeline_results
    
    async def _execute_data_phase(self, 
                                 start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Execute data collection and validation phase."""
        
        phase_results = {'phase': 'data_collection_validation'}
        
        try:
            # Collect data from multiple sources
            raw_data = await self.data_ingestion.collect_data(start_date, end_date)
            
            if raw_data.empty:
                raise ValueError("No data collected from any source")
            
            # Validate data quality
            validation_results = self.data_validator.validate_dataset(raw_data)
            
            # Store validated data in repository
            storage_success = self.data_repository.store_data(raw_data, "pipeline_analysis")
            
            phase_results.update({
                'success': True,
                'raw_data_shape': raw_data.shape,
                'data_quality_valid': validation_results.is_valid,
                'validation_issues': len(validation_results.issues),
                'storage_success': storage_success,
                'validated_data': raw_data,
                'validation_details': validation_results.summary
            })
            
            self.logger.info(f"Data phase completed: {raw_data.shape[0]} records collected")
            
        except Exception as e:
            phase_results.update({
                'success': False,
                'error': str(e)
            })
            self.logger.error(f"Data phase failed: {e}")
        
        return phase_results
    
    async def _execute_analysis_phase(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute statistical analysis phase."""
        
        phase_results = {'phase': 'statistical_analysis'}
        
        try:
            # Seasonality Analysis
            seasonality_analyzer = SeasonalityAnalyzer(
                data, 
                significance_level=self.config.analysis.significance_level
            )
            
            seasonality_results = seasonality_analyzer.test_monthly_patterns()
            dow_results = seasonality_analyzer.test_day_of_week_patterns()
            quarter_results = seasonality_analyzer.test_quarter_patterns()
            
            # Rolling analysis for robustness
            rolling_results = seasonality_analyzer.rolling_seasonality_analysis(
                window_years=5, step_months=6
            )
            
            # Regression Analysis
            regression_model = SeasonalRegressionModel(data)
            fitted_model = regression_model.fit_seasonal_model()
            
            # Joint significance tests
            feature_groups = {
                'monthly_effects': [f'month_{i}' for i in range(2, 13)],
                'day_of_week_effects': ['dow_tuesday', 'dow_wednesday', 'dow_thursday', 'dow_friday'],
                'japanese_factors': ['fiscal_year_end', 'golden_week']
            }
            
            joint_tests = regression_model.test_joint_significance(fitted_model, feature_groups)
            
            # Mechanism Analysis
            mechanism_analyzer = MechanismAnalyzer(data)
            mechanism_results = mechanism_analyzer.comprehensive_mechanism_analysis()
            
            phase_results.update({
                'success': True,
                'seasonality_results': seasonality_results,
                'day_of_week_results': dow_results,
                'quarterly_results': quarter_results,
                'rolling_analysis': rolling_results,
                'regression_results': {
                    'model_summary': str(fitted_model.summary()),
                    'r_squared': fitted_model.rsquared,
                    'adjusted_r_squared': fitted_model.rsquared_adj,
                    'joint_significance_tests': joint_tests
                },
                'mechanism_analysis': mechanism_results,
                'significant_months': [
                    month for month, result in seasonality_results.items() 
                    if result.is_significant
                ]
            })
            
            self.logger.info(f"Analysis phase completed: {len(phase_results['significant_months'])} significant months found")
            
        except Exception as e:
            phase_results.update({
                'success': False,
                'error': str(e)
            })
            self.logger.error(f"Analysis phase failed: {e}")
        
        return phase_results
    
    async def _execute_strategy_phase(self, 
                                    data: pd.DataFrame,
                                    seasonality_results: Dict[int, Any]) -> Dict[str, Any]:
        """Execute strategy development phase."""
        
        phase_results = {'phase': 'strategy_development'}
        
        try:
            strategy_engine = SeasonalOptionsStrategy(data, seasonality_results)
            
            # Generate strategies for significant months
            strategies = {}
            backtest_results = {}
            
            current_price = data['close_price'].iloc[-1] if 'close_price' in data.columns else 100
            
            for month, result in seasonality_results.items():
                if result.is_significant:
                    
                    # Design strategy based on expected return direction
                    if result.mean_return > 0:
                        # Bullish month - use call spread
                        strategy = strategy_engine.design_call_spread_strategy(month)
                    else:
                        # Bearish month - use put spread
                        strategy = strategy_engine.design_put_spread_strategy(month)
                    
                    strategies[month] = strategy
                    
                    # Calculate strategy metrics
                    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 100)
                    payoff_analysis = strategy_engine.calculate_strategy_payoff(
                        strategy, price_range, result.std_return
                    )
                    
                    # Backtest strategy
                    backtester = StrategyBacktester(data)
                    backtest_result = backtester.backtest_strategy(
                        strategy, 
                        data.index.min(), 
                        data.index.max()
                    )
                    
                    strategies[month] = {
                        'definition': strategy,
                        'payoff_analysis': payoff_analysis,
                        'expected_performance': {
                            'max_profit': payoff_analysis['max_profit'],
                            'max_loss': payoff_analysis['max_loss'],
                            'breakeven_points': payoff_analysis['breakeven_points']
                        }
                    }
                    
                    if 'error' not in backtest_result:
                        backtest_results[month] = backtest_result
            
            phase_results.update({
                'success': True,
                'strategies': strategies,
                'backtest_results': backtest_results,
                'total_strategies_developed': len(strategies),
                'successfully_backtested': len(backtest_results)
            })
            
            self.logger.info(f"Strategy phase completed: {len(strategies)} strategies developed")
            
        except Exception as e:
            phase_results.update({
                'success': False,
                'error': str(e)
            })
            self.logger.error(f"Strategy phase failed: {e}")
        
        return phase_results
    
    async def _execute_risk_phase(self, 
                                data: pd.DataFrame,
                                strategies: Dict[int, Any]) -> Dict[str, Any]:
        """Execute risk assessment phase."""
        
        phase_results = {'phase': 'risk_assessment'}
        
        try:
            risk_results = {}
            current_price = data['close_price'].iloc[-1] if 'close_price' in data.columns else 100
            
            # Calculate historical volatility
            returns = data['returns'] if 'returns' in data.columns else data['close_price'].pct_change()
            historical_vol = returns.std() * np.sqrt(252)  # Annualized
            
            # Setup process parameters
            process_params = ProcessParameters(
                mu=returns.mean() * 252,  # Annualized drift
                sigma=historical_vol,
                historical_returns=returns.dropna().values
            )
            
            # Run Monte Carlo for each strategy
            for month, strategy_data in strategies.items():
                strategy_def = strategy_data['definition']
                
                # Monte Carlo simulation
                mc_results = self.mc_engine.simulate_strategy_performance(
                    strategy_def, current_price, StochasticProcess.GEOMETRIC_BROWNIAN_MOTION,
                    process_params
                )
                
                # Stress testing
                stress_scenarios = {
                    'high_volatility': {'mu': process_params.mu, 'sigma': historical_vol * 1.5},
                    'low_volatility': {'mu': process_params.mu, 'sigma': historical_vol * 0.7},
                    'bear_market': {'mu': -0.15, 'sigma': historical_vol * 1.2},
                    'bull_market': {'mu': 0.15, 'sigma': historical_vol * 0.9}
                }
                
                stress_results = self.mc_engine.stress_test(
                    strategy_def, current_price, historical_vol, stress_scenarios
                )
                
                # VaR calculations
                strategy_returns = mc_results['payoffs']
                var_results = {}
                
                for confidence in [0.95, 0.99]:
                    var_result = self.var_calculator.calculate_var(
                        strategy_returns, confidence
                    )
                    var_results[f'var_{int(confidence*100)}'] = var_result
                
                risk_results[month] = {
                    'monte_carlo': mc_results,
                    'stress_testing': stress_results,
                    'var_analysis': var_results
                }
            
            # Portfolio-level risk if multiple strategies
            if len(risk_results) > 1:
                # Simple equal-weight portfolio analysis
                all_payoffs = [results['monte_carlo']['payoffs'] for results in risk_results.values()]
                portfolio_payoffs = np.mean(all_payoffs, axis=0)
                
                portfolio_mc = {
                    'payoffs': portfolio_payoffs,
                    'risk_metrics': self.mc_engine._calculate_risk_metrics(portfolio_payoffs)
                }
                
                phase_results['portfolio_risk'] = portfolio_mc
            
            phase_results.update({
                'success': True,
                'individual_strategy_risk': risk_results,
                'risk_assessment_completed': len(risk_results)
            })
            
            self.logger.info(f"Risk phase completed: {len(risk_results)} strategies assessed")
            
        except Exception as e:
            phase_results.update({
                'success': False,
                'error': str(e)
            })
            self.logger.error(f"Risk phase failed: {e}")
        
        return phase_results
    
    async def _execute_visualization_phase(self, 
                                         data: pd.DataFrame,
                                         analysis_results: Dict[str, Any],
                                         strategy_results: Dict[str, Any],
                                         risk_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization and reporting phase."""
        
        phase_results = {'phase': 'visualization_reporting'}
        
        try:
            generated_plots = []
            
            # Seasonality visualizations
            if 'seasonality_results' in analysis_results:
                # Main seasonality heatmap
                fig1 = self.seasonality_viz.create_seasonal_heatmap(
                    analysis_results['seasonality_results'],
                    save_path='seasonality_heatmap.png'
                )
                generated_plots.append('seasonality_heatmap.png')
                
                # Statistical summary
                fig2 = self.seasonality_viz.create_statistical_summary_plot(
                    analysis_results['seasonality_results'],
                    save_path='statistical_summary.png'
                )
                generated_plots.append('statistical_summary.png')
                
                # Rolling analysis
                if 'rolling_analysis' in analysis_results:
                    fig3 = self.seasonality_viz.create_rolling_seasonality_plot(
                        analysis_results['rolling_analysis'],
                        save_path='rolling_seasonality.png'
                    )
                    generated_plots.append('rolling_seasonality.png')
                
                # Time series with highlights
                significant_months = analysis_results.get('significant_months', [])
                fig4 = self.seasonality_viz.create_seasonal_time_series(
                    data, highlight_months=significant_months,
                    save_path='time_series_seasonal.png'
                )
                generated_plots.append('time_series_seasonal.png')
                
                # Comprehensive dashboard
                fig5 = self.seasonality_viz.create_dashboard(
                    data, analysis_results['seasonality_results'],
                    save_path='seasonality_dashboard.png'
                )
                generated_plots.append('seasonality_dashboard.png')
            
            # Strategy visualizations
            if 'strategies' in strategy_results:
                current_price = data['close_price'].iloc[-1] if 'close_price' in data.columns else 100
                
                for month, strategy_data in strategy_results['strategies'].items():
                    strategy_def = strategy_data['definition']
                    
                    # Payoff diagram
                    fig = self.options_viz.create_payoff_diagram(
                        strategy_def, current_price,
                        save_path=f'payoff_diagram_month_{month}.png'
                    )
                    generated_plots.append(f'payoff_diagram_month_{month}.png')
                
                # Backtest performance
                if 'backtest_results' in strategy_results:
                    for month, backtest_data in strategy_results['backtest_results'].items():
                        fig = self.options_viz.create_backtest_performance(
                            backtest_data,
                            save_path=f'backtest_performance_month_{month}.png'
                        )
                        generated_plots.append(f'backtest_performance_month_{month}.png')
            
            # Risk visualizations
            if 'individual_strategy_risk' in risk_results:
                for month, risk_data in risk_results['individual_strategy_risk'].items():
                    # Monte Carlo results
                    fig = self.risk_viz.create_monte_carlo_results(
                        risk_data['monte_carlo'],
                        save_path=f'monte_carlo_results_month_{month}.png'
                    )
                    generated_plots.append(f'monte_carlo_results_month_{month}.png')
                    
                    # Stress test results
                    fig = self.risk_viz.create_stress_test_results(
                        risk_data['stress_testing'],
                        save_path=f'stress_test_results_month_{month}.png'
                    )
                    generated_plots.append(f'stress_test_results_month_{month}.png')
                
                # Portfolio risk if available
                if 'portfolio_risk' in risk_results:
                    fig = self.risk_viz.create_monte_carlo_results(
                        risk_results['portfolio_risk'],
                        save_path='portfolio_monte_carlo.png'
                    )
                    generated_plots.append('portfolio_monte_carlo.png')
            
            phase_results.update({
                'success': True,
                'generated_plots': generated_plots,
                'total_plots_created': len(generated_plots)
            })
            
            self.logger.info(f"Visualization phase completed: {len(generated_plots)} plots generated")
            
        except Exception as e:
            phase_results.update({
                'success': False,
                'error': str(e)
            })
            self.logger.error(f"Visualization phase failed: {e}")
        
        return phase_results
    
    def _generate_summary_report(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analysis results."""
        
        summary = {
            'analysis_overview': {
                'total_runtime': (datetime.now() - pipeline_results['metadata']['analysis_timestamp']).total_seconds(),
                'data_period': {
                    'start': pipeline_results['metadata']['start_date'],
                    'end': pipeline_results['metadata']['end_date']
                }
            }
        }
        
        # Data quality summary
        if pipeline_results.get('data_phase', {}).get('success'):
            data_phase = pipeline_results['data_phase']
            summary['data_quality'] = {
                'records_analyzed': data_phase['raw_data_shape'][0],
                'data_quality_valid': data_phase['data_quality_valid'],
                'validation_issues': data_phase['validation_issues']
            }
        
        # Key findings
        if pipeline_results.get('analysis_phase', {}).get('success'):
            analysis_phase = pipeline_results['analysis_phase']
            summary['key_findings'] = {
                'significant_months': analysis_phase.get('significant_months', []),
                'total_significant_patterns': len(analysis_phase.get('significant_months', [])),
                'strongest_effect': None,  # Would calculate from results
                'regression_r_squared': analysis_phase.get('regression_results', {}).get('r_squared', 0)
            }
        
        # Strategy performance
        if pipeline_results.get('strategy_phase', {}).get('success'):
            strategy_phase = pipeline_results['strategy_phase']
            summary['strategy_performance'] = {
                'strategies_developed': strategy_phase['total_strategies_developed'],
                'successfully_backtested': strategy_phase['successfully_backtested'],
                'average_win_rate': None,  # Would calculate from backtest results
                'best_performing_month': None  # Would identify from results
            }
        
        # Risk assessment summary
        if pipeline_results.get('risk_phase', {}).get('success'):
            risk_phase = pipeline_results['risk_phase']
            summary['risk_assessment'] = {
                'strategies_risk_assessed': risk_phase['risk_assessment_completed'],
                'portfolio_diversification_available': 'portfolio_risk' in risk_phase,
                'maximum_var_95': None,  # Would calculate from individual strategy risks
                'average_sharpe_ratio': None  # Would calculate from risk metrics
            }
        
        # Recommendations
        summary['recommendations'] = self._generate_recommendations(pipeline_results)
        
        return summary
    
    def _generate_recommendations(self, pipeline_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        
        recommendations = []
        
        # Check if analysis was successful
        if not pipeline_results.get('analysis_phase', {}).get('success'):
            recommendations.append("Improve data quality before proceeding with strategy implementation")
            return recommendations
        
        significant_months = pipeline_results.get('analysis_phase', {}).get('significant_months', [])
        
        if len(significant_months) == 0:
            recommendations.append("No statistically significant seasonal patterns detected. Consider alternative strategies.")
        elif len(significant_months) == 1:
            recommendations.append(f"Focus strategy development on Month {significant_months[0]} due to strong seasonal effect")
        elif len(significant_months) > 3:
            recommendations.append("Multiple seasonal opportunities detected. Consider portfolio approach to diversify risk")
        
        # Strategy-specific recommendations
        if pipeline_results.get('strategy_phase', {}).get('success'):
            strategy_count = pipeline_results['strategy_phase']['total_strategies_developed']
            backtest_count = pipeline_results['strategy_phase']['successfully_backtested']
            
            if backtest_count < strategy_count:
                recommendations.append("Some strategies could not be backtested. Increase historical data coverage")
        
        # Risk management recommendations
        if pipeline_results.get('risk_phase', {}).get('success'):
            if 'portfolio_risk' in pipeline_results['risk_phase']:
                recommendations.append("Portfolio diversification reduces individual strategy risk. Consider equal-weight allocation")
            else:
                recommendations.append("Consider developing multiple strategies for risk diversification")
        
        # Data quality recommendations
        data_issues = pipeline_results.get('data_phase', {}).get('validation_issues', 0)
        if data_issues > 0:
            recommendations.append(f"Address {data_issues} data quality issues to improve analysis reliability")
        
        return recommendations
    
    async def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results to disk."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.config.output_dir / f"analysis_results_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON (excluding large data objects)
        summary_results = {
            'metadata': results['metadata'],
            'summary': results.get('summary', {}),
            'phase_results': {
                phase: {k: v for k, v in phase_data.items() 
                       if k not in ['validated_data', 'raw_data_shape']}
                for phase, phase_data in results.items()
                if phase.endswith('_phase')
            }
        }
        
        with open(results_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary_results, f, indent=2, default=str)
        
        # Save detailed data if available
        if 'data_phase' in results and 'validated_data' in results['data_phase']:
            data = results['data_phase']['validated_data']
            data.to_csv(results_dir / 'validated_market_data.csv')
        
        self.logger.info(f"Pipeline results saved to {results_dir}")
    
    async def run_daily_update(self) -> Dict[str, Any]:
        """Run daily incremental analysis update."""
        
        self.logger.info("Starting daily analysis update")
        
        # Get latest available data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Last 5 days
        
        try:
            # Quick data update
            daily_data = await self.data_ingestion.collect_data(start_date, end_date)
            
            if not daily_data.empty:
                # Store new data
                self.data_repository.store_data(daily_data, "daily_update")
                
                # Quick validation
                validation_results = self.data_validator.validate_dataset(daily_data)
                
                return {
                    'success': True,
                    'records_updated': len(daily_data),
                    'data_quality_valid': validation_results.is_valid,
                    'update_timestamp': datetime.now()
                }
            else:
                return {
                    'success': True,
                    'records_updated': 0,
                    'message': 'No new data available'
                }
                
        except Exception as e:
            self.logger.error(f"Daily update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health check."""
        
        status = {
            'pipeline_healthy': True,
            'components': {},
            'data_coverage': {},
            'last_analysis': None
        }
        
        try:
            # Check data repository
            data_stats = self.data_repository.get_statistics()
            status['components']['data_repository'] = {
                'status': 'healthy' if data_stats.get('total_records', 0) > 0 else 'warning',
                'total_records': data_stats.get('total_records', 0),
                'database_size_mb': data_stats.get('database_size_mb', 0)
            }
            
            # Check data coverage
            coverage = self.data_repository.get_data_coverage()
            status['data_coverage'] = coverage
            
            # Check configuration validity
            config_validation = self.config.validate_config()
            status['components']['configuration'] = {
                'status': 'healthy' if config_validation['valid'] else 'error',
                'issues': config_validation.get('issues', []),
                'warnings': config_validation.get('warnings', [])
            }
            
            # Overall health
            component_statuses = [comp['status'] for comp in status['components'].values()]
            if 'error' in component_statuses:
                status['pipeline_healthy'] = False
            elif 'warning' in component_statuses:
                status['pipeline_healthy'] = 'warning'
            
        except Exception as e:
            status['pipeline_healthy'] = False
            status['error'] = str(e)
        
        return status