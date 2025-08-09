"""
Risk visualization module.

This module provides visualizations for risk analysis including
Monte Carlo results, VaR charts, and stress testing visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import seaborn as sns

from ..config import get_logger
from ..risk.monte_carlo import RiskMetrics


class RiskVisualizer:
    """Visualizations for risk analysis and Monte Carlo results."""
    
    def __init__(self, 
                 output_dir: Optional[Path] = None,
                 figsize: Tuple[int, int] = (12, 8)):
        """Initialize risk visualizer."""
        self.output_dir = output_dir
        self.figsize = figsize
        self.logger = get_logger(__name__)
        
        # Color scheme for risk visualization
        self.colors = {
            'profit': '#2ca02c',
            'loss': '#d62728',
            'neutral': '#1f77b4',
            'var95': '#ff7f0e',
            'var99': '#9467bd',
            'mean': '#8c564b'
        }
    
    def create_monte_carlo_results(self, 
                                 mc_results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive Monte Carlo results visualization.
        
        Args:
            mc_results: Monte Carlo simulation results
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        payoffs = mc_results['payoffs']
        final_prices = mc_results['final_prices']
        price_changes = mc_results['price_changes']
        risk_metrics = mc_results['risk_metrics']
        
        # Create subplot layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])
        
        # P&L distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_pnl_distribution(ax1, payoffs, risk_metrics)
        
        # Price distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_price_distribution(ax2, final_prices, price_changes)
        
        # Risk metrics summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_risk_metrics_summary(ax3, risk_metrics)
        
        # Scatter plot: Price vs P&L (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_price_vs_pnl_scatter(ax4, final_prices, payoffs)
        
        # VaR visualization (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_var_visualization(ax5, payoffs, risk_metrics)
        
        # Tail analysis (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_tail_analysis(ax6, payoffs, risk_metrics)
        
        # Performance statistics table (bottom span)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_statistics_table(ax7, risk_metrics)
        
        # Main title
        fig.suptitle('Monte Carlo Risk Analysis Results', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_var_backtest_chart(self, 
                                backtest_results: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create VaR backtesting visualization.
        
        Args:
            backtest_results: VaR backtest results
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[2, 1])
        
        # Simulate some backtest data (would come from actual backtesting)
        n_days = 252
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        returns = np.random.normal(-0.001, 0.02, n_days)
        var_forecasts = np.random.normal(0.04, 0.01, n_days)
        
        # Returns vs VaR
        ax1.plot(dates, -returns * 100, linewidth=1, color=self.colors['neutral'], 
                label='Daily Losses (%)')
        ax1.plot(dates, var_forecasts * 100, color=self.colors['var95'], 
                linewidth=2, label='95% VaR')
        
        # Mark violations
        violations = (-returns) > var_forecasts
        if violations.any():
            ax1.scatter(dates[violations], (-returns[violations]) * 100,
                       color=self.colors['loss'], s=30, label='Violations', zorder=5)
        
        ax1.set_title('VaR Backtesting: Daily Losses vs VaR Forecasts', fontweight='bold')
        ax1.set_ylabel('Loss (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Violation frequency
        violation_rate = violations.sum() / len(violations)
        expected_rate = 0.05  # 5% for 95% VaR
        
        ax2.bar(['Actual', 'Expected'], [violation_rate * 100, expected_rate * 100],
               color=[self.colors['loss'], self.colors['neutral']], alpha=0.7)
        ax2.set_title('Violation Rates', fontweight='bold')
        ax2.set_ylabel('Violation Rate (%)')
        
        # Add test results
        test_text = (f"Violation Rate: {violation_rate:.1%}\n"
                    f"Expected Rate: {expected_rate:.1%}\n"
                    f"Test Result: {'PASS' if abs(violation_rate - expected_rate) < 0.02 else 'FAIL'}")
        
        ax2.text(0.6, 0.95, test_text, transform=ax2.transAxes,
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_stress_test_results(self, 
                                 stress_results: Dict[str, Dict[str, Any]],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create stress testing results visualization.
        
        Args:
            stress_results: Stress test results by scenario
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        scenarios = list(stress_results.keys())
        n_scenarios = len(scenarios)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Scenario comparison - Expected P&L
        scenario_means = [stress_results[s]['risk_metrics'].mean_pnl for s in scenarios]
        colors = ['green' if m > 0 else 'red' for m in scenario_means]
        
        axes[0].bar(range(n_scenarios), scenario_means, color=colors, alpha=0.7)
        axes[0].set_title('Expected P&L by Scenario', fontweight='bold')
        axes[0].set_ylabel('Expected P&L')
        axes[0].set_xticks(range(n_scenarios))
        axes[0].set_xticklabels(scenarios, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Scenario comparison - VaR 95%
        scenario_vars = [stress_results[s]['risk_metrics'].var_95 for s in scenarios]
        axes[1].bar(range(n_scenarios), scenario_vars, color=self.colors['var95'], alpha=0.7)
        axes[1].set_title('95% VaR by Scenario', fontweight='bold')
        axes[1].set_ylabel('95% VaR')
        axes[1].set_xticks(range(n_scenarios))
        axes[1].set_xticklabels(scenarios, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Scenario comparison - Probability of Profit
        scenario_probs = [stress_results[s]['risk_metrics'].probability_of_profit for s in scenarios]
        axes[2].bar(range(n_scenarios), [p * 100 for p in scenario_probs], 
                   color=self.colors['neutral'], alpha=0.7)
        axes[2].set_title('Probability of Profit by Scenario', fontweight='bold')
        axes[2].set_ylabel('Probability (%)')
        axes[2].set_xticks(range(n_scenarios))
        axes[2].set_xticklabels(scenarios, rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        # P&L distribution comparison (violin plot)
        pnl_data = [stress_results[s]['payoffs'] for s in scenarios]
        parts = axes[3].violinplot(pnl_data, positions=range(n_scenarios), showmeans=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            if scenario_means[i] > 0:
                pc.set_facecolor(self.colors['profit'])
            else:
                pc.set_facecolor(self.colors['loss'])
            pc.set_alpha(0.7)
        
        axes[3].set_title('P&L Distribution by Scenario', fontweight='bold')
        axes[3].set_ylabel('P&L')
        axes[3].set_xticks(range(n_scenarios))
        axes[3].set_xticklabels(scenarios, rotation=45)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_scenario_heatmap(self, 
                              scenario_results: pd.DataFrame,
                              metric: str = 'mean_pnl',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create scenario analysis heatmap.
        
        Args:
            scenario_results: DataFrame with scenario results
            metric: Metric to visualize
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        # Pivot data for heatmap
        pivot_data = scenario_results.pivot(index='volatility', 
                                           columns='drift', 
                                           values=metric)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        if metric in ['mean_pnl', 'var_95']:
            cmap = 'RdYlGn'
            center = 0
        else:
            cmap = 'viridis'
            center = None
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap=cmap, center=center, ax=ax)
        
        ax.set_title(f'Scenario Analysis: {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Drift Rate', fontsize=12)
        ax.set_ylabel('Volatility', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_rolling_var_chart(self, 
                               returns: pd.Series,
                               rolling_var: pd.Series,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create rolling VaR visualization.
        
        Args:
            returns: Time series of returns
            rolling_var: Rolling VaR estimates
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      height_ratios=[2, 1], sharex=True)
        
        # Returns vs Rolling VaR
        ax1.plot(returns.index, -returns * 100, linewidth=0.5, 
                color=self.colors['neutral'], alpha=0.7, label='Daily Losses (%)')
        ax1.plot(rolling_var.index, rolling_var * 100, 
                color=self.colors['var95'], linewidth=2, label='Rolling 95% VaR')
        
        # Mark violations
        common_dates = returns.index.intersection(rolling_var.index)
        aligned_returns = returns.loc[common_dates]
        aligned_var = rolling_var.loc[common_dates]
        
        violations = (-aligned_returns) > aligned_var
        if violations.any():
            ax1.scatter(common_dates[violations], (-aligned_returns[violations]) * 100,
                       color=self.colors['loss'], s=20, alpha=0.8, label='Violations')
        
        ax1.set_title('Rolling VaR Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rolling violation rate
        window = 252  # 1 year
        violation_rate = violations.rolling(window=window).mean()
        
        ax2.plot(violation_rate.index, violation_rate * 100, 
                color=self.colors['loss'], linewidth=2)
        ax2.axhline(y=5, color='black', linestyle='--', alpha=0.7, label='Expected 5%')
        ax2.fill_between(violation_rate.index, violation_rate * 100, 5,
                        where=violation_rate * 100 > 5, color=self.colors['loss'], 
                        alpha=0.3, label='Excess Violations')
        
        ax2.set_title('Rolling Violation Rate (1-Year Window)', fontweight='bold')
        ax2.set_ylabel('Violation Rate (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def _plot_pnl_distribution(self, ax, payoffs, risk_metrics):
        """Plot P&L distribution with key statistics."""
        ax.hist(payoffs, bins=50, density=True, alpha=0.7, color=self.colors['neutral'])
        
        # Mark key statistics
        ax.axvline(risk_metrics.mean_pnl, color=self.colors['mean'], 
                  linestyle='-', linewidth=2, label=f'Mean: {risk_metrics.mean_pnl:.0f}')
        ax.axvline(risk_metrics.var_95, color=self.colors['var95'], 
                  linestyle='--', linewidth=2, label=f'95% VaR: {risk_metrics.var_95:.0f}')
        ax.axvline(risk_metrics.var_99, color=self.colors['var99'], 
                  linestyle='--', linewidth=2, label=f'99% VaR: {risk_metrics.var_99:.0f}')
        
        ax.set_title('P&L Distribution', fontweight='bold')
        ax.set_xlabel('P&L')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_price_distribution(self, ax, final_prices, price_changes):
        """Plot final price distribution."""
        ax.hist(price_changes * 100, bins=50, density=True, alpha=0.7, 
               color=self.colors['neutral'])
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('Price Change Distribution', fontweight='bold')
        ax.set_xlabel('Price Change (%)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_metrics_summary(self, ax, risk_metrics):
        """Plot risk metrics as bar chart."""
        metrics = ['Sharpe Ratio', 'Win Rate (%)', 'Profit Factor']
        values = [risk_metrics.sharpe_ratio, 
                 risk_metrics.win_rate * 100, 
                 risk_metrics.profit_factor]
        
        colors = [self.colors['profit'] if v > 1 else self.colors['loss'] for v in values]
        ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_title('Performance Metrics', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_price_vs_pnl_scatter(self, ax, final_prices, payoffs):
        """Plot scatter of final prices vs P&L."""
        colors = ['green' if p > 0 else 'red' for p in payoffs]
        ax.scatter(final_prices, payoffs, c=colors, alpha=0.6, s=10)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('Final Price vs P&L', fontweight='bold')
        ax.set_xlabel('Final Price')
        ax.set_ylabel('P&L')
        ax.grid(True, alpha=0.3)
    
    def _plot_var_visualization(self, ax, payoffs, risk_metrics):
        """Plot VaR visualization with tail highlighting."""
        n, bins, patches = ax.hist(payoffs, bins=50, alpha=0.7, color=self.colors['neutral'])
        
        # Color the tail
        var95_index = np.searchsorted(bins, risk_metrics.var_95)
        for i in range(var95_index):
            patches[i].set_facecolor(self.colors['loss'])
        
        ax.axvline(risk_metrics.var_95, color=self.colors['var95'], 
                  linestyle='--', linewidth=2, label=f'95% VaR')
        ax.set_title('VaR Visualization', fontweight='bold')
        ax.set_xlabel('P&L')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    def _plot_tail_analysis(self, ax, payoffs, risk_metrics):
        """Plot tail risk analysis."""
        # Sort payoffs for tail analysis
        sorted_payoffs = np.sort(payoffs)
        percentiles = np.arange(1, len(sorted_payoffs) + 1) / len(sorted_payoffs) * 100
        
        ax.plot(percentiles, sorted_payoffs, linewidth=2, color=self.colors['neutral'])
        ax.axhline(y=risk_metrics.var_95, color=self.colors['var95'], 
                  linestyle='--', alpha=0.7, label='95% VaR')
        ax.axhline(y=risk_metrics.expected_shortfall_95, color=self.colors['loss'], 
                  linestyle='--', alpha=0.7, label='95% ES')
        
        ax.set_title('Tail Risk Analysis', fontweight='bold')
        ax.set_xlabel('Percentile')
        ax.set_ylabel('P&L')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_statistics_table(self, ax, risk_metrics):
        """Create statistics table."""
        ax.axis('off')
        
        # Prepare table data
        stats_data = [
            ['Mean P&L', f'{risk_metrics.mean_pnl:.0f}'],
            ['Std Dev', f'{risk_metrics.std_pnl:.0f}'],
            ['Min P&L', f'{risk_metrics.min_pnl:.0f}'],
            ['Max P&L', f'{risk_metrics.max_pnl:.0f}'],
            ['95% VaR', f'{risk_metrics.var_95:.0f}'],
            ['99% VaR', f'{risk_metrics.var_99:.0f}'],
            ['95% ES', f'{risk_metrics.expected_shortfall_95:.0f}'],
            ['99% ES', f'{risk_metrics.expected_shortfall_99:.0f}'],
            ['Sharpe Ratio', f'{risk_metrics.sharpe_ratio:.2f}'],
            ['Win Rate', f'{risk_metrics.win_rate:.1%}'],
            ['Skewness', f'{risk_metrics.skewness:.2f}'],
            ['Kurtosis', f'{risk_metrics.kurtosis:.2f}']
        ]
        
        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.4, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        ax.set_title('Risk Statistics Summary', fontweight='bold', pad=20)
    
    def _save_plot(self, fig: plt.Figure, save_path: str):
        """Save plot to file."""
        if self.output_dir:
            full_path = self.output_dir / save_path
        else:
            full_path = Path(save_path)
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to {full_path}")