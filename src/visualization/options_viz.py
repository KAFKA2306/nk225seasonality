"""
Options visualization module.

This module provides visualizations for options strategies including
payoff diagrams, Greeks surfaces, and strategy performance charts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..config import get_logger
from ..options.strategies import StrategyDefinition, OptionType


class OptionsVisualizer:
    """Visualizations for options strategies and analysis."""
    
    def __init__(self, 
                 output_dir: Optional[Path] = None,
                 figsize: Tuple[int, int] = (12, 8)):
        """Initialize options visualizer."""
        self.output_dir = output_dir
        self.figsize = figsize
        self.logger = get_logger(__name__)
        
        # Color scheme
        self.colors = {
            'profit': '#2ca02c',
            'loss': '#d62728', 
            'breakeven': '#ff7f0e',
            'payoff': '#1f77b4',
            'current': '#9467bd'
        }
    
    def create_payoff_diagram(self, 
                            strategy: StrategyDefinition,
                            current_price: float,
                            price_range_pct: float = 0.3,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create options strategy payoff diagram.
        
        Args:
            strategy: Strategy definition
            current_price: Current underlying price
            price_range_pct: Price range as percentage of current price
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        # Generate price range
        price_min = current_price * (1 - price_range_pct)
        price_max = current_price * (1 + price_range_pct)
        prices = np.linspace(price_min, price_max, 200)
        
        # Calculate payoffs
        payoffs = self._calculate_strategy_payoffs(strategy, prices, current_price)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot payoff line
        ax.plot(prices, payoffs, linewidth=3, color=self.colors['payoff'], label='Strategy P&L')
        
        # Fill profit/loss areas
        profit_mask = payoffs > 0
        loss_mask = payoffs < 0
        
        ax.fill_between(prices, payoffs, 0, where=profit_mask, 
                       color=self.colors['profit'], alpha=0.3, label='Profit Zone')
        ax.fill_between(prices, payoffs, 0, where=loss_mask,
                       color=self.colors['loss'], alpha=0.3, label='Loss Zone')
        
        # Mark breakeven points
        breakevens = self._find_breakeven_points(prices, payoffs)
        for be in breakevens:
            ax.axvline(x=be, color=self.colors['breakeven'], 
                      linestyle='--', alpha=0.7, linewidth=2)
            ax.text(be, max(payoffs) * 0.1, f'BE: {be:.0f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        # Mark current price
        ax.axvline(x=current_price, color=self.colors['current'], 
                  linestyle='-', alpha=0.8, linewidth=2, label=f'Current: {current_price:.0f}')
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('Underlying Price', fontsize=12)
        ax.set_ylabel('Profit/Loss', fontsize=12)
        ax.set_title(f'{strategy.strategy_type.value.replace("_", " ").title()} Strategy Payoff', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add strategy details
        max_profit = np.max(payoffs)
        max_loss = np.min(payoffs)
        
        details_text = f'Max Profit: {max_profit:.0f}\nMax Loss: {max_loss:.0f}'
        ax.text(0.02, 0.98, details_text, transform=ax.transAxes, 
               va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_greeks_surface(self, 
                            greeks_data: Dict[str, np.ndarray],
                            greek: str = 'delta',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 3D surface plot for option Greeks.
        
        Args:
            greeks_data: Greeks surface data
            greek: Greek to plot ('delta', 'gamma', 'theta', 'vega')
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get data
        spot_grid = greeks_data['spot_grid']
        vol_grid = greeks_data['vol_grid'] 
        greek_surface = greeks_data[greek]
        
        # Create surface plot
        surf = ax.plot_surface(spot_grid, vol_grid * 100, greek_surface,
                              cmap='viridis', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Labels
        ax.set_xlabel('Underlying Price')
        ax.set_ylabel('Volatility (%)')
        ax.set_zlabel(greek.title())
        ax.set_title(f'{greek.title()} Surface', fontsize=14, fontweight='bold')
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_volatility_smile(self, 
                              strikes: np.ndarray,
                              implied_vols: np.ndarray,
                              current_price: float,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create volatility smile plot.
        
        Args:
            strikes: Strike prices
            implied_vols: Implied volatilities
            current_price: Current underlying price
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        # Calculate moneyness
        moneyness = strikes / current_price
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot volatility smile
        ax.plot(moneyness, implied_vols * 100, 'o-', linewidth=2, 
               markersize=8, color=self.colors['payoff'])
        
        # Mark ATM
        ax.axvline(x=1.0, color=self.colors['current'], 
                  linestyle='--', alpha=0.7, label='ATM')
        
        # Labels
        ax.set_xlabel('Moneyness (K/S)', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.set_title('Volatility Smile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_strategy_comparison(self, 
                                 strategies: Dict[str, Dict[str, Any]],
                                 current_price: float,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple strategies on one plot.
        
        Args:
            strategies: Dictionary of strategy names and payoff data
            current_price: Current underlying price
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        
        for i, (name, data) in enumerate(strategies.items()):
            prices = data['prices']
            payoffs = data['payoffs']
            
            ax.plot(prices, payoffs, linewidth=2, 
                   color=colors[i], label=name)
        
        # Mark current price and zero line
        ax.axvline(x=current_price, color='black', 
                  linestyle='--', alpha=0.7, label=f'Current: {current_price:.0f}')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Labels
        ax.set_xlabel('Underlying Price', fontsize=12)
        ax.set_ylabel('Profit/Loss', fontsize=12)
        ax.set_title('Strategy Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_backtest_performance(self, 
                                  backtest_results: Dict[str, Any],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create strategy backtest performance visualization.
        
        Args:
            backtest_results: Backtest results
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        trades = backtest_results['individual_trades']
        
        if not trades:
            raise ValueError("No trades to plot")
        
        # Extract data
        entry_dates = [trade['entry_date'] for trade in trades]
        returns = [trade['return_pct'] * 100 for trade in trades]
        cumulative_returns = np.cumsum(returns)
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Individual trade returns
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax1.bar(range(len(returns)), returns, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('Individual Trade Returns', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative performance
        ax2.plot(range(len(cumulative_returns)), cumulative_returns, 
                linewidth=2, color=self.colors['payoff'])
        ax2.fill_between(range(len(cumulative_returns)), cumulative_returns, 0,
                        alpha=0.3, color=self.colors['payoff'])
        ax2.set_title('Cumulative Performance', fontweight='bold')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Return distribution
        ax3.hist(returns, bins=20, alpha=0.7, color=self.colors['payoff'], density=True)
        ax3.axvline(x=np.mean(returns), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        ax3.set_title('Return Distribution', fontweight='bold')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add performance metrics
        metrics_text = (f"Total Return: {backtest_results['total_return']*100:.2f}%\n"
                       f"Win Rate: {backtest_results['win_rate']*100:.1f}%\n"
                       f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}\n"
                       f"Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
        
        ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes,
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_risk_metrics_chart(self, 
                                risk_metrics: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create risk metrics visualization.
        
        Args:
            risk_metrics: Risk metrics data
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # VaR and Expected Shortfall
        confidence_levels = [90, 95, 99]
        var_values = [risk_metrics.get(f'var_{cl}', 0) for cl in confidence_levels]
        es_values = [risk_metrics.get(f'es_{cl}', 0) for cl in confidence_levels]
        
        x = np.arange(len(confidence_levels))
        width = 0.35
        
        ax1.bar(x - width/2, var_values, width, label='VaR', color=self.colors['loss'], alpha=0.7)
        ax1.bar(x + width/2, es_values, width, label='Expected Shortfall', color=self.colors['payoff'], alpha=0.7)
        ax1.set_xlabel('Confidence Level (%)')
        ax1.set_ylabel('Loss Amount')
        ax1.set_title('Value at Risk & Expected Shortfall', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(confidence_levels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # P&L distribution
        if 'pnl_distribution' in risk_metrics:
            ax2.hist(risk_metrics['pnl_distribution'], bins=50, alpha=0.7, 
                    color=self.colors['payoff'], density=True)
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax2.axvline(x=np.mean(risk_metrics['pnl_distribution']), 
                       color='red', linestyle='--', label='Mean')
            ax2.set_title('P&L Distribution', fontweight='bold')
            ax2.set_xlabel('P&L')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Performance ratios
        ratios = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
        ratio_values = [risk_metrics.get(ratio, 0) for ratio in ratios]
        ratio_labels = ['Sharpe', 'Sortino', 'Profit Factor']
        
        colors = ['green' if v > 1 else 'red' for v in ratio_values]
        ax3.bar(ratio_labels, ratio_values, color=colors, alpha=0.7)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Benchmark')
        ax3.set_title('Performance Ratios', fontweight='bold')
        ax3.set_ylabel('Ratio Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Risk measures
        risk_measures = ['max_drawdown', 'volatility', 'downside_deviation']
        risk_values = [risk_metrics.get(measure, 0) for measure in risk_measures]
        risk_labels = ['Max DD', 'Volatility', 'Downside Dev']
        
        ax4.bar(risk_labels, risk_values, color=self.colors['loss'], alpha=0.7)
        ax4.set_title('Risk Measures', fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def _calculate_strategy_payoffs(self, 
                                  strategy: StrategyDefinition,
                                  prices: np.ndarray,
                                  current_price: float) -> np.ndarray:
        """Calculate strategy payoffs for given price range."""
        
        payoffs = np.zeros(len(prices))
        initial_cost = 0.0  # Simplified - would need actual option pricing
        
        for price in range(len(prices)):
            total_payoff = 0.0
            
            for leg in strategy.legs:
                if leg.option_type == OptionType.CALL:
                    intrinsic_value = max(prices[price] - leg.strike_price, 0)
                else:
                    intrinsic_value = max(leg.strike_price - prices[price], 0)
                
                total_payoff += leg.position * leg.quantity * intrinsic_value
            
            payoffs[price] = total_payoff - initial_cost
        
        return payoffs
    
    def _find_breakeven_points(self, prices: np.ndarray, payoffs: np.ndarray) -> List[float]:
        """Find breakeven points where payoff crosses zero."""
        
        breakeven_points = []
        
        for i in range(len(payoffs) - 1):
            if payoffs[i] * payoffs[i + 1] < 0:  # Sign change
                # Linear interpolation
                breakeven = prices[i] - payoffs[i] * (prices[i + 1] - prices[i]) / (payoffs[i + 1] - payoffs[i])
                breakeven_points.append(breakeven)
        
        return breakeven_points
    
    def _save_plot(self, fig: plt.Figure, save_path: str):
        """Save plot to file."""
        if self.output_dir:
            full_path = self.output_dir / save_path
        else:
            full_path = Path(save_path)
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to {full_path}")