"""
Seasonality visualization module.

This module provides publication-quality visualizations for seasonal analysis
including heatmaps, time series plots, and statistical charts.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path

from ..config import get_logger


class SeasonalityVisualizer:
    """Generate publication-quality seasonality visualizations."""
    
    def __init__(self, 
                 output_dir: Optional[Path] = None,
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        """
        Initialize seasonality visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style
            figsize: Default figure size
            dpi: Resolution for saved plots
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.logger = get_logger(__name__)
        
        # Set style
        plt.style.use(style)
        
        # Color palette for consistency
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def create_seasonal_heatmap(self, 
                               seasonality_results: Dict[int, Any],
                               metric: str = 'mean_return',
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create monthly seasonality heatmap.
        
        Args:
            seasonality_results: Results from seasonality analysis
            metric: Metric to visualize ('mean_return', 'std_return', 't_pvalue')
            title: Custom title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        # Prepare data for heatmap
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Extract data
        data_values = []
        significance_markers = []
        
        for month in range(1, 13):
            if month in seasonality_results:
                result = seasonality_results[month]
                
                if metric == 'mean_return':
                    value = result.mean_return * 100  # Convert to percentage
                elif metric == 'std_return':
                    value = result.std_return * 100
                elif metric == 't_pvalue':
                    value = result.t_pvalue
                else:
                    value = getattr(result, metric, 0)
                
                data_values.append(value)
                
                # Check significance for markers
                if result.is_significant:
                    if result.t_pvalue < 0.01:
                        significance_markers.append('**')
                    else:
                        significance_markers.append('*')
                else:
                    significance_markers.append('')
            else:
                data_values.append(0)
                significance_markers.append('')
        
        # Create DataFrame for heatmap
        heatmap_data = pd.DataFrame([data_values], columns=month_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Choose colormap based on metric
        if metric == 'mean_return':
            cmap = 'RdYlGn'
            center = 0
            fmt = '.2f'
        elif metric in ['std_return', 't_pvalue']:
            cmap = 'viridis'
            center = None
            fmt = '.3f'
        else:
            cmap = 'viridis'
            center = None
            fmt = '.2f'
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True,
                   fmt=fmt,
                   cmap=cmap,
                   center=center,
                   ax=ax,
                   cbar_kws={'label': self._get_metric_label(metric)})
        
        # Add significance markers
        for i, marker in enumerate(significance_markers):
            if marker:
                ax.text(i + 0.5, 0.8, marker, 
                       ha='center', va='center',
                       color='black', fontsize=16, fontweight='bold')
        
        # Customize plot
        if title is None:
            title = f'Nikkei 225 Monthly {self._get_metric_title(metric)}'
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('')
        
        # Add legend for significance
        ax.text(1.02, 0.5, '** p < 0.01\n* p < 0.05', 
               transform=ax.transAxes, va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_returns_distribution_plot(self, 
                                       data: pd.DataFrame,
                                       months_to_highlight: Optional[List[int]] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create distribution plots for different months.
        
        Args:
            data: Market data with returns
            months_to_highlight: Specific months to highlight
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        # Calculate returns if not present
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close_price'].pct_change()
        
        data['month'] = data.index.month
        
        # Create subplots
        if months_to_highlight:
            n_months = len(months_to_highlight)
            cols = min(3, n_months)
            rows = (n_months - 1) // cols + 1
        else:
            months_to_highlight = list(range(1, 13))
            cols = 4
            rows = 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for i, month in enumerate(months_to_highlight):
            if i >= len(axes.flat):
                break
                
            ax = axes.flat[i]
            month_data = data[data['month'] == month]['returns'].dropna()
            
            if len(month_data) > 0:
                # Histogram
                ax.hist(month_data * 100, bins=30, alpha=0.7, 
                       color=self.colors['primary'], density=True)
                
                # Fit normal distribution
                mu, sigma = month_data.mean() * 100, month_data.std() * 100
                x = np.linspace(month_data.min() * 100, month_data.max() * 100, 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 
                       'r-', linewidth=2, label='Normal fit')
                
                # Add statistics text
                ax.axvline(mu, color='red', linestyle='--', alpha=0.7)
                ax.text(0.05, 0.95, f'μ = {mu:.2f}%\nσ = {sigma:.2f}%\nn = {len(month_data)}',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'{month_names[month]} Returns Distribution')
                ax.set_xlabel('Returns (%)')
                ax.set_ylabel('Density')
                ax.legend()
        
        # Hide empty subplots
        for i in range(len(months_to_highlight), len(axes.flat)):
            axes.flat[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_seasonal_time_series(self, 
                                  data: pd.DataFrame,
                                  highlight_months: Optional[List[int]] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create time series plot with seasonal highlights.
        
        Args:
            data: Market data
            highlight_months: Months to highlight
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      height_ratios=[3, 1], sharex=True)
        
        # Price time series
        price_col = 'close_price' if 'close_price' in data.columns else 'adjusted_close'
        ax1.plot(data.index, data[price_col], color=self.colors['primary'], linewidth=1)
        
        # Highlight specific months
        if highlight_months:
            for month in highlight_months:
                month_mask = data.index.month == month
                ax1.scatter(data.index[month_mask], data[price_col][month_mask], 
                           alpha=0.6, s=10, color=self.colors['danger'])
        
        ax1.set_title('Nikkei 225 Price with Seasonal Highlights', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Returns time series
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data[price_col].pct_change()
        
        ax2.plot(data.index, data['returns'] * 100, color=self.colors['secondary'], 
                linewidth=0.5, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Highlight returns in specific months
        if highlight_months:
            for month in highlight_months:
                month_mask = data.index.month == month
                ax2.scatter(data.index[month_mask], data['returns'][month_mask] * 100,
                           alpha=0.6, s=10, color=self.colors['danger'])
        
        ax2.set_ylabel('Returns (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_minor_locator(mdates.MonthLocator())
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_rolling_seasonality_plot(self, 
                                      rolling_results: Dict[str, Any],
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create plot showing evolution of seasonal patterns over time.
        
        Args:
            rolling_results: Results from rolling seasonality analysis
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        windows = rolling_results['rolling_windows']
        consistency_analysis = rolling_results['consistency_analysis']
        
        # Extract data for plotting
        dates = [w['window_end'] for w in windows]
        
        # Get monthly means for each window
        monthly_means = {month: [] for month in range(1, 13)}
        
        for window in windows:
            for month in range(1, 13):
                if month in window['monthly_means']:
                    monthly_means[month].append(window['monthly_means'][month] * 100)
                else:
                    monthly_means[month].append(np.nan)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Plot rolling means for each month
        for month in range(1, 13):
            if month in consistency_analysis['consistent_months']:
                alpha = 1.0
                linewidth = 2
            else:
                alpha = 0.3
                linewidth = 1
            
            ax.plot(dates, monthly_means[month], 
                   label=month_names[month], alpha=alpha, linewidth=linewidth)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Rolling Monthly Returns Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Mean Return (%)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_statistical_summary_plot(self, 
                                      seasonality_results: Dict[int, Any],
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive statistical summary plot.
        
        Args:
            seasonality_results: Results from seasonality analysis
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        # Prepare data
        months = list(seasonality_results.keys())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        means = [seasonality_results[m].mean_return * 100 for m in months]
        stds = [seasonality_results[m].std_return * 100 for m in months]
        t_stats = [seasonality_results[m].t_statistic for m in months]
        p_values = [seasonality_results[m].t_pvalue for m in months]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean returns bar plot
        colors = ['green' if mean > 0 else 'red' for mean in means]
        bars1 = ax1.bar([month_names[m-1] for m in months], means, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Mean Monthly Returns', fontweight='bold')
        ax1.set_ylabel('Mean Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                    f'{mean:.2f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # Volatility plot
        ax2.bar([month_names[m-1] for m in months], stds, color=self.colors['info'], alpha=0.7)
        ax2.set_title('Monthly Volatility', fontweight='bold')
        ax2.set_ylabel('Standard Deviation (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # t-statistics plot
        significant_mask = [seasonality_results[m].is_significant for m in months]
        colors = [self.colors['success'] if sig else self.colors['primary'] for sig in significant_mask]
        ax3.bar([month_names[m-1] for m in months], t_stats, color=colors, alpha=0.7)
        ax3.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='5% significance')
        ax3.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('t-Statistics', fontweight='bold')
        ax3.set_ylabel('t-statistic')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # P-values plot
        ax4.bar([month_names[m-1] for m in months], p_values, color=self.colors['warning'], alpha=0.7)
        ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% threshold')
        ax4.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='1% threshold')
        ax4.set_title('P-values', fontweight='bold')
        ax4.set_ylabel('p-value')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def create_correlation_matrix(self, 
                                 data: pd.DataFrame,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation matrix of monthly returns.
        
        Args:
            data: Market data
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        # Calculate returns if not present
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close_price'].pct_change()
        
        # Create monthly return series
        data['month'] = data.index.month
        data['year'] = data.index.year
        
        # Pivot to get monthly returns by year
        monthly_pivot = data.pivot_table(
            values='returns', 
            index='year', 
            columns='month',
            aggfunc='mean'
        )
        
        # Calculate correlation matrix
        corr_matrix = monthly_pivot.corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Rename columns
        corr_matrix.columns = [month_names[i-1] for i in corr_matrix.columns]
        corr_matrix.index = [month_names[i-1] for i in corr_matrix.index]
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   ax=ax,
                   fmt='.2f')
        
        ax.set_title('Monthly Returns Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def _get_metric_label(self, metric: str) -> str:
        """Get label for metric."""
        labels = {
            'mean_return': 'Mean Return (%)',
            'std_return': 'Volatility (%)',
            't_pvalue': 'p-value',
            't_statistic': 't-statistic'
        }
        return labels.get(metric, metric.replace('_', ' ').title())
    
    def _get_metric_title(self, metric: str) -> str:
        """Get title for metric."""
        titles = {
            'mean_return': 'Seasonality Pattern',
            'std_return': 'Volatility Pattern',
            't_pvalue': 'Statistical Significance',
            't_statistic': 't-Statistics'
        }
        return titles.get(metric, metric.replace('_', ' ').title())
    
    def _save_plot(self, fig: plt.Figure, save_path: str):
        """Save plot to file."""
        if self.output_dir:
            full_path = self.output_dir / save_path
        else:
            full_path = Path(save_path)
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
        self.logger.info(f"Plot saved to {full_path}")
    
    def create_dashboard(self, 
                        data: pd.DataFrame,
                        seasonality_results: Dict[int, Any],
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive seasonality dashboard.
        
        Args:
            data: Market data
            seasonality_results: Seasonality analysis results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        
        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Define grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Main heatmap (top left, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._create_heatmap_subplot(ax1, seasonality_results, 'mean_return')
        
        # Statistical summary (top right, spanning 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._create_stats_subplot(ax2, seasonality_results)
        
        # Time series (middle, spanning all columns)
        ax3 = fig.add_subplot(gs[1, :])
        self._create_timeseries_subplot(ax3, data)
        
        # Distribution plots (bottom row)
        significant_months = [m for m, r in seasonality_results.items() if r.is_significant][:4]
        for i, month in enumerate(significant_months):
            ax = fig.add_subplot(gs[2, i])
            self._create_distribution_subplot(ax, data, month)
        
        # Main title
        fig.suptitle('Nikkei 225 Seasonality Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def _create_heatmap_subplot(self, ax, seasonality_results, metric):
        """Create heatmap subplot."""
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        data_values = []
        for month in range(1, 13):
            if month in seasonality_results:
                value = getattr(seasonality_results[month], metric) * 100
                data_values.append(value)
            else:
                data_values.append(0)
        
        heatmap_data = pd.DataFrame([data_values], columns=month_names)
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Monthly Mean Returns (%)', fontweight='bold')
    
    def _create_stats_subplot(self, ax, seasonality_results):
        """Create statistics subplot."""
        months = list(seasonality_results.keys())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        t_stats = [seasonality_results[m].t_statistic for m in months]
        significant = [seasonality_results[m].is_significant for m in months]
        
        colors = ['green' if sig else 'gray' for sig in significant]
        ax.bar([month_names[m-1] for m in months], t_stats, color=colors, alpha=0.7)
        ax.axhline(y=1.96, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
        ax.set_title('Statistical Significance (t-statistics)', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    
    def _create_timeseries_subplot(self, ax, data):
        """Create time series subplot."""
        price_col = 'close_price' if 'close_price' in data.columns else 'adjusted_close'
        ax.plot(data.index, data[price_col], linewidth=1, color=self.colors['primary'])
        ax.set_title('Price Time Series', fontweight='bold')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
    
    def _create_distribution_subplot(self, ax, data, month):
        """Create distribution subplot for specific month."""
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close_price'].pct_change()
        
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        month_data = data[data.index.month == month]['returns'].dropna() * 100
        
        if len(month_data) > 0:
            ax.hist(month_data, bins=20, alpha=0.7, color=self.colors['primary'], density=True)
            
            # Add statistics
            mu, sigma = month_data.mean(), month_data.std()
            ax.axvline(mu, color='red', linestyle='--', alpha=0.7)
            ax.set_title(f'{month_names[month]} Returns', fontweight='bold')
            ax.set_xlabel('Return (%)')
            ax.text(0.05, 0.95, f'μ={mu:.2f}%', transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))