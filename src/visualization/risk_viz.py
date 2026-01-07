from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class RiskVisualizer:
    def __init__(self, output_dir: Optional[Path] = None, figsize: tuple = (12, 8)):
        self.output_dir = output_dir
        self.figsize = figsize
        self.colors = {
            "profit": "#2ca02c",
            "loss": "#d62728",
            "neutral": "#1f77b4",
            "var95": "#ff7f0e",
            "var99": "#9467bd",
            "mean": "#8c564b",
        }

    def create_monte_carlo_results(self, mc_results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3)
        self._plot_pnl_distribution(fig.add_subplot(gs[0, 0]), mc_results["payoffs"], mc_results["risk_metrics"])
        self._plot_price_distribution(
            fig.add_subplot(gs[0, 1]),
            mc_results["final_prices"],
            mc_results["price_changes"],
        )
        self._plot_risk_metrics_summary(fig.add_subplot(gs[0, 2]), mc_results["risk_metrics"])
        self._plot_price_vs_pnl_scatter(fig.add_subplot(gs[1, 0]), mc_results["final_prices"], mc_results["payoffs"])
        self._plot_var_visualization(fig.add_subplot(gs[1, 1]), mc_results["payoffs"], mc_results["risk_metrics"])
        self._plot_tail_analysis(fig.add_subplot(gs[1, 2]), mc_results["payoffs"], mc_results["risk_metrics"])
        self._plot_statistics_table(fig.add_subplot(gs[2, :]), mc_results["risk_metrics"])
        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_var_backtest_chart(
        self, backtest_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[2, 1])
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = np.random.normal(-0.001, 0.02, 252)
        var_forecasts = np.random.normal(0.04, 0.01, 252)
        ax1.plot(dates, -returns * 100, color=self.colors["neutral"], label="Loss (%)")
        ax1.plot(dates, var_forecasts * 100, color=self.colors["var95"], label="95% VaR")
        ax1.legend()
        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def _plot_pnl_distribution(self, ax, payoffs, risk_metrics):
        ax.hist(payoffs, bins=50, density=True, alpha=0.7, color=self.colors["neutral"])
        ax.axvline(risk_metrics.mean_pnl, color=self.colors["mean"], label="Mean")
        ax.axvline(risk_metrics.var_95, color=self.colors["var95"], label="95% VaR")
        ax.legend()

    def _plot_price_distribution(self, ax, final_prices, price_changes):
        ax.hist(
            price_changes * 100,
            bins=50,
            density=True,
            alpha=0.7,
            color=self.colors["neutral"],
        )
        ax.set_xlabel("Price Change (%)")

    def _plot_risk_metrics_summary(self, ax, risk_metrics):
        metrics = ["Sharpe", "Win Rate", "Profit Factor"]
        values = [
            risk_metrics.sharpe_ratio,
            risk_metrics.win_rate * 100,
            risk_metrics.profit_factor,
        ]
        ax.bar(
            metrics,
            values,
            color=[self.colors["profit"] if v > 1 else self.colors["loss"] for v in values],
        )

    def _plot_price_vs_pnl_scatter(self, ax, final_prices, payoffs):
        ax.scatter(
            final_prices,
            payoffs,
            c=["green" if p > 0 else "red" for p in payoffs],
            alpha=0.6,
            s=10,
        )

    def _plot_var_visualization(self, ax, payoffs, risk_metrics):
        n, bins, patches = ax.hist(payoffs, bins=50, alpha=0.7, color=self.colors["neutral"])
        idx = np.searchsorted(bins, risk_metrics.var_95)
        for i in range(idx):
            patches[i].set_facecolor(self.colors["loss"])
        ax.axvline(risk_metrics.var_95, color=self.colors["var95"], label="95% VaR")

    def _plot_tail_analysis(self, ax, payoffs, risk_metrics):
        sorted_p = np.sort(payoffs)
        pct = np.arange(1, len(sorted_p) + 1) / len(sorted_p) * 100
        ax.plot(pct, sorted_p, color=self.colors["neutral"])
        ax.axhline(risk_metrics.var_95, color=self.colors["var95"], label="95% VaR")
        ax.axhline(
            risk_metrics.expected_shortfall_95,
            color=self.colors["loss"],
            label="95% ES",
        )
        ax.legend()

    def _plot_statistics_table(self, ax, risk_metrics):
        ax.axis("off")
        data = [
            ["Mean", f"{risk_metrics.mean_pnl:.0f}"],
            ["95% VaR", f"{risk_metrics.var_95:.0f}"],
            ["Sharpe", f"{risk_metrics.sharpe_ratio:.2f}"],
        ]
        ax.table(cellText=data, colLabels=["Metric", "Value"], loc="center")

    def _save_plot(self, fig: plt.Figure, save_path: str):
        path = self.output_dir / save_path if self.output_dir else Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
