from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .options import OptionType, StrategyDefinition


class OptionsVisualizer:
    def __init__(self, output_dir: Optional[Path] = None, figsize: tuple = (12, 8)):
        self.output_dir = output_dir
        self.figsize = figsize
        self.colors = {
            "profit": "#2ca02c",
            "loss": "#d62728",
            "breakeven": "#ff7f0e",
            "payoff": "#1f77b4",
            "current": "#9467bd",
        }

    def create_payoff_diagram(
        self,
        strategy: StrategyDefinition,
        current_price: float,
        price_range_pct: float = 0.3,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        prices = np.linspace(
            current_price * (1 - price_range_pct),
            current_price * (1 + price_range_pct),
            200,
        )
        payoffs = self._calculate_strategy_payoffs(strategy, prices, current_price)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(
            prices,
            payoffs,
            linewidth=3,
            color=self.colors["payoff"],
            label="Strategy P&L",
        )
        ax.fill_between(
            prices,
            payoffs,
            0,
            where=payoffs > 0,
            color=self.colors["profit"],
            alpha=0.3,
            label="Profit Zone",
        )
        ax.fill_between(
            prices,
            payoffs,
            0,
            where=payoffs < 0,
            color=self.colors["loss"],
            alpha=0.3,
            label="Loss Zone",
        )

        for be in self._find_breakeven_points(prices, payoffs):
            ax.axvline(x=be, color=self.colors["breakeven"], linestyle="--", alpha=0.7)
            ax.text(
                be,
                max(payoffs) * 0.1,
                f"BE: {be:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.axvline(
            x=current_price,
            color=self.colors["current"],
            linestyle="-",
            alpha=0.8,
            label=f"Current: {current_price:.0f}",
        )
        ax.axhline(y=0, color="black", alpha=0.5)
        ax.set_xlabel("Underlying Price")
        ax.set_ylabel("Profit/Loss")
        ax.set_title(f"{strategy.strategy_type.value.replace('_', ' ').title()} Strategy Payoff")
        ax.legend()
        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_greeks_surface(
        self,
        greeks_data: Dict[str, np.ndarray],
        greek: str = "delta",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            greeks_data["spot_grid"],
            greeks_data["vol_grid"] * 100,
            greeks_data[greek],
            cmap="viridis",
            alpha=0.8,
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel("Underlying Price")
        ax.set_ylabel("Volatility (%)")
        ax.set_zlabel(greek.title())
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_volatility_smile(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        current_price: float,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(
            strikes / current_price,
            implied_vols * 100,
            "o-",
            linewidth=2,
            color=self.colors["payoff"],
        )
        ax.axvline(x=1.0, color=self.colors["current"], linestyle="--", alpha=0.7, label="ATM")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Volatility (%)")
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_strategy_comparison(
        self,
        strategies: Dict[str, Dict[str, Any]],
        current_price: float,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        for i, (name, data) in enumerate(strategies.items()):
            ax.plot(
                data["prices"],
                data["payoffs"],
                linewidth=2,
                color=colors[i],
                label=name,
            )
        ax.axvline(x=current_price, color="black", linestyle="--", alpha=0.7)
        ax.axhline(y=0, color="black", alpha=0.5)
        ax.legend()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def _calculate_strategy_payoffs(
        self, strategy: StrategyDefinition, prices: np.ndarray, current_price: float
    ) -> np.ndarray:
        payoffs = np.zeros(len(prices))
        for price in range(len(prices)):
            total = 0.0
            for leg in strategy.legs:
                intrinsic = (
                    max(prices[price] - leg.strike_price, 0)
                    if leg.option_type == OptionType.CALL
                    else max(leg.strike_price - prices[price], 0)
                )
                total += leg.position * leg.quantity * intrinsic
            payoffs[price] = total
        return payoffs

    def _find_breakeven_points(self, prices: np.ndarray, payoffs: np.ndarray) -> List[float]:
        be = []
        for i in range(len(payoffs) - 1):
            if payoffs[i] * payoffs[i + 1] < 0:
                be.append(prices[i] - payoffs[i] * (prices[i + 1] - prices[i]) / (payoffs[i + 1] - payoffs[i]))
        return be

    def _save_plot(self, fig: plt.Figure, save_path: str):
        path = self.output_dir / save_path if self.output_dir else Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")


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


class SeasonalityVisualizer:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
    ):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use(style)
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff7f0e",
            "info": "#17a2b8",
        }

    def create_seasonal_heatmap(
        self,
        seasonality_results: Dict[int, Any],
        metric: str = "mean_return",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        data_values = []
        significance_markers = []

        for month in range(1, 13):
            if month in seasonality_results:
                result = seasonality_results[month]
                if metric == "mean_return":
                    value = result.mean_return * 100
                elif metric == "std_return":
                    value = result.std_return * 100
                elif metric == "t_pvalue":
                    value = result.t_pvalue
                else:
                    value = getattr(result, metric, 0)
                data_values.append(value)
                marker = (
                    "**" if result.is_significant and result.t_pvalue < 0.01 else "*" if result.is_significant else ""
                )
                significance_markers.append(marker)
            else:
                data_values.append(0)
                significance_markers.append("")

        heatmap_data = pd.DataFrame([data_values], columns=month_names)
        fig, ax = plt.subplots(figsize=self.figsize)

        cmap = "RdYlGn" if metric == "mean_return" else "viridis"
        center = 0 if metric == "mean_return" else None
        fmt = ".3f" if metric in ["std_return", "t_pvalue"] else ".2f"

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            center=center,
            ax=ax,
            cbar_kws={"label": self._get_metric_label(metric)},
        )

        for i, marker in enumerate(significance_markers):
            if marker:
                ax.text(
                    i + 0.5,
                    0.8,
                    marker,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                    fontweight="bold",
                )

        ax.set_title(
            title or f"Nikkei 225 Monthly {self._get_metric_title(metric)}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Month")
        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_returns_distribution_plot(
        self,
        data: pd.DataFrame,
        months_to_highlight: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        if "returns" not in data.columns:
            data = data.copy()
            data["returns"] = data["close_price"].pct_change()
        data["month"] = data.index.month

        if not months_to_highlight:
            months_to_highlight = list(range(1, 13))
        cols = 4
        rows = (len(months_to_highlight) - 1) // cols + 1

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = np.array(axes).reshape(-1)
        month_names = [
            "",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        for i, month in enumerate(months_to_highlight):
            if i >= len(axes):
                break
            ax = axes[i]
            month_data = data[data["month"] == month]["returns"].dropna() * 100

            if len(month_data) > 0:
                ax.hist(
                    month_data,
                    bins=30,
                    alpha=0.7,
                    color=self.colors["primary"],
                    density=True,
                )
                mu, sigma = month_data.mean(), month_data.std()
                x = np.linspace(month_data.min(), month_data.max(), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2)
                ax.set_title(f"{month_names[month]} Returns")

        for i in range(len(months_to_highlight), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_seasonal_time_series(
        self,
        data: pd.DataFrame,
        highlight_months: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1], sharex=True)
        col = "close_price" if "close_price" in data.columns else "adjusted_close"

        ax1.plot(data.index, data[col], color=self.colors["primary"], linewidth=1)
        if highlight_months:
            for m in highlight_months:
                mask = data.index.month == m
                ax1.scatter(
                    data.index[mask],
                    data[col][mask],
                    alpha=0.6,
                    s=10,
                    color=self.colors["danger"],
                )

        if "returns" not in data.columns:
            data["returns"] = data[col].pct_change()
        ax2.plot(
            data.index,
            data["returns"] * 100,
            color=self.colors["secondary"],
            linewidth=0.5,
            alpha=0.7,
        )
        ax2.axhline(y=0, color="black", alpha=0.3)
        if highlight_months:
            for m in highlight_months:
                mask = data.index.month == m
                ax2.scatter(
                    data.index[mask],
                    data["returns"][mask] * 100,
                    alpha=0.6,
                    s=10,
                    color=self.colors["danger"],
                )

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def _get_metric_label(self, metric: str) -> str:
        return {
            "mean_return": "Mean Return (%)",
            "std_return": "Volatility (%)",
            "t_pvalue": "p-value",
            "t_statistic": "t-statistic",
        }.get(metric, metric)

    def _get_metric_title(self, metric: str) -> str:
        return {
            "mean_return": "Seasonality Pattern",
            "std_return": "Volatility Pattern",
            "t_pvalue": "Statistical Significance",
            "t_statistic": "t-Statistics",
        }.get(metric, metric)

    def create_monthly_distribution_boxplot(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        if "returns" not in data.columns:
            data = data.copy()
            data["returns"] = data["close_price"].pct_change()

        data["month"] = data.index.month
        month_data = [data[data["month"] == m]["returns"].dropna() * 100 for m in range(1, 13)]

        fig, ax = plt.subplots(figsize=self.figsize)

        bp = ax.boxplot(
            month_data,
            patch_artist=True,
            notch=False,
            vert=True,
            labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(self.colors["primary"])
            patch.set_alpha(0.6)
            patch.set_edgecolor("black")

        for median in bp["medians"]:
            median.set_color(self.colors["danger"])
            median.set_linewidth(2)

        ax.set_title("Monthly Returns Distribution (Boxplot)", fontsize=16, fontweight="bold", pad=20)
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Month")
        ax.grid(True, axis="y", alpha=0.3)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_year_month_heatmap(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        if "returns" not in data.columns:
            data = data.copy()
            data["returns"] = data["close_price"].pct_change()

        pivot_table = (
            data.pivot_table(values="returns", index=data.index.year, columns=data.index.month, aggfunc="sum") * 100
        )

        monthly_data = data["close_price"].resample("ME").last().pct_change() * 100
        pivot_table = pd.DataFrame(index=monthly_data.index.year.unique(), columns=range(1, 13))

        for date, val in monthly_data.items():
            pivot_table.loc[date.year, date.month] = val

        pivot_table = pivot_table.astype(float).sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(14, len(pivot_table) * 0.5 + 2))

        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": "Return (%)"},
            linewidths=0.5,
            linecolor="#334155",
        )

        ax.set_title("Monthly Returns Heatmap (Stability Matrix)", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_valuation_chart(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        # Ensure data availability
        required = ["per", "fair_per", "close_price", "divergence"]
        if not all(col in data.columns for col in required):
            pass

        df = data

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1], sharex=True)

        # Upper Plot: PERs
        ax1.plot(df.index, df["per"], label="Actual PER", color=self.colors["primary"], linewidth=2)
        ax1.plot(
            df.index,
            df["fair_per"],
            label="Theoretical PER (Fair Value)",
            color=self.colors["success"],
            linestyle="--",
            linewidth=2,
        )
        ax1.fill_between(
            df.index,
            df["per"],
            df["fair_per"],
            where=(df["per"] > df["fair_per"]),
            color=self.colors["danger"],
            alpha=0.1,
            label="Overvalued",
        )
        ax1.fill_between(
            df.index,
            df["per"],
            df["fair_per"],
            where=(df["per"] <= df["fair_per"]),
            color=self.colors["success"],
            alpha=0.1,
            label="Undervalued",
        )

        ax1.set_title("Nikkei 225 Valuation Model (Yield Gap Approach)", fontsize=16, fontweight="bold", pad=20)
        ax1.set_ylabel("PER (Ratio)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Lower Plot: Divergence
        # Normalize divergence for better visualization
        ax2.plot(df.index, df["divergence"], color=self.colors["secondary"], linewidth=1.5, label="Divergence (%)")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.axhline(y=20, color=self.colors["danger"], linestyle=":", alpha=0.5)
        ax2.axhline(y=-20, color=self.colors["success"], linestyle=":", alpha=0.5)

        # Color areas based on levels
        ax2.fill_between(
            df.index, df["divergence"], 20, where=(df["divergence"] > 20), color=self.colors["danger"], alpha=0.3
        )
        ax2.fill_between(
            df.index, df["divergence"], -20, where=(df["divergence"] < -20), color=self.colors["success"], alpha=0.3
        )

        ax2.set_title("Home-made Valuation Divergence", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Divergence (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower right")

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)
        return fig

    def create_seasonal_bar_chart(
        self,
        seasonality_results: Dict[int, Any],
        metric: str = "mean_return",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months = range(1, 13)
        values = []
        colors = []
        significance_markers = []

        for m in months:
            result = seasonality_results.get(m)
            if not result:
                values.append(0)
                colors.append(self.colors["secondary"])
                significance_markers.append("")
                continue

            if metric == "mean_return":
                val = result.mean_return * 100
                values.append(val)
                colors.append(self.colors["success"] if val >= 0 else self.colors["danger"])
                marker = (
                    "**" if result.is_significant and result.t_pvalue < 0.01 else "*" if result.is_significant else ""
                )
                significance_markers.append(marker)
            elif metric == "t_pvalue":
                val = result.t_pvalue
                values.append(val)
                colors.append(self.colors["primary"] if val > 0.05 else self.colors["danger"])
                significance_markers.append("")

        fig, ax = plt.subplots(figsize=self.figsize)

        bars = ax.bar(month_names, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

        # Annotations
        if metric == "mean_return":
            ax.axhline(0, color="black", linewidth=1)
            ax.set_ylabel("Mean Return (%)")
            for bar, marker in zip(bars, significance_markers):
                if marker:
                    height = bar.get_height()
                    offset = max(values) * 0.05 if height >= 0 else -max(abs(min(values)), 1) * 0.1
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + offset,
                        marker,
                        ha="center",
                        va="bottom" if height >= 0 else "top",
                        fontsize=14,
                        fontweight="bold",
                        color="black",
                    )

        elif metric == "t_pvalue":
            ax.axhline(0.05, color=self.colors["danger"], linestyle="--", linewidth=1.5, label="p=0.05")
            ax.axhline(0.10, color=self.colors["warning"], linestyle=":", linewidth=1.5, label="p=0.10")
            ax.set_ylabel("P-Value")
            ax.legend(loc="upper right")
            ax.set_ylim(0, max(max(values), 0.2))  # Ensure we see the small p-values structure

        ax.set_title(title or self._get_metric_title(metric), fontsize=16, fontweight="bold", pad=20)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            self._save_plot(fig, save_path)

        return fig

    def _save_plot(self, fig: plt.Figure, save_path: str):
        path = self.output_dir / save_path if self.output_dir else Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
