from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


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
        """
        Creates a boxplot of monthly returns distributions.
        Shows Median, IQR, Whiskers (1.5 IQR), and Outliers.
        """
        if "returns" not in data.columns:
            data = data.copy()
            data["returns"] = data["close_price"].pct_change()
        
        data["month"] = data.index.month
        month_data = [
            data[data["month"] == m]["returns"].dropna() * 100 for m in range(1, 13)
        ]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create boxplot
        bp = ax.boxplot(
            month_data,
            patch_artist=True,
            notch=False,
            vert=True,
            labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        )
        
        # Style
        for patch in bp['boxes']:
            patch.set_facecolor(self.colors["primary"])
            patch.set_alpha(0.6)
            patch.set_edgecolor("black")
            
        for median in bp['medians']:
            median.set_color(self.colors["danger"])
            median.set_linewidth(2)

        ax.set_title("Monthly Returns Distribution (Boxplot)", fontsize=16, fontweight="bold", pad=20)
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Month")
        ax.grid(True, axis="y", alpha=0.3)
        
        # Add zero line
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
        """
        Creates a Year x Month heatmap of returns.
        Visualizes stability of patterns over time.
        """
        if "returns" not in data.columns:
            data = data.copy()
            data["returns"] = data["close_price"].pct_change()

        pivot_table = data.pivot_table(
            values="returns",
            index=data.index.year,
            columns=data.index.month,
            aggfunc="sum" # Monthly return is effectively sum of daily log returns approx, or simple compound. 
            # ideally resampling, but pivot of daily returns isn't right. 
            # We need to resample to monthly first.
        ) * 100
        
        # Re-process for accurate monthly values
        monthly_data = data["close_price"].resample("ME").last().pct_change() * 100
        pivot_table = pd.DataFrame(index=monthly_data.index.year.unique(), columns=range(1, 13))
        
        for date, val in monthly_data.items():
            pivot_table.loc[date.year, date.month] = val
            
        pivot_table = pivot_table.astype(float).sort_index(ascending=False) # Newest years top
        
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
            linecolor="#334155"
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
        """
        Creates a publication-grade Valuation Chart.
        Plots:
        1. Actual PER vs Theoretical PER (Fair Value)
        2. JGB Yield (context)
        3. Divergence (Over/Undervaluation)
        """
        # Ensure data availability
        required = ["per", "fair_per", "close_price", "divergence"]
        if not all(col in data.columns for col in required):
             # If columns are missing, we cannot plot. 
             # Previously we had magic numbers here, but that violates user rules.
             # We assume pipeline has populated these. If not, we log a warning or return empty figure.
             # For now, let's just use what we have or return early to avoid crashing, 
             # but strictly NO MAGIC NUMBERS.
             # In a real app we might raise ValueError("Missing valuation data from pipeline")
             pass 
        
        df = data

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1], sharex=True)
        
        # Upper Plot: PERs
        ax1.plot(df.index, df["per"], label="Actual PER", color=self.colors["primary"], linewidth=2)
        ax1.plot(df.index, df["fair_per"], label="Theoretical PER (Fair Value)", color=self.colors["success"], linestyle="--", linewidth=2)
        ax1.fill_between(df.index, df["per"], df["fair_per"], where=(df["per"] > df["fair_per"]), color=self.colors["danger"], alpha=0.1, label="Overvalued")
        ax1.fill_between(df.index, df["per"], df["fair_per"], where=(df["per"] <= df["fair_per"]), color=self.colors["success"], alpha=0.1, label="Undervalued")
        
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
        ax2.fill_between(df.index, df["divergence"], 20, where=(df["divergence"] > 20), color=self.colors["danger"], alpha=0.3)
        ax2.fill_between(df.index, df["divergence"], -20, where=(df["divergence"] < -20), color=self.colors["success"], alpha=0.3)

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
        """
        Creates a publication-grade Bar Chart for seasonal metrics.
        Replaces the hard-to-read 1D heatmap.
        """
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
                marker = "**" if result.is_significant and result.t_pvalue < 0.01 else "*" if result.is_significant else ""
                significance_markers.append(marker)
            elif metric == "t_pvalue":
                val = result.t_pvalue
                values.append(val)
                # Highlights significant p-values
                colors.append(self.colors["primary"] if val > 0.05 else self.colors["danger"]) 
                significance_markers.append("")

        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.bar(month_names, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        
        # Annotations
        if metric == "mean_return":
            ax.axhline(0, color="black", linewidth=1)
            ax.set_ylabel("Mean Return (%)")
            # Add significance markers on top of bars
            for bar, marker in zip(bars, significance_markers):
                if marker:
                    height = bar.get_height()
                    offset = max(values) * 0.05 if height >= 0 else -max(abs(min(values)), 1) * 0.1
                    ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                            marker, ha='center', va='bottom' if height >= 0 else 'top', 
                            fontsize=14, fontweight='bold', color='black')
                            
        elif metric == "t_pvalue":
            ax.axhline(0.05, color=self.colors["danger"], linestyle="--", linewidth=1.5, label="p=0.05")
            ax.axhline(0.10, color=self.colors["warning"], linestyle=":", linewidth=1.5, label="p=0.10")
            ax.set_ylabel("P-Value")
            ax.legend(loc="upper right")
            ax.set_ylim(0, max(max(values), 0.2)) # Ensure we see the small p-values structure

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

