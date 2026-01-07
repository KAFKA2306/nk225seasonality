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

    def _save_plot(self, fig: plt.Figure, save_path: str):
        path = self.output_dir / save_path if self.output_dir else Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
