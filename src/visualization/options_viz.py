from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..options.strategies import OptionType, StrategyDefinition


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
