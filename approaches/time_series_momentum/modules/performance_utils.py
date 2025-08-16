import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SimplePerformanceAnalyzer:
    """
    Simplified performance analysis and visualization for TSMOM strategies.
    
    Clean, focused plotting with proper date formatting and streamlined interface.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_equity_curves(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot equity curves for all strategies with proper date formatting.
        
        Args:
            results: Dictionary with backtest results
            save_path: Optional path to save the plot
        """
        # All strategies are monthly now
        if not results:
            print("No results to plot")
            return
            
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Plot all monthly strategies
        for strategy_name, strategy_results in results.items():
            equity = self._ensure_series_format(strategy_results['equity'])
            ax.plot(equity.index, equity.values, label=strategy_name, linewidth=2)
        
        ax.set_title('Monthly Contrarian Strategies - Net Equity Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        self._format_dates(ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_summary(self, metrics: pd.DataFrame, 
                               save_path: Optional[str] = None) -> None:
        """
        Create consolidated performance summary plots.
        
        Args:
            metrics: DataFrame with performance metrics
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Plot 1: Risk-Return scatter
        annual_return = metrics['Annual_Return'] * 100
        annual_vol = metrics['Annual_Vol'] * 100
        
        scatter = axes[0].scatter(annual_vol, annual_return, 
                                c=metrics['Sharpe_Ratio'], 
                                cmap='RdYlGn', s=100, alpha=0.8)
        
        for i, strategy in enumerate(metrics.index):
            axes[0].annotate(strategy, (annual_vol.iloc[i], annual_return.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[0].set_xlabel('Volatility (%)')
        axes[0].set_ylabel('Annual Return (%)')
        axes[0].set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Sharpe Ratio')
        
        # Plot 2: Sharpe ratio comparison
        sharpe_ratios = metrics['Sharpe_Ratio']
        colors = ['green' if x > 0 else 'red' for x in sharpe_ratios]
        bars = axes[1].bar(range(len(sharpe_ratios)), sharpe_ratios.values, color=colors, alpha=0.7)
        axes[1].set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].set_xticks(range(len(sharpe_ratios)))
        axes[1].set_xticklabels(sharpe_ratios.index, rotation=45, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Maximum drawdown
        max_dd = metrics['Max_Drawdown'] * 100
        axes[2].bar(range(len(max_dd)), max_dd.values, color='red', alpha=0.7)
        axes[2].set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Max Drawdown (%)')
        axes[2].set_xticks(range(len(max_dd)))
        axes[2].set_xticklabels(max_dd.index, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Annual returns distribution
        annual_returns = metrics['Annual_Return'] * 100
        colors = ['green' if x > 0 else 'red' for x in annual_returns]
        axes[3].bar(range(len(annual_returns)), annual_returns.values, color=colors, alpha=0.7)
        axes[3].set_title('Annual Returns Distribution', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Annual Return (%)')
        axes[3].set_xticks(range(len(annual_returns)))
        axes[3].set_xticklabels(annual_returns.index, rotation=45, ha='right')
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_drawdown_analysis(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot drawdown analysis with fixed Series handling.
        
        Args:
            results: Dictionary with backtest results
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Calculate drawdowns for all strategies
        drawdowns = {}
        for strategy_name, strategy_results in results.items():
            equity = self._ensure_series_format(strategy_results['equity'])
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            drawdowns[strategy_name] = drawdown
        
        # Plot 1: Drawdown time series
        for strategy_name, drawdown in drawdowns.items():
            axes[0].plot(drawdown.index, drawdown.values, label=strategy_name, alpha=0.8)
        
        # Use the last drawdown for fill_between (they should have similar indices)
        last_drawdown = list(drawdowns.values())[-1]
        axes[0].fill_between(last_drawdown.index, last_drawdown.values, 0, alpha=0.3)
        axes[0].set_title('Drawdown Time Series', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Drawdown')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        self._format_dates(axes[0])
        
        # Plot 2: Maximum drawdown comparison
        max_drawdowns = [dd.min() for dd in drawdowns.values()]
        strategy_names = list(drawdowns.keys())
        
        colors = ['red' if x < -0.1 else 'orange' if x < -0.05 else 'green' for x in max_drawdowns]
        bars = axes[1].bar(range(len(strategy_names)), max_drawdowns, color=colors, alpha=0.7)
        axes[1].set_title('Maximum Drawdown Comparison', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Max Drawdown')
        axes[1].set_xticks(range(len(strategy_names)))
        axes[1].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Drawdown distribution
        axes[2].boxplot([dd.dropna().values for dd in drawdowns.values()], 
                       labels=strategy_names)
        axes[2].set_title('Drawdown Distribution', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Drawdown')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Recovery analysis (time to recover from drawdowns)
        for strategy_name, drawdown in drawdowns.items():
            # Simple recovery metric: periods below -1%
            underwater_periods = (drawdown < -0.01).sum()
            axes[3].bar(strategy_name, underwater_periods, alpha=0.7)
        
        axes[3].set_title('Time Underwater (>1% drawdown)', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Periods Underwater')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_table(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Create a nicely formatted summary table for display.
        
        Args:
            metrics: DataFrame with performance metrics
            
        Returns:
            Formatted DataFrame for display
        """
        # Select key metrics
        display_cols = [
            'Total_Return', 'Annual_Return', 'Annual_Vol', 
            'Sharpe_Ratio', 'Max_Drawdown'
        ]
        
        formatted_df = metrics[display_cols].copy()
        
        # Format as percentages
        pct_cols = ['Total_Return', 'Annual_Return', 'Annual_Vol', 
                   'Max_Drawdown']
        
        for col in pct_cols:
            if col in formatted_df.columns:
                formatted_df[col] = (formatted_df[col] * 100).round(2).astype(str) + '%'
        
        # Format Sharpe ratio
        if 'Sharpe_Ratio' in formatted_df.columns:
            formatted_df['Sharpe_Ratio'] = formatted_df['Sharpe_Ratio'].round(2)
        
        # Rename columns for display
        formatted_df.columns = [
            'Total Return', 'Annual Return', 'Volatility', 
            'Sharpe Ratio', 'Max Drawdown'
        ]
        
        return formatted_df
    
    def generate_summary_report(self, results: Dict, metrics: pd.DataFrame) -> str:
        """
        Generate a concise text summary report.
        
        Args:
            results: Dictionary with backtest results
            metrics: DataFrame with performance metrics
            
        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 50)
        report.append("CONTRARIAN STRATEGIES - PERFORMANCE SUMMARY")
        report.append("=" * 50)
        
        # Best performing strategy
        best_strategy = metrics['Sharpe_Ratio'].idxmax()
        best_sharpe = metrics.loc[best_strategy, 'Sharpe_Ratio']
        
        report.append(f"Best Strategy: {best_strategy}")
        report.append(f"Best Sharpe Ratio: {best_sharpe:.2f}")
        
        # Average strategy performance
        avg_sharpe = metrics['Sharpe_Ratio'].mean()
        report.append(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        
        # Lookback period analysis
        lookback_1m = [s for s in metrics.index if '1M' in s]
        lookback_3m = [s for s in metrics.index if '3M' in s]
        lookback_6m = [s for s in metrics.index if '6M' in s]
        lookback_12m = [s for s in metrics.index if '12M' in s]
        
        if lookback_1m:
            sharpe_1m = metrics.loc[lookback_1m, 'Sharpe_Ratio'].mean()
            report.append(f"1M Lookback Sharpe: {sharpe_1m:.2f}")
        if lookback_3m:
            sharpe_3m = metrics.loc[lookback_3m, 'Sharpe_Ratio'].mean()
            report.append(f"3M Lookback Sharpe: {sharpe_3m:.2f}")
        if lookback_6m:
            sharpe_6m = metrics.loc[lookback_6m, 'Sharpe_Ratio'].mean()
            report.append(f"6M Lookback Sharpe: {sharpe_6m:.2f}")
        if lookback_12m:
            sharpe_12m = metrics.loc[lookback_12m, 'Sharpe_Ratio'].mean()
            report.append(f"12M Lookback Sharpe: {sharpe_12m:.2f}")
        
        # Performance summary (no transaction costs)
        avg_return = metrics['Annual_Return'].mean() * 100
        report.append(f"Avg Annual Return: {avg_return:.2f}%")
        
        report.append("=" * 50)
        
        return "\n".join(report)
    
    def _ensure_series_format(self, data) -> pd.Series:
        """Ensure data is in pandas Series format with proper index."""
        if isinstance(data, np.ndarray):
            # If it's a numpy array, we need to create a Series
            # This shouldn't happen with the new backtest engine, but safety check
            return pd.Series(data)
        elif isinstance(data, pd.Series):
            return data
        else:
            # Try to convert whatever it is to Series
            return pd.Series(data)
    
    def _format_dates(self, ax) -> None:
        """Apply consistent date formatting to an axis."""
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))  # Jan and Jul
        
        # Rotate labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


