#!/usr/bin/env python3
"""
Performance Analyzer Module for TSMOM Strategy

Calculates performance metrics and generates visualizations
following academic standards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Tuple, Optional, List
from pathlib import Path

class TSMOMPerformanceAnalyzer:
    """
    Performance analysis and visualization for TSMOM strategy
    """
    
    def __init__(self, results_dir: str = "approaches/TSMOM/results"):
        """
        Initialize performance analyzer
        
        Args:
            results_dir: Directory to save results and plots
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print(f"Performance Analyzer initialized")
        print(f"Results directory: {self.results_dir}")
    
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    name: str = "Strategy") -> Dict[str, float]:
        """
        Calculate performance metrics (vectorized)
        
        Args:
            returns: Monthly returns series
            name: Strategy name
            
        Returns:
            Dictionary with performance metrics
        """
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 12:
            return {'error': f'Insufficient data for {name}'}
        
        # VECTORIZED: Calculate all basic stats in single pass
        n_periods = len(clean_returns)
        n_years = n_periods / 12
        mean_ret, std_ret = clean_returns.mean(), clean_returns.std()
        
        # VECTORIZED: Equity curve and drawdown in single operation
        equity_curve = (1 + clean_returns).cumprod()
        drawdown = (equity_curve / equity_curve.expanding().max() - 1)
        
        # VECTORIZED: All metrics in dictionary comprehension
        return {
            'total_return': equity_curve.iloc[-1] - 1,
            'cagr': equity_curve.iloc[-1] ** (1/n_years) - 1,
            'annualized_return': mean_ret * 12,
            'annualized_volatility': std_ret * np.sqrt(12),
            'sharpe_ratio': mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else np.nan,
            'max_drawdown': drawdown.min(),
            'win_rate': (clean_returns > 0).mean(),
            'best_month': clean_returns.max(),
            'worst_month': clean_returns.min(),
            'skewness': clean_returns.skew(),
            'start_date': clean_returns.index[0],
            'end_date': clean_returns.index[-1],
            'n_periods': n_periods,
            'n_years': n_years
        }
    
    def create_performance_summary(self, portfolio_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Create performance summary (vectorized)
        
        Args:
            portfolio_returns: Dictionary with portfolio return series
            
        Returns:
            DataFrame with performance metrics
        """
        print("\nCalculating performance metrics (vectorized)...")
        
        # VECTORIZED: Process all portfolios in single dict comprehension
        summary_data = {name: self.calculate_performance_metrics(returns, name)
                       for name, returns in portfolio_returns.items()
                       if name != 'n_active_assets' and 'error' not in self.calculate_performance_metrics(returns, name)}
        
        if not summary_data:
            raise ValueError("No valid portfolio data")
        
        # VECTORIZED: DataFrame creation and rounding
        summary_df = pd.DataFrame(summary_data).T
        return summary_df.round(4)
    
    def calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series
        
        Args:
            returns: Returns series
            
        Returns:
            Drawdown series
        """
        equity_curve = (1 + returns.fillna(0)).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def create_equity_curve_plot(self, portfolio_returns: Dict[str, pd.Series], 
                               save_path: Optional[str] = None) -> None:
        """
        Create equity curve visualization
        
        Args:
            portfolio_returns: Dictionary with portfolio return series
            save_path: Optional path to save plot
        """
        print("\nCreating equity curve plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # VECTORIZED: Plot all equity curves in single operation
        equity_curves = {name: (1 + returns.dropna()).cumprod() 
                        for name, returns in portfolio_returns.items() 
                        if name != 'n_active_assets' and len(returns.dropna()) > 0}
        
        for name, equity_curve in equity_curves.items():
            ax1.plot(equity_curve.index, equity_curve.values, 
                    label=name.title(), linewidth=2)
        
        ax1.set_title('TSMOM Strategy - Equity Curves', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        
        # VECTORIZED: Plot all drawdowns in single operation
        for name, equity_curve in equity_curves.items():
            drawdown = equity_curve / equity_curve.expanding().max() - 1
            ax2.fill_between(drawdown.index, drawdown.values, 0, 
                           alpha=0.3, label=name.title())
        
        ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.results_dir / "equity_tsmom.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Equity curve plot saved: {save_path}")
        
        plt.show()
    
    def create_drawdown_plot(self, portfolio_returns: Dict[str, pd.Series],
                           save_path: Optional[str] = None) -> None:
        """
        Create detailed drawdown analysis plot
        
        Args:
            portfolio_returns: Dictionary with portfolio return series
            save_path: Optional path to save plot
        """
        print("\nCreating drawdown plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (name, returns) in enumerate(portfolio_returns.items()):
            if name == 'n_active_assets':
                continue
                
            clean_returns = returns.dropna()
            if len(clean_returns) > 0:
                drawdown = self.calculate_drawdown_series(clean_returns)
                
                color = colors[i % len(colors)]
                ax.fill_between(drawdown.index, drawdown.values, 0, 
                              alpha=0.7, color=color, label=name.title())
                ax.plot(drawdown.index, drawdown.values, 
                       color=color, linewidth=1.5)
        
        ax.set_title('TSMOM Strategy - Drawdown Analysis', fontsize=16, fontweight='bold')
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Format axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.results_dir / "dd_tsmom.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Drawdown plot saved: {save_path}")
        
        plt.show()
    
    def save_results_csv(self, portfolio_returns: Dict[str, pd.Series],
                        weights: pd.DataFrame,
                        filename: str = "tsmom_yahoo_2000.csv") -> None:
        """
        Save results to CSV (vectorized)
        
        Args:
            portfolio_returns: Dictionary with portfolio return series
            weights: Position weights DataFrame
            filename: Output filename
        """
        print(f"\nSaving results...")
        
        # VECTORIZED: Combine data in single dict operation
        results_data = {**{f'{name}_returns': returns for name, returns in portfolio_returns.items() if name != 'n_active_assets'},
                       **{f'weight_{col}': weights[col] for col in weights.columns[:10]}}
        
        if 'n_active_assets' in portfolio_returns:
            results_data['n_active_assets'] = portfolio_returns['n_active_assets']
        
        # VECTORIZED: Single operation CSV save
        csv_path = self.results_dir / filename
        pd.DataFrame(results_data).to_csv(csv_path)
        print(f"Results saved: {csv_path} ({len(results_data)} columns)")
    
    def print_performance_summary(self, summary_df: pd.DataFrame) -> None:
        """
        Print formatted performance summary
        
        Args:
            summary_df: Performance summary DataFrame
        """
        print(f"\n{'='*80}")
        print(f"TSMOM PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        # Key metrics to display
        key_metrics = [
            ('Period', ['start_date', 'end_date', 'n_years']),
            ('Returns', ['cagr', 'annualized_volatility', 'sharpe_ratio']),
            ('Risk', ['max_drawdown', 'calmar_ratio', 'win_rate']),
            ('Distribution', ['best_month', 'worst_month', 'skewness'])
        ]
        
        for portfolio in summary_df.index:
            print(f"\n{portfolio.upper()} PORTFOLIO:")
            print("-" * 40)
            
            for category, metrics in key_metrics:
                print(f"\n{category}:")
                for metric in metrics:
                    if metric in summary_df.columns:
                        value = summary_df.loc[portfolio, metric]
                        
                        if pd.isna(value):
                            print(f"  {metric}: N/A")
                        elif metric in ['start_date', 'end_date']:
                            if hasattr(value, 'strftime'):
                                print(f"  {metric}: {value.strftime('%Y-%m-%d')}")
                            else:
                                print(f"  {metric}: {value}")
                        elif metric in ['cagr', 'annualized_return', 'annualized_volatility', 'max_drawdown', 'win_rate', 'best_month', 'worst_month']:
                            print(f"  {metric}: {value:.2%}")
                        elif metric in ['sharpe_ratio', 'calmar_ratio', 'skewness', 'n_years']:
                            print(f"  {metric}: {value:.2f}")
                        else:
                            print(f"  {metric}: {value}")
        
        print(f"\n{'='*80}")
    
    def run_full_analysis(self, portfolio_returns: Dict[str, pd.Series],
                         weights: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete performance analysis
        
        Args:
            portfolio_returns: Dictionary with portfolio return series
            weights: Position weights DataFrame
            
        Returns:
            Performance summary DataFrame
        """
        print(f"\n{'='*60}")
        print(f"RUNNING PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Calculate performance metrics
        summary_df = self.create_performance_summary(portfolio_returns)
        
        # Print summary
        self.print_performance_summary(summary_df)
        
        # Create visualizations
        self.create_equity_curve_plot(portfolio_returns)
        self.create_drawdown_plot(portfolio_returns)
        
        # Save results
        self.save_results_csv(portfolio_returns, weights)
        
        # Save summary table
        summary_path = self.results_dir / "performance_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"\nPerformance summary saved: {summary_path}")
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        return summary_df