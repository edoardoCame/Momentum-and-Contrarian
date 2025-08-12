"""
Seasonal Strategy Backtesting Engine

Comprehensive backtesting framework specifically designed for seasonal trading strategies.
Includes proper bias prevention, transaction cost modeling, and performance attribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from seasonality_strategies import (
    BaseSeasonalStrategy, EnergySeasonalStrategy, 
    AgriculturalSeasonalStrategy, MetalsSeasonalStrategy, SectorRotationStrategy
)
from seasonality_engine import CommoditySeasonalityAnalyzer
import warnings
warnings.filterwarnings('ignore')


class SeasonalBacktestEngine:
    """
    Comprehensive backtesting engine for seasonal commodity strategies.
    
    Features:
    - Multiple strategy comparison
    - Transaction cost modeling
    - Drawdown analysis
    - Seasonal performance attribution
    - Risk-adjusted metrics
    - Statistical significance testing
    """
    
    def __init__(self, transaction_cost: float = 0.0010, 
                 risk_free_rate: float = 0.02):
        """
        Initialize the backtesting engine.
        
        Args:
            transaction_cost: Transaction cost per trade (default 10bp)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.analyzer = CommoditySeasonalityAnalyzer()
        self.results = {}
    
    def run_strategy_comparison(self, start_date: str = '2015-01-01', 
                              end_date: str = '2025-08-01') -> Dict:
        """
        Run comprehensive comparison of all seasonal strategies.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            Dictionary with strategy performance results
        """
        print("Loading data and calculating seasonal statistics...")
        
        # Load data and calculate seasonal patterns
        self.analyzer.load_commodity_data()
        returns_data = self.analyzer.extract_returns_data()
        
        # Filter date range for backtesting
        mask = (returns_data.index >= start_date) & (returns_data.index <= end_date)
        returns_data = returns_data.loc[mask]
        
        # Calculate seasonal statistics on full historical data
        seasonal_stats = self.analyzer.get_seasonal_summary_stats(returns_data)
        
        print(f"Backtesting period: {start_date} to {end_date}")
        print(f"Total observations: {len(returns_data):,} days")
        
        # Initialize strategies
        strategies = {
            'Energy_Seasonal': EnergySeasonalStrategy(),
            'Agricultural_Seasonal': AgriculturalSeasonalStrategy(), 
            'Metals_Seasonal': MetalsSeasonalStrategy(),
            'Sector_Rotation': SectorRotationStrategy()
        }
        
        # Run backtests
        strategy_results = {}
        for name, strategy in strategies.items():
            print(f"\nBacktesting {name}...")
            result = strategy.backtest_strategy(
                returns_data, seasonal_stats, self.transaction_cost
            )
            strategy_results[name] = result
            
            # Print quick summary
            annual_return = result['annual_return'] * 100
            volatility = result['annual_volatility'] * 100
            sharpe = result['sharpe_ratio']
            max_dd = result['max_drawdown'] * 100
            
            print(f"  Annual Return: {annual_return:+.2f}%")
            print(f"  Volatility: {volatility:.2f}%")
            print(f"  Sharpe Ratio: {sharpe:.2f}")
            print(f"  Max Drawdown: {max_dd:.2f}%")
        
        # Add benchmark (equal-weight buy-and-hold)
        benchmark_returns = returns_data.mean(axis=1)
        benchmark_result = self._calculate_benchmark_performance(benchmark_returns)
        strategy_results['Benchmark_EqualWeight'] = benchmark_result
        
        self.results = strategy_results
        return strategy_results
    
    def _calculate_benchmark_performance(self, returns: pd.Series) -> Dict:
        """Calculate benchmark performance metrics"""
        if len(returns) == 0 or returns.isna().all():
            return self._empty_performance_dict()
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'net_returns': returns,
            'hit_rate': (returns > 0).mean(),
            'total_trades': len(returns[returns != 0])
        }
    
    def _empty_performance_dict(self) -> Dict:
        """Return empty performance dictionary"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'hit_rate': 0.0,
            'total_trades': 0
        }
    
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate comprehensive performance comparison table"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        metrics = []
        for strategy_name, result in self.results.items():
            metrics.append({
                'Strategy': strategy_name,
                'Annual Return (%)': result['annual_return'] * 100,
                'Volatility (%)': result['annual_volatility'] * 100,
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown (%)': result['max_drawdown'] * 100,
                'Hit Rate (%)': result['hit_rate'] * 100,
                'Total Trades': result.get('total_trades', 0),
                'Return/Risk': result['annual_return'] / max(result['annual_volatility'], 0.001)
            })
        
        df = pd.DataFrame(metrics).set_index('Strategy')
        return df.round(2)
    
    def plot_performance_comparison(self, figsize: Tuple[int, int] = (16, 12)) -> None:
        """Create comprehensive performance visualization"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Cumulative returns
        for strategy_name, result in self.results.items():
            if 'net_returns' in result:
                cumulative = (1 + result['net_returns']).cumprod()
                ax1.plot(cumulative.index, cumulative, label=strategy_name, linewidth=2)
        
        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Risk-Return Scatter
        returns = [r['annual_return'] * 100 for r in self.results.values()]
        volatilities = [r['annual_volatility'] * 100 for r in self.results.values()]
        colors = plt.cm.Set1(np.arange(len(self.results)))
        
        scatter = ax2.scatter(volatilities, returns, c=colors, s=100, alpha=0.7)
        
        # Add strategy labels
        for i, (name, _) in enumerate(self.results.items()):
            ax2.annotate(name.replace('_', '\n'), 
                        (volatilities[i], returns[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Annual Return (%)')
        ax2.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown comparison
        for strategy_name, result in self.results.items():
            if 'net_returns' in result:
                cumulative = (1 + result['net_returns']).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max * 100
                ax3.plot(drawdown.index, drawdown, label=strategy_name, linewidth=2)
        
        ax3.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(ax3.get_xlim(), ax3.get_ylim()[0], 0, alpha=0.1, color='red')
        
        # Plot 4: Performance metrics radar chart
        performance_df = self.generate_performance_report()
        
        # Normalize metrics for radar chart (0-1 scale)
        normalized_metrics = performance_df.copy()
        for col in normalized_metrics.columns:
            if col == 'Max Drawdown (%)':
                # For drawdown, lower is better, so invert
                normalized_metrics[col] = 1 - (normalized_metrics[col] - normalized_metrics[col].min()) / (normalized_metrics[col].max() - normalized_metrics[col].min())
            else:
                normalized_metrics[col] = (normalized_metrics[col] - normalized_metrics[col].min()) / (normalized_metrics[col].max() - normalized_metrics[col].min())
        
        # Bar chart instead of radar for simplicity
        strategy_names = [name.replace('_', '\n') for name in normalized_metrics.index]
        x_pos = np.arange(len(strategy_names))
        
        # Plot Sharpe ratio as example metric
        sharpe_ratios = [r['sharpe_ratio'] for r in self.results.values()]
        bars = ax4.bar(x_pos, sharpe_ratios, color=colors, alpha=0.7)
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, sharpe_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../results/seasonal_strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_monthly_attribution(self) -> pd.DataFrame:
        """Analyze performance attribution by month"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        monthly_attribution = pd.DataFrame()
        
        for strategy_name, result in self.results.items():
            if 'net_returns' in result:
                returns = result['net_returns']
                monthly_returns = returns.groupby(returns.index.month).mean() * 100
                monthly_attribution[strategy_name] = monthly_returns
        
        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_attribution.index = month_names
        
        return monthly_attribution.round(3)
    
    def plot_monthly_attribution(self) -> None:
        """Plot monthly performance attribution"""
        monthly_attribution = self.analyze_monthly_attribution()
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(monthly_attribution.T, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Average Monthly Return (%)'})
        plt.title('Monthly Performance Attribution by Strategy', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Strategy')
        plt.tight_layout()
        plt.savefig('../results/monthly_attribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_significance_test(self, strategy_name: str) -> Dict:
        """Test statistical significance of strategy returns"""
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        result = self.results[strategy_name]
        if 'net_returns' not in result:
            return {}
        
        returns = result['net_returns'].dropna()
        benchmark_returns = self.results['Benchmark_EqualWeight']['net_returns']
        
        # T-test against zero (strategy has positive expected return)
        from scipy import stats
        t_stat_zero, p_value_zero = stats.ttest_1samp(returns, 0)
        
        # T-test against benchmark
        t_stat_bench, p_value_bench = stats.ttest_ind(returns, benchmark_returns)
        
        # Information ratio (excess return / tracking error)
        excess_returns = returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        return {
            'mean_return': returns.mean(),
            't_stat_vs_zero': t_stat_zero,
            'p_value_vs_zero': p_value_zero,
            't_stat_vs_benchmark': t_stat_bench,
            'p_value_vs_benchmark': p_value_bench,
            'information_ratio': information_ratio,
            'significant_vs_zero': p_value_zero < 0.05,
            'significant_vs_benchmark': p_value_bench < 0.05
        }
    
    def export_detailed_results(self, filepath: str = '../results/seasonal_strategy_results.xlsx'):
        """Export detailed results to Excel file"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Performance summary
            performance_summary = self.generate_performance_report()
            performance_summary.to_excel(writer, sheet_name='Performance_Summary')
            
            # Monthly attribution
            monthly_attribution = self.analyze_monthly_attribution()
            monthly_attribution.to_excel(writer, sheet_name='Monthly_Attribution')
            
            # Individual strategy details
            for strategy_name, result in self.results.items():
                if 'net_returns' in result:
                    # Daily returns and positions
                    strategy_data = pd.DataFrame({
                        'Date': result['net_returns'].index,
                        'Net_Returns': result['net_returns'],
                        'Cumulative_Returns': (1 + result['net_returns']).cumprod() - 1
                    })
                    
                    # Add positions if available
                    if 'positions' in result:
                        positions = result['positions']
                        for col in positions.columns:
                            strategy_data[f'Position_{col}'] = positions[col]
                    
                    strategy_data.to_excel(writer, sheet_name=f'{strategy_name}_Details', index=False)
            
            # Statistical significance tests
            significance_results = []
            for strategy_name in self.results.keys():
                if strategy_name != 'Benchmark_EqualWeight':
                    sig_test = self.statistical_significance_test(strategy_name)
                    sig_test['Strategy'] = strategy_name
                    significance_results.append(sig_test)
            
            if significance_results:
                sig_df = pd.DataFrame(significance_results).set_index('Strategy')
                sig_df.to_excel(writer, sheet_name='Statistical_Tests')
        
        print(f"Detailed results exported to: {filepath}")


def main():
    """Main execution function for seasonal strategy backtesting"""
    print("=== Seasonal Commodity Strategy Backtesting ===\n")
    
    # Initialize backtesting engine
    engine = SeasonalBacktestEngine(transaction_cost=0.0010)
    
    # Run comprehensive strategy comparison
    results = engine.run_strategy_comparison(
        start_date='2015-01-01', 
        end_date='2025-08-01'
    )
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Display performance summary
    performance_df = engine.generate_performance_report()
    print(performance_df)
    
    # Generate visualizations
    print("\nGenerating performance visualizations...")
    engine.plot_performance_comparison()
    engine.plot_monthly_attribution()
    
    # Export detailed results
    print("\nExporting detailed results...")
    engine.export_detailed_results()
    
    # Statistical significance testing
    print("\nStatistical Significance Tests:")
    print("-" * 40)
    for strategy_name in results.keys():
        if strategy_name != 'Benchmark_EqualWeight':
            sig_test = engine.statistical_significance_test(strategy_name)
            print(f"\n{strategy_name}:")
            print(f"  Mean Daily Return: {sig_test['mean_return']*100:.4f}%")
            print(f"  Significant vs Zero: {sig_test['significant_vs_zero']} (p={sig_test['p_value_vs_zero']:.4f})")
            print(f"  Significant vs Benchmark: {sig_test['significant_vs_benchmark']} (p={sig_test['p_value_vs_benchmark']:.4f})")
            print(f"  Information Ratio: {sig_test['information_ratio']:.4f}")
    
    print("\nBacktesting complete! Check the results/ directory for outputs.")


if __name__ == "__main__":
    main()