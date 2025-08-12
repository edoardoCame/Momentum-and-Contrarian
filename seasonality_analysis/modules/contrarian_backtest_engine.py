"""
Contrarian Seasonal Strategy Backtesting Engine

A/B testing framework to compare normal seasonal strategies vs contrarian strategies.
Tests the hypothesis that inverting traditional seasonal logic improves performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from seasonal_backtest_engine import SeasonalBacktestEngine
from seasonality_strategies import (
    EnergySeasonalStrategy, AgriculturalSeasonalStrategy, 
    MetalsSeasonalStrategy, SectorRotationStrategy
)
from contrarian_seasonal_strategies import (
    ContrarianEnergyStrategy, ContrarianAgriculturalStrategy,
    ContrarianMetalsStrategy, ContrarianSectorRotationStrategy,
    AdaptiveSeasonalStrategy
)
from seasonality_engine import CommoditySeasonalityAnalyzer
import warnings
warnings.filterwarnings('ignore')


class ContrarianBacktestEngine(SeasonalBacktestEngine):
    """
    Extended backtesting engine for A/B testing normal vs contrarian seasonal strategies.
    """
    
    def __init__(self, transaction_cost: float = 0.0010, risk_free_rate: float = 0.02):
        super().__init__(transaction_cost, risk_free_rate)
        self.ab_test_results = {}
    
    def run_ab_strategy_comparison(self, start_date: str = '2015-01-01', 
                                  end_date: str = '2025-08-01') -> Dict:
        """
        Run A/B comparison between normal and contrarian seasonal strategies.
        
        Returns:
            Dictionary with both normal and contrarian strategy results
        """
        print("🔬 A/B TESTING: Normal vs Contrarian Seasonal Strategies")
        print("="*60)
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Transaction Cost: {self.transaction_cost*10000:.1f} basis points")
        print()
        
        # Load data and calculate seasonal patterns
        print("📊 Loading data and calculating seasonal statistics...")
        self.analyzer.load_commodity_data()
        returns_data = self.analyzer.extract_returns_data()
        
        # Filter date range for backtesting
        mask = (returns_data.index >= start_date) & (returns_data.index <= end_date)
        returns_data = returns_data.loc[mask]
        
        # Calculate seasonal statistics on full historical data
        seasonal_stats = self.analyzer.get_seasonal_summary_stats(returns_data)
        
        print(f"Backtesting period: {start_date} to {end_date}")
        print(f"Total observations: {len(returns_data):,} days")
        
        # Initialize both normal and contrarian strategies
        normal_strategies = {
            'Normal_Energy': EnergySeasonalStrategy(),
            'Normal_Agricultural': AgriculturalSeasonalStrategy(),
            'Normal_Metals': MetalsSeasonalStrategy(),
            'Normal_Sector_Rotation': SectorRotationStrategy()
        }
        
        contrarian_strategies = {
            'Contrarian_Energy': ContrarianEnergyStrategy(),
            'Contrarian_Agricultural': ContrarianAgriculturalStrategy(),
            'Contrarian_Metals': ContrarianMetalsStrategy(),
            'Contrarian_Sector_Rotation': ContrarianSectorRotationStrategy(),
            'Adaptive_Seasonal': AdaptiveSeasonalStrategy()
        }
        
        # Combine all strategies for testing
        all_strategies = {**normal_strategies, **contrarian_strategies}
        
        # Run backtests
        strategy_results = {}
        print("\n🚀 Running strategy backtests...")
        
        for name, strategy in all_strategies.items():
            strategy_type = "NORMAL" if name.startswith('Normal_') else "CONTRARIAN"
            print(f"\n{strategy_type}: {name.replace('_', ' ')}...")
            
            result = strategy.backtest_strategy(
                returns_data, seasonal_stats, self.transaction_cost
            )
            strategy_results[name] = result
            
            # Print quick summary
            annual_return = result['annual_return'] * 100
            volatility = result['annual_volatility'] * 100
            sharpe = result['sharpe_ratio']
            max_dd = result['max_drawdown'] * 100
            
            print(f"  📈 Annual Return: {annual_return:+.2f}%")
            print(f"  📊 Volatility: {volatility:.2f}%")
            print(f"  ⚡ Sharpe Ratio: {sharpe:.2f}")
            print(f"  📉 Max Drawdown: {max_dd:.2f}%")\n        \n        # Add benchmark\n        benchmark_returns = returns_data.mean(axis=1)\n        benchmark_result = self._calculate_benchmark_performance(benchmark_returns)\n        strategy_results['Benchmark_EqualWeight'] = benchmark_result\n        \n        self.ab_test_results = strategy_results\n        return strategy_results\n    \n    def analyze_ab_test_results(self) -> pd.DataFrame:\n        \"\"\"Analyze A/B test results comparing normal vs contrarian strategies\"\"\"\n        if not self.ab_test_results:\n            raise ValueError(\"No A/B test results available. Run A/B tests first.\")\n        \n        # Separate normal and contrarian results\n        normal_results = {k: v for k, v in self.ab_test_results.items() \n                         if k.startswith('Normal_')}\n        contrarian_results = {k: v for k, v in self.ab_test_results.items() \n                            if k.startswith('Contrarian_') or k == 'Adaptive_Seasonal'}\n        \n        # Calculate comparative metrics\n        comparison_data = []\n        \n        # Compare sector strategies\n        sector_pairs = [\n            ('Normal_Energy', 'Contrarian_Energy'),\n            ('Normal_Agricultural', 'Contrarian_Agricultural'),\n            ('Normal_Metals', 'Contrarian_Metals'),\n            ('Normal_Sector_Rotation', 'Contrarian_Sector_Rotation')\n        ]\n        \n        for normal_key, contrarian_key in sector_pairs:\n            if normal_key in self.ab_test_results and contrarian_key in self.ab_test_results:\n                normal = self.ab_test_results[normal_key]\n                contrarian = self.ab_test_results[contrarian_key]\n                \n                sector = normal_key.replace('Normal_', '')\n                \n                comparison_data.append({\n                    'Sector': sector,\n                    'Normal_Annual_Return_%': normal['annual_return'] * 100,\n                    'Contrarian_Annual_Return_%': contrarian['annual_return'] * 100,\n                    'Return_Improvement_%': (contrarian['annual_return'] - normal['annual_return']) * 100,\n                    'Normal_Sharpe': normal['sharpe_ratio'],\n                    'Contrarian_Sharpe': contrarian['sharpe_ratio'],\n                    'Sharpe_Improvement': contrarian['sharpe_ratio'] - normal['sharpe_ratio'],\n                    'Normal_MaxDD_%': normal['max_drawdown'] * 100,\n                    'Contrarian_MaxDD_%': contrarian['max_drawdown'] * 100,\n                    'DD_Improvement_%': (normal['max_drawdown'] - contrarian['max_drawdown']) * 100,\n                    'Contrarian_Wins': 'YES' if contrarian['sharpe_ratio'] > normal['sharpe_ratio'] else 'NO'\n                })\n        \n        return pd.DataFrame(comparison_data).set_index('Sector')\n    \n    def plot_ab_comparison(self, figsize: Tuple[int, int] = (18, 14)) -> None:\n        \"\"\"Create comprehensive A/B testing visualization\"\"\"\n        if not self.ab_test_results:\n            raise ValueError(\"No A/B test results available. Run A/B tests first.\")\n        \n        fig = plt.figure(figsize=figsize)\n        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)\n        \n        # Plot 1: Cumulative returns comparison (large plot)\n        ax1 = fig.add_subplot(gs[0, :])\n        \n        # Separate normal and contrarian for color coding\n        normal_strategies = [k for k in self.ab_test_results.keys() if k.startswith('Normal_')]\n        contrarian_strategies = [k for k in self.ab_test_results.keys() \n                               if k.startswith('Contrarian_') or k == 'Adaptive_Seasonal']\n        \n        # Plot normal strategies in red tones\n        for i, strategy_name in enumerate(normal_strategies):\n            if 'net_returns' in self.ab_test_results[strategy_name]:\n                returns = self.ab_test_results[strategy_name]['net_returns']\n                cumulative = (1 + returns).cumprod()\n                ax1.plot(cumulative.index, cumulative, \n                        label=strategy_name.replace('_', ' '), \n                        linewidth=2, linestyle='--', alpha=0.7,\n                        color=plt.cm.Reds(0.5 + i * 0.15))\n        \n        # Plot contrarian strategies in blue/green tones\n        for i, strategy_name in enumerate(contrarian_strategies):\n            if 'net_returns' in self.ab_test_results[strategy_name]:\n                returns = self.ab_test_results[strategy_name]['net_returns']\n                cumulative = (1 + returns).cumprod()\n                color = plt.cm.Blues(0.5 + i * 0.15) if 'Contrarian' in strategy_name else 'purple'\n                ax1.plot(cumulative.index, cumulative, \n                        label=strategy_name.replace('_', ' '), \n                        linewidth=2.5, \n                        color=color)\n        \n        # Add benchmark\n        if 'Benchmark_EqualWeight' in self.ab_test_results:\n            benchmark_returns = self.ab_test_results['Benchmark_EqualWeight']['net_returns']\n            benchmark_cumulative = (1 + benchmark_returns).cumprod()\n            ax1.plot(benchmark_cumulative.index, benchmark_cumulative, \n                    label='Benchmark', linewidth=2, color='black', alpha=0.8)\n        \n        ax1.set_title('A/B Test: Cumulative Returns Comparison\\n(Dashed Red = Normal, Solid Blue = Contrarian)', \n                     fontsize=14, fontweight='bold')\n        ax1.set_ylabel('Cumulative Return')\n        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n        ax1.grid(True, alpha=0.3)\n        \n        # Plot 2: Sharpe ratio comparison\n        ax2 = fig.add_subplot(gs[1, 0])\n        \n        sharpe_data = []\n        strategy_names = []\n        colors = []\n        \n        for strategy_name, result in self.ab_test_results.items():\n            if strategy_name != 'Benchmark_EqualWeight':\n                sharpe_data.append(result['sharpe_ratio'])\n                strategy_names.append(strategy_name.replace('_', '\\n'))\n                if strategy_name.startswith('Normal_'):\n                    colors.append('red')\n                elif strategy_name.startswith('Contrarian_'):\n                    colors.append('blue')\n                else:\n                    colors.append('purple')\n        \n        bars = ax2.bar(range(len(sharpe_data)), sharpe_data, color=colors, alpha=0.7)\n        ax2.set_xticks(range(len(strategy_names)))\n        ax2.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=9)\n        ax2.set_ylabel('Sharpe Ratio')\n        ax2.set_title('Sharpe Ratio Comparison', fontweight='bold')\n        ax2.grid(axis='y', alpha=0.3)\n        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)\n        \n        # Add values on bars\n        for bar, value in zip(bars, sharpe_data):\n            height = bar.get_height()\n            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,\n                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)\n        \n        # Plot 3: Return improvement (Contrarian - Normal)\n        ax3 = fig.add_subplot(gs[1, 1])\n        \n        comparison_df = self.analyze_ab_test_results()\n        improvement_data = comparison_df['Return_Improvement_%']\n        \n        bars = ax3.bar(improvement_data.index, improvement_data.values, \n                      color=['green' if x > 0 else 'red' for x in improvement_data.values],\n                      alpha=0.7)\n        ax3.set_ylabel('Return Improvement (%)')\n        ax3.set_title('Contrarian vs Normal\\nReturn Improvement', fontweight='bold')\n        ax3.grid(axis='y', alpha=0.3)\n        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)\n        ax3.tick_params(axis='x', rotation=45)\n        \n        # Add values on bars\n        for bar, value in zip(bars, improvement_data.values):\n            height = bar.get_height()\n            ax3.text(bar.get_x() + bar.get_width()/2, height + (0.5 if height > 0 else -1),\n                    f'{value:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', \n                    fontweight='bold', fontsize=9)\n        \n        # Plot 4: Sharpe improvement\n        ax4 = fig.add_subplot(gs[1, 2])\n        \n        sharpe_improvement = comparison_df['Sharpe_Improvement']\n        \n        bars = ax4.bar(sharpe_improvement.index, sharpe_improvement.values,\n                      color=['green' if x > 0 else 'red' for x in sharpe_improvement.values],\n                      alpha=0.7)\n        ax4.set_ylabel('Sharpe Improvement')\n        ax4.set_title('Contrarian vs Normal\\nSharpe Improvement', fontweight='bold')\n        ax4.grid(axis='y', alpha=0.3)\n        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)\n        ax4.tick_params(axis='x', rotation=45)\n        \n        # Add values on bars\n        for bar, value in zip(bars, sharpe_improvement.values):\n            height = bar.get_height()\n            ax4.text(bar.get_x() + bar.get_width()/2, height + (0.02 if height > 0 else -0.05),\n                    f'{value:+.2f}', ha='center', va='bottom' if height > 0 else 'top', \n                    fontweight='bold', fontsize=9)\n        \n        # Plot 5: Risk-return scatter (large plot)\n        ax5 = fig.add_subplot(gs[2, :])\n        \n        # Separate normal and contrarian for plotting\n        normal_returns = []\n        normal_vols = []\n        contrarian_returns = []\n        contrarian_vols = []\n        normal_names = []\n        contrarian_names = []\n        \n        for strategy_name, result in self.ab_test_results.items():\n            if strategy_name == 'Benchmark_EqualWeight':\n                continue\n            \n            annual_return = result['annual_return'] * 100\n            volatility = result['annual_volatility'] * 100\n            \n            if strategy_name.startswith('Normal_'):\n                normal_returns.append(annual_return)\n                normal_vols.append(volatility)\n                normal_names.append(strategy_name.replace('Normal_', ''))\n            else:\n                contrarian_returns.append(annual_return)\n                contrarian_vols.append(volatility)\n                contrarian_names.append(strategy_name.replace('Contrarian_', '').replace('Adaptive_Seasonal', 'Adaptive'))\n        \n        # Plot normal strategies\n        ax5.scatter(normal_vols, normal_returns, c='red', s=100, alpha=0.7, \n                   label='Normal Strategies', marker='s')\n        \n        # Plot contrarian strategies\n        ax5.scatter(contrarian_vols, contrarian_returns, c='blue', s=100, alpha=0.7, \n                   label='Contrarian Strategies', marker='o')\n        \n        # Add benchmark\n        if 'Benchmark_EqualWeight' in self.ab_test_results:\n            bench_return = self.ab_test_results['Benchmark_EqualWeight']['annual_return'] * 100\n            bench_vol = self.ab_test_results['Benchmark_EqualWeight']['annual_volatility'] * 100\n            ax5.scatter(bench_vol, bench_return, c='black', s=150, alpha=0.8, \n                       label='Benchmark', marker='*')\n        \n        # Add strategy labels\n        for i, name in enumerate(normal_names):\n            ax5.annotate(name, (normal_vols[i], normal_returns[i]), \n                        xytext=(5, 5), textcoords='offset points', fontsize=8)\n        \n        for i, name in enumerate(contrarian_names):\n            ax5.annotate(name, (contrarian_vols[i], contrarian_returns[i]), \n                        xytext=(5, 5), textcoords='offset points', fontsize=8)\n        \n        ax5.set_xlabel('Volatility (%)')\n        ax5.set_ylabel('Annual Return (%)')\n        ax5.set_title('Risk-Return Profile: Normal vs Contrarian Strategies', \n                     fontsize=14, fontweight='bold')\n        ax5.legend()\n        ax5.grid(True, alpha=0.3)\n        \n        plt.suptitle('A/B Testing Results: Normal vs Contrarian Seasonal Strategies', \n                    fontsize=16, fontweight='bold', y=0.98)\n        \n        plt.tight_layout()\n        plt.savefig('../results/ab_test_comparison.png', dpi=300, bbox_inches='tight')\n        plt.show()\n    \n    def generate_ab_summary_report(self) -> str:\n        \"\"\"Generate comprehensive A/B testing summary report\"\"\"\n        if not self.ab_test_results:\n            return \"No A/B test results available.\"\n        \n        comparison_df = self.analyze_ab_test_results()\n        \n        # Calculate overall statistics\n        contrarian_wins = (comparison_df['Contrarian_Wins'] == 'YES').sum()\n        total_comparisons = len(comparison_df)\n        avg_return_improvement = comparison_df['Return_Improvement_%'].mean()\n        avg_sharpe_improvement = comparison_df['Sharpe_Improvement'].mean()\n        \n        # Find best performing strategies\n        all_performance = [(k, v['sharpe_ratio']) for k, v in self.ab_test_results.items() \n                          if k != 'Benchmark_EqualWeight']\n        all_performance.sort(key=lambda x: x[1], reverse=True)\n        \n        report = f\"\"\"\n╔═══════════════════════════════════════════════════════════════════════════════╗\n║                     A/B TEST RESULTS SUMMARY REPORT                          ║\n╚═══════════════════════════════════════════════════════════════════════════════╝\n\n📊 OVERALL A/B TEST RESULTS:\n   • Contrarian strategies outperform: {contrarian_wins}/{total_comparisons} sectors\n   • Success rate: {contrarian_wins/total_comparisons*100:.1f}%\n   • Average return improvement: {avg_return_improvement:+.2f}%\n   • Average Sharpe improvement: {avg_sharpe_improvement:+.3f}\n\n🏆 TOP PERFORMING STRATEGIES:\n\"\"\"\n        \n        for i, (strategy, sharpe) in enumerate(all_performance[:3], 1):\n            strategy_type = \"CONTRARIAN\" if not strategy.startswith('Normal_') else \"NORMAL\"\n            report += f\"   {i}. {strategy.replace('_', ' ')} ({strategy_type}): {sharpe:.3f} Sharpe\\n\"\n        \n        report += f\"\"\"\n\n📈 SECTOR-BY-SECTOR ANALYSIS:\n\"\"\"\n        \n        for sector, row in comparison_df.iterrows():\n            winner = \"CONTRARIAN WINS\" if row['Contrarian_Wins'] == 'YES' else \"NORMAL WINS\"\n            report += f\"\"\"\n   {sector}:\n     • Return: {row['Normal_Annual_Return_%']:+.1f}% → {row['Contrarian_Annual_Return_%']:+.1f}% ({row['Return_Improvement_%']:+.1f}%)\n     • Sharpe: {row['Normal_Sharpe']:.2f} → {row['Contrarian_Sharpe']:.2f} ({row['Sharpe_Improvement']:+.2f})\n     • Result: {winner}\n\"\"\"\n        \n        # Key insights\n        best_improvement_sector = comparison_df['Return_Improvement_%'].idxmax()\n        best_improvement_value = comparison_df.loc[best_improvement_sector, 'Return_Improvement_%']\n        \n        worst_improvement_sector = comparison_df['Return_Improvement_%'].idxmin()\n        worst_improvement_value = comparison_df.loc[worst_improvement_sector, 'Return_Improvement_%']\n        \n        report += f\"\"\"\n\n💡 KEY INSIGHTS:\n   • Best contrarian opportunity: {best_improvement_sector} ({best_improvement_value:+.1f}% improvement)\n   • Weakest contrarian performance: {worst_improvement_sector} ({worst_improvement_value:+.1f}% change)\n   • Contrarian strategies show {'STRONG' if contrarian_wins >= total_comparisons/2 else 'WEAK'} evidence of pattern inversion\n   • Traditional seasonal logic appears to be {'INVERTED' if avg_return_improvement > 0 else 'INTACT'} in this period\n\n⚠️  IMPORTANT CONSIDERATIONS:\n   • Pattern inversions may be temporary or regime-dependent\n   • Transaction costs and implementation challenges affect real-world performance\n   • Continued monitoring needed to detect regime changes\n   • Risk management crucial given potential volatility of contrarian approaches\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nA/B Testing Complete • {len(self.ab_test_results)} strategies tested\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\"\"\"\n        \n        return report


def main():\n    \"\"\"Main execution function for contrarian strategy A/B testing\"\"\"\n    print(\"🚀 CONTRARIAN SEASONAL STRATEGY A/B TESTING\")\n    print(\"=\"*50)\n    \n    # Initialize A/B testing engine\n    engine = ContrarianBacktestEngine(transaction_cost=0.0010)\n    \n    # Run A/B comparison\n    results = engine.run_ab_strategy_comparison(\n        start_date='2015-01-01',\n        end_date='2025-08-01'\n    )\n    \n    # Generate comparison analysis\n    print(\"\\n\" + \"=\"*60)\n    print(\"A/B TEST COMPARISON ANALYSIS\")\n    print(\"=\"*60)\n    \n    comparison_df = engine.analyze_ab_test_results()\n    print(comparison_df.round(2))\n    \n    # Generate visualizations\n    print(\"\\nGenerating A/B test visualizations...\")\n    engine.plot_ab_comparison()\n    \n    # Generate summary report\n    summary = engine.generate_ab_summary_report()\n    print(summary)\n    \n    # Export results\n    print(\"\\nExporting A/B test results...\")\n    with open('../results/ab_test_summary_report.txt', 'w') as f:\n        f.write(summary)\n    \n    comparison_df.to_csv('../results/ab_test_comparison_table.csv')\n    \n    print(\"\\n✅ A/B testing complete! Check results/ directory for outputs.\")\n    \n    return results, comparison_df\n\n\nif __name__ == \"__main__\":\n    results = main()"