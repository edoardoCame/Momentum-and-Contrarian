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
        """
        print("🔬 A/B TESTING: Normal vs Contrarian Seasonal Strategies")
        print("="*60)
        
        # Load data
        self.analyzer.load_commodity_data()
        returns_data = self.analyzer.extract_returns_data()
        
        # Filter date range
        mask = (returns_data.index >= start_date) & (returns_data.index <= end_date)
        returns_data = returns_data.loc[mask]
        
        # Calculate seasonal statistics
        seasonal_stats = self.analyzer.get_seasonal_summary_stats(returns_data)
        
        print(f"Backtesting period: {start_date} to {end_date}")
        print(f"Total observations: {len(returns_data):,} days")
        
        # Initialize strategies
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
            'Contrarian_Sector_Rotation': ContrarianSectorRotationStrategy()
        }
        
        # Combine strategies
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
            
            # Print summary
            annual_return = result['annual_return'] * 100
            sharpe = result['sharpe_ratio']
            max_dd = result['max_drawdown'] * 100
            
            print(f"  📈 Annual Return: {annual_return:+.2f}%")
            print(f"  ⚡ Sharpe Ratio: {sharpe:.2f}")
            print(f"  📉 Max Drawdown: {max_dd:.2f}%")
        
        # Add benchmark
        benchmark_returns = returns_data.mean(axis=1)
        benchmark_result = self._calculate_benchmark_performance(benchmark_returns)
        strategy_results['Benchmark_EqualWeight'] = benchmark_result
        
        self.ab_test_results = strategy_results
        return strategy_results
    
    def analyze_ab_test_results(self) -> pd.DataFrame:
        """Analyze A/B test results comparing normal vs contrarian strategies"""
        if not self.ab_test_results:
            raise ValueError("No A/B test results available.")
        
        # Compare sector strategies
        sector_pairs = [
            ('Normal_Energy', 'Contrarian_Energy'),
            ('Normal_Agricultural', 'Contrarian_Agricultural'),
            ('Normal_Metals', 'Contrarian_Metals'),
            ('Normal_Sector_Rotation', 'Contrarian_Sector_Rotation')
        ]
        
        comparison_data = []
        
        for normal_key, contrarian_key in sector_pairs:
            if normal_key in self.ab_test_results and contrarian_key in self.ab_test_results:
                normal = self.ab_test_results[normal_key]
                contrarian = self.ab_test_results[contrarian_key]
                
                sector = normal_key.replace('Normal_', '')
                
                comparison_data.append({
                    'Sector': sector,
                    'Normal_Return_%': normal['annual_return'] * 100,
                    'Contrarian_Return_%': contrarian['annual_return'] * 100,
                    'Improvement_%': (contrarian['annual_return'] - normal['annual_return']) * 100,
                    'Normal_Sharpe': normal['sharpe_ratio'],
                    'Contrarian_Sharpe': contrarian['sharpe_ratio'],
                    'Sharpe_Improvement': contrarian['sharpe_ratio'] - normal['sharpe_ratio'],
                    'Contrarian_Wins': 'YES' if contrarian['sharpe_ratio'] > normal['sharpe_ratio'] else 'NO'
                })
        
        return pd.DataFrame(comparison_data).set_index('Sector')
    
    def generate_summary_report(self) -> str:
        """Generate A/B testing summary report"""
        if not self.ab_test_results:
            return "No results available."
        
        comparison_df = self.analyze_ab_test_results()
        
        # Calculate statistics
        contrarian_wins = (comparison_df['Contrarian_Wins'] == 'YES').sum()
        total_comparisons = len(comparison_df)
        avg_improvement = comparison_df['Improvement_%'].mean()
        
        # Find best performers
        all_performance = [(k, v['sharpe_ratio']) for k, v in self.ab_test_results.items() 
                          if k != 'Benchmark_EqualWeight']
        all_performance.sort(key=lambda x: x[1], reverse=True)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      A/B TEST RESULTS SUMMARY                       ║
╚══════════════════════════════════════════════════════════════════════╝

📊 OVERALL RESULTS:
   • Contrarian wins: {contrarian_wins}/{total_comparisons} sectors
   • Success rate: {contrarian_wins/total_comparisons*100:.1f}%
   • Average improvement: {avg_improvement:+.2f}%

🏆 TOP STRATEGIES:
"""
        
        for i, (strategy, sharpe) in enumerate(all_performance[:3], 1):
            strategy_type = "CONTRARIAN" if not strategy.startswith('Normal_') else "NORMAL"
            report += f"   {i}. {strategy.replace('_', ' ')} ({strategy_type}): {sharpe:.3f}\n"
        
        report += f"""

📈 SECTOR ANALYSIS:
"""
        
        for sector, row in comparison_df.iterrows():
            winner = "✅ CONTRARIAN WINS" if row['Contrarian_Wins'] == 'YES' else "❌ NORMAL WINS"
            report += f"""
   {sector}:
     • Normal: {row['Normal_Return_%']:+.1f}% return, {row['Normal_Sharpe']:.2f} Sharpe
     • Contrarian: {row['Contrarian_Return_%']:+.1f}% return, {row['Contrarian_Sharpe']:.2f} Sharpe
     • Result: {winner}
"""
        
        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return report


def main():
    """Main execution function"""
    print("🚀 CONTRARIAN SEASONAL STRATEGY A/B TESTING")
    print("="*50)
    
    # Initialize engine
    engine = ContrarianBacktestEngine(transaction_cost=0.0010)
    
    # Run A/B comparison
    results = engine.run_ab_strategy_comparison(
        start_date='2015-01-01',
        end_date='2025-08-01'
    )
    
    # Analyze results
    print("\n" + "="*60)
    print("A/B TEST RESULTS")
    print("="*60)
    
    comparison_df = engine.analyze_ab_test_results()
    print(comparison_df.round(2))
    
    # Generate summary
    summary = engine.generate_summary_report()
    print(summary)
    
    return results, comparison_df


if __name__ == "__main__":
    results = main()