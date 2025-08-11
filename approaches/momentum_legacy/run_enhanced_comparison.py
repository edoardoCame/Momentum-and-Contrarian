#!/usr/bin/env python3
"""
Script to run comparison between basic contrarian and enhanced multi-timeframe contrarian.
"""
import sys
sys.path.append('src')

from data_loader import CommodityDataLoader
from backtest_engine import BacktestEngine
from signals import (
    basic_contrarian_signals,
    volatility_adjusted_signals,
    multi_timeframe_contrarian_enhanced,
    multi_timeframe_contrarian_simplified
)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 ENHANCED CONTRARIAN STRATEGY COMPARISON")
    print("=" * 60)
    
    # Load data
    print("📊 Loading commodity data...")
    loader = CommodityDataLoader('raw')
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns(prices)
    
    # Print data summary
    summary = loader.get_data_summary()
    print(f"✅ Loaded {summary['n_commodities']} commodities")
    print(f"📅 Period: {summary['date_range'][0].date()} to {summary['date_range'][1].date()}")
    print(f"📊 Total trading days: {summary['trading_days']:,}")
    
    # Initialize backtest engine
    engine = BacktestEngine(returns)
    
    # Define strategies to compare
    strategies = {
        'Basic Contrarian 1D': basic_contrarian_signals(returns, lookback=1),
        'Basic Contrarian 5D': basic_contrarian_signals(returns, lookback=5), 
        'Volatility-Adjusted (20,2.0)': volatility_adjusted_signals(returns, window=20, threshold=2.0),
        'Enhanced Multi-TF Contrarian': multi_timeframe_contrarian_enhanced(returns, prices),
        'Simplified Multi-TF Contrarian': multi_timeframe_contrarian_simplified(returns, prices)
    }
    
    print(f"\n🔄 Running backtest for {len(strategies)} strategies...")
    results = {}
    
    # Run backtests
    for name, signals in strategies.items():
        print(f"  📈 Testing: {name}")
        try:
            result = engine.run_backtest(
                signals=signals,
                strategy_name=name,
                portfolio_type="long_short"
            )
            results[name] = result
        except Exception as e:
            print(f"    ⚠️ Failed: {e}")
    
    # Generate comparison
    print(f"\n📊 STRATEGY COMPARISON RESULTS")
    print("=" * 60)
    
    comparison = engine.compare_strategies(results)
    
    # Enhanced display
    print(f"\n🏆 STRATEGY RANKINGS (by Sharpe Ratio)")
    print("-" * 50)
    
    # Sort by Sharpe ratio
    sorted_strategies = comparison.sort_values('sharpe_ratio', ascending=False)
    
    for i, (name, row) in enumerate(sorted_strategies.iterrows(), 1):
        print(f"{i}. {name}")
        print(f"   📈 Total Return: {row['total_return']:.2%}")
        print(f"   📊 Annual Return: {row['annualized_return']:.2%}")
        print(f"   📈 Sharpe Ratio: {row['sharpe_ratio']:.3f}")
        print(f"   📉 Max Drawdown: {row['max_drawdown']:.2%}")
        print(f"   🎯 Win Rate: {row['win_rate']:.2%}")
        print()
    
    # Save results
    comparison.to_csv('results/enhanced_contrarian_comparison.csv')
    
    # Save individual equity curves
    for name, result in results.items():
        clean_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
        result.equity_curve.to_csv(f'results/{clean_name}_equity.csv')
        result.strategy_returns.to_csv(f'results/{clean_name}_returns.csv')
    
    print("💾 Results saved to results/ directory")
    print("\n✅ ENHANCED CONTRARIAN COMPARISON COMPLETED!")
    
    # Show improvement
    basic_sharpe = comparison.loc['Basic Contrarian 1D', 'sharpe_ratio']
    enhanced_sharpe = comparison.loc['Enhanced Multi-TF Contrarian', 'sharpe_ratio']
    improvement = (enhanced_sharpe - basic_sharpe) / basic_sharpe * 100
    
    print(f"\n🎯 IMPROVEMENT ANALYSIS:")
    print(f"Basic Contrarian Sharpe: {basic_sharpe:.3f}")
    print(f"Enhanced Contrarian Sharpe: {enhanced_sharpe:.3f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    return results, comparison

if __name__ == "__main__":
    results, comparison = main()