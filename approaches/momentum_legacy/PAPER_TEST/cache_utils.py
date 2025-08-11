#!/usr/bin/env python3
"""
Cache Management Utilities for Commodity Cross-Sectional Backtest

Provides convenient functions for managing data and strategy caches,
including loading cached results in notebooks and scripts.

🚀 Generated with Claude Code
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from data_cache_manager import DataCacheManager

def load_cached_data(base_dir: str = ".") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convenience function to load all cached data.
    
    Args:
        base_dir: Base directory path
    
    Returns:
        Tuple of (daily_prices, monthly_prices, monthly_returns) or (None, None, None)
    """
    cache_manager = DataCacheManager(base_dir)
    cached_data = cache_manager.load_price_data()
    
    if cached_data[0] is not None:
        daily_prices, monthly_prices, monthly_returns, metadata = cached_data
        print(f"✅ Loaded cached data:")
        print(f"   Downloaded: {metadata['download_timestamp']}")
        print(f"   Commodities: {len(monthly_returns.columns)}")
        print(f"   Period: {monthly_returns.index.min().date()} to {monthly_returns.index.max().date()}")
        
        return daily_prices, monthly_prices, monthly_returns
    
    print("❌ No cached data available")
    return None, None, None

def load_strategy_results(base_dir: str = ".", top_n: int = None) -> Dict[str, pd.Series]:
    """
    Load strategy results from CSV files.
    
    Args:
        base_dir: Base directory path
        top_n: Number of top strategies to load (None = all)
    
    Returns:
        Dictionary mapping strategy names to return series
    """
    results_dir = Path(base_dir) / "results" / "series"
    
    if not results_dir.exists():
        print("❌ No strategy results found")
        return {}
    
    # Load summary to get performance ranking
    summary_file = Path(base_dir) / "results" / "summary" / "strategy_summary.csv"
    
    strategy_files = list(results_dir.glob("*_returns.csv"))
    if not strategy_files:
        print("❌ No strategy return files found")
        return {}
    
    strategies = {}
    
    # If we have a summary file, use it to prioritize loading
    if summary_file.exists() and top_n is not None:
        try:
            summary_df = pd.read_csv(summary_file)
            top_strategies = summary_df.nlargest(top_n, 'sharpe')['Strategy_Key'].tolist()
            
            for strategy_key in top_strategies:
                strategy_file = results_dir / f"{strategy_key}_returns.csv"
                if strategy_file.exists():
                    returns = pd.read_csv(strategy_file, index_col=0, parse_dates=True).squeeze()
                    strategies[strategy_key] = returns
            
            print(f"✅ Loaded top {len(strategies)} strategies by Sharpe ratio")
            
        except Exception as e:
            print(f"⚠️ Error loading summary file: {e}")
            # Fall back to loading all files
            top_n = None
    
    # Load all files if no summary or top_n not specified
    if top_n is None:
        for strategy_file in strategy_files:
            strategy_key = strategy_file.stem.replace('_returns', '')
            try:
                returns = pd.read_csv(strategy_file, index_col=0, parse_dates=True).squeeze()
                strategies[strategy_key] = returns
            except Exception as e:
                print(f"⚠️ Error loading {strategy_key}: {e}")
        
        print(f"✅ Loaded {len(strategies)} strategy results")
    
    return strategies

def get_performance_summary(base_dir: str = ".") -> Optional[pd.DataFrame]:
    """
    Load performance summary table.
    
    Args:
        base_dir: Base directory path
    
    Returns:
        Performance summary DataFrame or None
    """
    summary_file = Path(base_dir) / "results" / "summary" / "strategy_summary.csv"
    
    if not summary_file.exists():
        print("❌ No performance summary found")
        return None
    
    try:
        summary_df = pd.read_csv(summary_file)
        print(f"✅ Loaded performance summary: {len(summary_df)} strategies")
        return summary_df
    except Exception as e:
        print(f"❌ Error loading summary: {e}")
        return None

def quick_analysis_report(base_dir: str = ".") -> None:
    """
    Generate a quick analysis report of cached results.
    
    Args:
        base_dir: Base directory path
    """
    print("📊 QUICK ANALYSIS REPORT")
    print("=" * 50)
    
    # Check data cache
    cache_manager = DataCacheManager(base_dir)
    data_exists = cache_manager.download_metadata_file.exists()
    
    if data_exists:
        cached_data = cache_manager.load_price_data()
        if cached_data[0] is not None:
            _, _, monthly_returns, metadata = cached_data
            print(f"📅 Data Period: {metadata['start_date']} to {metadata['end_date']}")
            print(f"🎯 Commodities: {len(metadata['tickers_successful'])}")
            print(f"📊 Monthly Observations: {len(monthly_returns)}")
            
            # Data quality summary
            completeness = metadata['data_summary']['data_completeness']
            avg_completeness = sum(completeness.values()) / len(completeness) * 100
            print(f"📈 Avg Data Completeness: {avg_completeness:.1f}%")
    
    # Check strategy results
    summary_df = get_performance_summary(base_dir)
    if summary_df is not None:
        print(f"\n🏆 Strategy Results: {len(summary_df)} combinations tested")
        
        # Top performers
        valid_strategies = summary_df.dropna(subset=['sharpe'])
        if len(valid_strategies) > 0:
            top_3 = valid_strategies.nlargest(3, 'sharpe')
            print(f"📈 Valid Strategies: {len(valid_strategies)}")
            print(f"\n🥇 Top 3 by Sharpe Ratio:")
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"   {i}. {row['Strategy_Key']}: {row['sharpe']:.3f}")
            
            # Performance distribution
            print(f"\n📊 Performance Distribution:")
            print(f"   Mean Sharpe: {valid_strategies['sharpe'].mean():.3f}")
            print(f"   Std Sharpe: {valid_strategies['sharpe'].std():.3f}")
            print(f"   Positive Sharpe: {(valid_strategies['sharpe'] > 0).sum()}/{len(valid_strategies)}")
    
    # Cache recommendations
    print(f"\n💡 Recommendations:")
    recommendations = cache_manager.recommend_actions()
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

def create_quick_notebook_loader() -> str:
    """
    Generate code snippet for loading data in notebooks.
    
    Returns:
        Python code string for notebook loading
    """
    code = '''
# Quick data loader for Jupyter notebooks
from cache_utils import load_cached_data, load_strategy_results, get_performance_summary

# Load cached data
daily_prices, monthly_prices, monthly_returns = load_cached_data()

# Load top 10 strategy results
strategy_results = load_strategy_results(top_n=10)

# Load performance summary
summary_df = get_performance_summary()

print(f"📊 Data loaded: {len(monthly_returns.columns) if monthly_returns is not None else 0} commodities")
print(f"🏆 Strategies loaded: {len(strategy_results)}")
print(f"📈 Summary rows: {len(summary_df) if summary_df is not None else 0}")

# Display top strategies
if summary_df is not None:
    top_strategies = summary_df.nlargest(5, 'sharpe')[['Strategy_Key', 'ann_return', 'sharpe', 'max_dd']]
    print("\\n🏆 Top 5 Strategies:")
    print(top_strategies.round(4))
'''
    return code

# Convenience functions for specific data types
def get_momentum_results(base_dir: str = ".") -> Dict[str, pd.Series]:
    """Get only momentum strategy results."""
    all_results = load_strategy_results(base_dir)
    return {k: v for k, v in all_results.items() if 'momentum' in k.lower()}

def get_contrarian_results(base_dir: str = ".") -> Dict[str, pd.Series]:
    """Get only contrarian strategy results."""
    all_results = load_strategy_results(base_dir)
    return {k: v for k, v in all_results.items() if 'contrarian' in k.lower()}

def get_long_short_results(base_dir: str = ".") -> Dict[str, pd.Series]:
    """Get only long-short strategy results."""
    all_results = load_strategy_results(base_dir)
    return {k: v for k, v in all_results.items() if 'long_short' in k}

def create_equity_curves(strategy_results: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """
    Convert return series to equity curves.
    
    Args:
        strategy_results: Dictionary of return series
        
    Returns:
        Dictionary of equity curve series
    """
    equity_curves = {}
    for strategy_name, returns in strategy_results.items():
        equity_curves[strategy_name] = (1 + returns.fillna(0)).cumprod()
    
    return equity_curves

if __name__ == "__main__":
    # Demo functionality
    print("🧪 CACHE UTILITIES DEMO")
    print("=" * 40)
    
    quick_analysis_report()
    
    print("\n📝 Notebook loader code:")
    print(create_quick_notebook_loader())