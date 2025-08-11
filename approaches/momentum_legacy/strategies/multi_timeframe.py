"""
Multi-timeframe momentum strategy.
Combines momentum signals from different lookback periods (5D, 10D, 20D).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from data_loader import CommodityDataLoader
from signals import multi_timeframe_momentum, basic_momentum_signals
from backtest_engine import BacktestEngine
from portfolio import PortfolioOptimizer
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeMomentumStrategy:
    """
    Multi-timeframe momentum strategy combining different lookback periods.
    """
    
    def __init__(self, timeframes=[1, 5, 10, 20]):
        """
        Initialize multi-timeframe momentum strategy.
        
        Args:
            timeframes: List of lookback periods in days
        """
        self.timeframes = timeframes
        self.name = f"Multi-Timeframe Momentum ({','.join(map(str, timeframes))}D)"
        
    def generate_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-timeframe momentum signals.
        
        Args:
            returns: DataFrame of daily returns
        
        Returns:
            DataFrame of combined momentum signals
        """
        print(f"📊 Generating {self.name} signals...")
        
        # Use the multi_timeframe_momentum function from signals.py
        signals = multi_timeframe_momentum(returns, self.timeframes)
        
        print(f"✅ Generated signals for {len(signals.columns)} commodities")
        print(f"📅 Signal period: {signals.index.min().date()} to {signals.index.max().date()}")
        
        # Calculate signal statistics
        long_signals = (signals == 1).sum().sum()
        short_signals = (signals == -1).sum().sum()
        neutral_signals = (signals == 0).sum().sum()
        
        print(f"📈 Long signals: {long_signals:,}")
        print(f"📉 Short signals: {short_signals:,}")
        print(f"⚪ Neutral signals: {neutral_signals:,}")
        
        return signals
    
    def run_backtest(self, returns: pd.DataFrame, 
                    portfolio_type: str = "long_short") -> dict:
        """
        Run backtest for multi-timeframe momentum strategy.
        
        Args:
            returns: DataFrame of daily returns
            portfolio_type: Portfolio construction method
        
        Returns:
            Dictionary with backtest results
        """
        print(f"🚀 Running backtest: {self.name}")
        
        # Generate signals
        signals = self.generate_signals(returns)
        
        # Run backtest
        engine = BacktestEngine(returns)
        result = engine.run_backtest(
            signals=signals,
            strategy_name=self.name,
            portfolio_type=portfolio_type
        )
        
        return {
            'result': result,
            'signals': signals,
            'strategy_name': self.name
        }

class IndividualTimeframeMomentumStrategy:
    """
    Individual timeframe momentum strategies for comparison.
    """
    
    def __init__(self, timeframe: int):
        """
        Initialize individual timeframe strategy.
        
        Args:
            timeframe: Lookback period in days
        """
        self.timeframe = timeframe
        self.name = f"Momentum {timeframe}D"
        
    def generate_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals for specific timeframe.
        """
        print(f"📊 Generating {self.name} signals...")
        
        # Use basic momentum with specific lookback
        signals = basic_momentum_signals(returns, lookback=self.timeframe)
        
        # Convert to long/short signals (contrarian style for better performance)
        contrarian_signals = -signals  # Flip momentum to contrarian
        contrarian_signals[signals == 0] = 0  # Keep neutral as neutral
        
        return contrarian_signals
    
    def run_backtest(self, returns: pd.DataFrame,
                    portfolio_type: str = "long_short") -> dict:
        """Run backtest for individual timeframe."""
        signals = self.generate_signals(returns)
        
        engine = BacktestEngine(returns)
        result = engine.run_backtest(
            signals=signals,
            strategy_name=self.name,
            portfolio_type=portfolio_type
        )
        
        return {
            'result': result,
            'signals': signals,
            'strategy_name': self.name
        }

def run_all_timeframe_strategies(data_dir: str = 'raw') -> dict:
    """
    Run all timeframe momentum strategies and compare performance.
    
    Args:
        data_dir: Directory containing commodity data
    
    Returns:
        Dictionary of all results
    """
    print("🚀 Running All Timeframe Momentum Strategies")
    print("=" * 60)
    
    # Load data
    loader = CommodityDataLoader(data_dir)
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns(prices)
    
    results = {}
    
    # 1. Individual timeframe strategies
    individual_timeframes = [1, 5, 10, 20, 60]  # Include monthly
    
    for tf in individual_timeframes:
        print(f"\n📊 Running {tf}-day momentum strategy...")
        strategy = IndividualTimeframeMomentumStrategy(tf)
        results[f'momentum_{tf}d'] = strategy.run_backtest(returns)
    
    # 2. Multi-timeframe combinations
    combinations = [
        [1, 5],
        [1, 5, 10], 
        [5, 10, 20],
        [1, 5, 10, 20],
        [1, 10, 20, 60]
    ]
    
    for combo in combinations:
        print(f"\n📊 Running multi-timeframe strategy {combo}...")
        strategy = MultiTimeframeMomentumStrategy(combo)
        combo_name = f"multi_tf_{'_'.join(map(str, combo))}"
        results[combo_name] = strategy.run_backtest(returns)
    
    # 3. Compare all results
    print("\n📈 Performance Comparison")
    print("=" * 60)
    
    backtest_results = {}
    for name, result_dict in results.items():
        backtest_results[name] = result_dict['result']
    
    engine = BacktestEngine(returns)
    comparison = engine.compare_strategies(backtest_results)
    
    print(comparison[['total_return', 'annualized_return', 'volatility', 
                     'sharpe_ratio', 'max_drawdown']].round(4))
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save comparison table
    comparison.to_csv(results_dir / 'timeframe_strategies_comparison.csv')
    
    # Save individual results
    for name, result_dict in results.items():
        result = result_dict['result']
        
        # Save equity curve and returns
        result.equity_curve.to_csv(results_dir / f'{name}_equity.csv')
        result.strategy_returns.to_csv(results_dir / f'{name}_returns.csv')
        result.weights.to_csv(results_dir / f'{name}_weights.csv')
    
    print(f"\n💾 Results saved to {results_dir}/")
    
    return results

if __name__ == "__main__":
    # Run all timeframe strategies
    results = run_all_timeframe_strategies()
    
    # Print best performing strategy
    best_strategy = None
    best_sharpe = -999
    
    for name, result_dict in results.items():
        sharpe = result_dict['result'].metrics['sharpe_ratio']
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_strategy = name
    
    print(f"\n🏆 Best performing strategy: {best_strategy}")
    print(f"🏆 Sharpe ratio: {best_sharpe:.3f}")
    
    print("\n✅ Multi-timeframe momentum analysis completed!")