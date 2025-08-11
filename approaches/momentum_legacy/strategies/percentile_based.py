"""
Percentile-based momentum/contrarian strategy.
Uses cross-sectional ranking to identify top/bottom performers.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from data_loader import CommodityDataLoader
from signals import percentile_based_signals
from backtest_engine import BacktestEngine
import warnings
warnings.filterwarnings('ignore')

class PercentileBasedStrategy:
    """
    Percentile-based strategy using cross-sectional ranking.
    """
    
    def __init__(self, top_pct: float = 0.2, bottom_pct: float = 0.2, 
                 strategy_type: str = 'contrarian', window: int = 1):
        """
        Initialize percentile-based strategy.
        
        Args:
            top_pct: Percentile for top performers (0.2 = top 20%)
            bottom_pct: Percentile for bottom performers
            strategy_type: 'momentum' or 'contrarian'
            window: Lookback window for performance calculation
        """
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.strategy_type = strategy_type
        self.window = window
        self.name = f"Percentile {strategy_type.title()} (T={top_pct:.0%}, B={bottom_pct:.0%}, W={window})"
        
    def generate_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate percentile-based signals using cross-sectional ranking.
        
        Args:
            returns: DataFrame of daily returns
        
        Returns:
            DataFrame of percentile-based signals
        """
        print(f"📊 Generating {self.name} signals...")
        
        # Use the percentile_based_signals function
        signals = percentile_based_signals(
            returns, 
            window=252,  # Use longer window for more stable percentiles
            top_pct=self.top_pct,
            bottom_pct=self.bottom_pct,
            strategy_type=self.strategy_type
        )
        
        print(f"✅ Generated signals for {len(signals.columns)} commodities")
        print(f"📅 Signal period: {signals.index.min().date()} to {signals.index.max().date()}")
        
        # Calculate signal statistics
        long_signals = (signals == 1).sum().sum()
        short_signals = (signals == -1).sum().sum()
        neutral_signals = (signals == 0).sum().sum()
        total_signals = long_signals + short_signals + neutral_signals
        
        print(f"📈 Long signals: {long_signals:,} ({long_signals/total_signals:.1%})")
        print(f"📉 Short signals: {short_signals:,} ({short_signals/total_signals:.1%})")
        print(f"⚪ Neutral signals: {neutral_signals:,} ({neutral_signals/total_signals:.1%})")
        
        return signals
    
    def run_backtest(self, returns: pd.DataFrame, 
                    portfolio_type: str = "long_short") -> dict:
        """
        Run backtest for percentile-based strategy.
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
            'strategy_name': self.name,
            'parameters': {
                'top_pct': self.top_pct,
                'bottom_pct': self.bottom_pct,
                'strategy_type': self.strategy_type,
                'window': self.window
            }
        }

def run_all_percentile_strategies(data_dir: str = 'raw') -> dict:
    """
    Run all percentile-based strategies with different parameters.
    """
    print("🚀 Running All Percentile-Based Strategies")
    print("=" * 60)
    
    # Load data
    loader = CommodityDataLoader(data_dir)
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns(prices)
    
    results = {}
    
    # Different percentile configurations
    configurations = [
        (0.1, 0.1, 'contrarian'),  # Top/bottom 10%
        (0.2, 0.2, 'contrarian'),  # Top/bottom 20%  
        (0.3, 0.3, 'contrarian'),  # Top/bottom 30%
        (0.2, 0.2, 'momentum'),    # Momentum version
        (0.1, 0.3, 'contrarian'),  # Asymmetric
        (0.3, 0.1, 'contrarian'),  # Asymmetric reverse
    ]
    
    for top_pct, bottom_pct, strategy_type in configurations:
        print(f"\n📊 Running {strategy_type} strategy (T={top_pct:.0%}, B={bottom_pct:.0%})...")
        strategy = PercentileBasedStrategy(
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            strategy_type=strategy_type
        )
        name = f'{strategy_type}_pct_t{int(top_pct*100)}_b{int(bottom_pct*100)}'
        results[name] = strategy.run_backtest(returns)
    
    # Compare results
    print("\n📈 Performance Comparison")
    print("=" * 60)
    
    backtest_results = {name: result_dict['result'] for name, result_dict in results.items()}
    
    engine = BacktestEngine(returns)
    comparison = engine.compare_strategies(backtest_results)
    
    print(comparison[['total_return', 'annualized_return', 'volatility', 
                     'sharpe_ratio', 'max_drawdown']].round(4))
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    comparison.to_csv(results_dir / 'percentile_strategies_comparison.csv')
    
    for name, result_dict in results.items():
        result = result_dict['result']
        result.equity_curve.to_csv(results_dir / f'{name}_equity.csv')
        result.strategy_returns.to_csv(results_dir / f'{name}_returns.csv')
    
    return results

if __name__ == "__main__":
    results = run_all_percentile_strategies()
    
    # Find best strategy
    best_strategy = max(results.items(), 
                       key=lambda x: x[1]['result'].metrics['sharpe_ratio'])
    
    print(f"\n🏆 Best strategy: {best_strategy[0]}")
    print(f"🏆 Sharpe ratio: {best_strategy[1]['result'].metrics['sharpe_ratio']:.3f}")
    
    print("\n✅ Percentile-based strategy analysis completed!")