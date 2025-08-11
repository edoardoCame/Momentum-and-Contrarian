"""
Cross-sectoral momentum strategy.
Momentum/contrarian signals based on performance relative to sector.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from data_loader import CommodityDataLoader
from signals import cross_sectoral_momentum
from backtest_engine import BacktestEngine
import warnings
warnings.filterwarnings('ignore')

class CrossSectoralStrategy:
    """
    Cross-sectoral momentum strategy.
    """
    
    def __init__(self, lookback: int = 5):
        """
        Initialize cross-sectoral strategy.
        
        Args:
            lookback: Lookback period for momentum calculation
        """
        self.lookback = lookback
        self.name = f"Cross-Sectoral Momentum ({lookback}D)"
        
    def run_backtest(self, returns: pd.DataFrame, sectors_map: dict) -> dict:
        """
        Run backtest for cross-sectoral strategy.
        """
        print(f"🚀 Running backtest: {self.name}")
        
        # Generate signals
        signals = cross_sectoral_momentum(returns, sectors_map, self.lookback)
        
        # Run backtest
        engine = BacktestEngine(returns)
        result = engine.run_backtest(
            signals=signals,
            strategy_name=self.name,
            portfolio_type="long_short"
        )
        
        return {
            'result': result,
            'signals': signals,
            'strategy_name': self.name
        }

def run_cross_sectoral_strategies(data_dir: str = 'raw') -> dict:
    """
    Run cross-sectoral strategies.
    """
    print("🚀 Running Cross-Sectoral Strategies")
    print("=" * 50)
    
    # Load data
    loader = CommodityDataLoader(data_dir)
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns(prices)
    sectors_map = loader.get_sector_mapping()
    
    results = {}
    
    # Different lookback periods
    lookbacks = [1, 5, 10, 20]
    
    for lookback in lookbacks:
        print(f"\n📊 Running cross-sectoral strategy ({lookback}D)...")
        strategy = CrossSectoralStrategy(lookback)
        results[f'cross_sectoral_{lookback}d'] = strategy.run_backtest(returns, sectors_map)
    
    # Compare results
    backtest_results = {name: result_dict['result'] for name, result_dict in results.items()}
    
    engine = BacktestEngine(returns)
    comparison = engine.compare_strategies(backtest_results)
    
    print("\n📈 Performance Comparison")
    print("=" * 50)
    print(comparison[['total_return', 'sharpe_ratio', 'max_drawdown']].round(4))
    
    return results

if __name__ == "__main__":
    results = run_cross_sectoral_strategies()
    print("✅ Cross-sectoral analysis completed!")