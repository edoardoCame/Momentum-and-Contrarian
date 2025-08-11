"""
Regime-adaptive strategy.
Switches between momentum and contrarian based on market conditions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from data_loader import CommodityDataLoader
from signals import regime_adaptive_signals
from backtest_engine import BacktestEngine
import warnings
warnings.filterwarnings('ignore')

class RegimeAdaptiveStrategy:
    """
    Regime-adaptive strategy that switches between momentum and contrarian.
    """
    
    def __init__(self, regime_window: int = 60, volatility_threshold: float = 1.5):
        """
        Initialize regime-adaptive strategy.
        
        Args:
            regime_window: Window for regime detection
            volatility_threshold: Threshold for high/low volatility regime
        """
        self.regime_window = regime_window
        self.volatility_threshold = volatility_threshold
        self.name = f"Regime Adaptive (W={regime_window}, VT={volatility_threshold})"
        
    def run_backtest(self, returns: pd.DataFrame) -> dict:
        """
        Run backtest for regime-adaptive strategy.
        """
        print(f"🚀 Running backtest: {self.name}")
        
        # Generate adaptive signals
        signals = regime_adaptive_signals(returns, self.regime_window, self.volatility_threshold)
        
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

def run_regime_adaptive_strategies(data_dir: str = 'raw') -> dict:
    """
    Run regime-adaptive strategies.
    """
    print("🚀 Running Regime-Adaptive Strategies")
    print("=" * 50)
    
    # Load data
    loader = CommodityDataLoader(data_dir)
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns(prices)
    
    results = {}
    
    # Different configurations
    configs = [
        (60, 1.5),
        (120, 1.2),
        (30, 2.0)
    ]
    
    for regime_window, vol_threshold in configs:
        print(f"\n📊 Running regime adaptive (W={regime_window}, VT={vol_threshold})...")
        strategy = RegimeAdaptiveStrategy(regime_window, vol_threshold)
        name = f'regime_w{regime_window}_vt{int(vol_threshold*10)}'
        results[name] = strategy.run_backtest(returns)
    
    # Compare results
    backtest_results = {name: result_dict['result'] for name, result_dict in results.items()}
    
    engine = BacktestEngine(returns)
    comparison = engine.compare_strategies(backtest_results)
    
    print("\n📈 Performance Comparison")
    print("=" * 50)
    print(comparison[['total_return', 'sharpe_ratio', 'max_drawdown']].round(4))
    
    return results

if __name__ == "__main__":
    results = run_regime_adaptive_strategies()
    print("✅ Regime-adaptive analysis completed!")