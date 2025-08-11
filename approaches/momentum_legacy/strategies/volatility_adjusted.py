"""
Volatility-adjusted momentum/contrarian strategy.
Uses z-scores to identify extreme price movements for mean reversion.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from data_loader import CommodityDataLoader
from signals import volatility_adjusted_signals
from backtest_engine import BacktestEngine
from portfolio import PortfolioOptimizer
import warnings
warnings.filterwarnings('ignore')

class VolatilityAdjustedStrategy:
    """
    Volatility-adjusted strategy using z-scores for signal generation.
    """
    
    def __init__(self, window: int = 20, threshold: float = 2.0, strategy_type: str = 'contrarian'):
        """
        Initialize volatility-adjusted strategy.
        
        Args:
            window: Rolling window for mean/std calculation
            threshold: Z-score threshold (standard deviations)
            strategy_type: 'momentum' or 'contrarian'
        """
        self.window = window
        self.threshold = threshold
        self.strategy_type = strategy_type
        self.name = f"Vol-Adjusted {strategy_type.title()} (W={window}, T={threshold})"
        
    def generate_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility-adjusted signals using z-scores.
        
        Args:
            returns: DataFrame of daily returns
        
        Returns:
            DataFrame of volatility-adjusted signals
        """
        print(f"📊 Generating {self.name} signals...")
        
        # Calculate rolling z-scores
        rolling_mean = returns.rolling(self.window).mean()
        rolling_std = returns.rolling(self.window).std()
        z_scores = (returns - rolling_mean) / rolling_std
        
        # Generate signals based on z-score thresholds
        signals = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        
        if self.strategy_type == 'contrarian':
            # Contrarian: fade extreme moves
            signals[z_scores > self.threshold] = -1   # Short extreme winners
            signals[z_scores < -self.threshold] = 1   # Long extreme losers
        else:  # momentum
            # Momentum: follow extreme moves  
            signals[z_scores > self.threshold] = 1    # Long extreme winners
            signals[z_scores < -self.threshold] = -1  # Short extreme losers
        
        # CRITICAL: Shift to avoid lookahead bias
        signals = signals.shift(1).dropna(how='all')
        
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
    
    def analyze_z_score_distribution(self, returns: pd.DataFrame) -> dict:
        """
        Analyze the distribution of z-scores to optimize thresholds.
        
        Args:
            returns: DataFrame of daily returns
        
        Returns:
            Dictionary with z-score statistics
        """
        print(f"📊 Analyzing z-score distribution...")
        
        # Calculate z-scores
        rolling_mean = returns.rolling(self.window).mean()
        rolling_std = returns.rolling(self.window).std()
        z_scores = (returns - rolling_mean) / rolling_std
        
        # Flatten z-scores for analysis
        z_flat = z_scores.values.flatten()
        z_flat = z_flat[~np.isnan(z_flat)]  # Remove NaN values
        
        analysis = {
            'mean': np.mean(z_flat),
            'std': np.std(z_flat),
            'skewness': pd.Series(z_flat).skew(),
            'kurtosis': pd.Series(z_flat).kurtosis(),
            'percentiles': {
                '1%': np.percentile(z_flat, 1),
                '5%': np.percentile(z_flat, 5),
                '10%': np.percentile(z_flat, 10),
                '90%': np.percentile(z_flat, 90),
                '95%': np.percentile(z_flat, 95),
                '99%': np.percentile(z_flat, 99)
            },
            'extreme_counts': {
                f'below_-{self.threshold}': np.sum(z_flat < -self.threshold),
                f'above_{self.threshold}': np.sum(z_flat > self.threshold),
                'total_observations': len(z_flat)
            }
        }
        
        # Calculate frequency of extreme events
        total_obs = analysis['extreme_counts']['total_observations']
        below_threshold = analysis['extreme_counts'][f'below_-{self.threshold}']
        above_threshold = analysis['extreme_counts'][f'above_{self.threshold}']
        
        print(f"📈 Z-score statistics:")
        print(f"   Mean: {analysis['mean']:.3f}")
        print(f"   Std: {analysis['std']:.3f}")
        print(f"   Skewness: {analysis['skewness']:.3f}")
        print(f"   Kurtosis: {analysis['kurtosis']:.3f}")
        print(f"📊 Extreme events:")
        print(f"   Below -{self.threshold}: {below_threshold:,} ({below_threshold/total_obs:.2%})")
        print(f"   Above +{self.threshold}: {above_threshold:,} ({above_threshold/total_obs:.2%})")
        print(f"   Total extreme: {(below_threshold+above_threshold)/total_obs:.2%}")
        
        return analysis
    
    def run_backtest(self, returns: pd.DataFrame, 
                    portfolio_type: str = "long_short") -> dict:
        """
        Run backtest for volatility-adjusted strategy.
        
        Args:
            returns: DataFrame of daily returns
            portfolio_type: Portfolio construction method
        
        Returns:
            Dictionary with backtest results
        """
        print(f"🚀 Running backtest: {self.name}")
        
        # Analyze z-score distribution first
        z_analysis = self.analyze_z_score_distribution(returns)
        
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
            'z_analysis': z_analysis,
            'strategy_name': self.name,
            'parameters': {
                'window': self.window,
                'threshold': self.threshold,
                'strategy_type': self.strategy_type
            }
        }

class AdaptiveVolatilityStrategy:
    """
    Adaptive volatility strategy that adjusts thresholds based on market conditions.
    """
    
    def __init__(self, base_threshold: float = 2.0, window: int = 20, 
                 adaptation_window: int = 252):
        """
        Initialize adaptive volatility strategy.
        
        Args:
            base_threshold: Base z-score threshold
            window: Rolling window for z-score calculation
            adaptation_window: Window for threshold adaptation
        """
        self.base_threshold = base_threshold
        self.window = window
        self.adaptation_window = adaptation_window
        self.name = f"Adaptive Vol Strategy (Base={base_threshold})"
        
    def generate_adaptive_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals with adaptive thresholds.
        
        Args:
            returns: DataFrame of daily returns
        
        Returns:
            DataFrame of adaptive volatility signals
        """
        print(f"📊 Generating {self.name} signals...")
        
        # Calculate base z-scores
        rolling_mean = returns.rolling(self.window).mean()
        rolling_std = returns.rolling(self.window).std()
        z_scores = (returns - rolling_mean) / rolling_std
        
        # Calculate adaptive thresholds based on market volatility regime
        market_returns = returns.mean(axis=1)  # Market-wide average
        market_vol = market_returns.rolling(self.adaptation_window).std() * np.sqrt(252)
        long_term_vol = market_vol.rolling(self.adaptation_window * 2).mean()
        
        # Adjust thresholds based on volatility regime
        vol_ratio = market_vol / long_term_vol
        adaptive_threshold = self.base_threshold * (1 + 0.5 * (vol_ratio - 1))  # Adjust by 50% of vol change
        adaptive_threshold = adaptive_threshold.clip(0.5, 4.0)  # Reasonable bounds
        
        # Generate signals with adaptive thresholds
        signals = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        
        for date in z_scores.index[self.adaptation_window:]:
            threshold = adaptive_threshold.loc[date]
            
            if not pd.isna(threshold):
                # Contrarian signals with adaptive threshold
                daily_z = z_scores.loc[date]
                signals.loc[date, daily_z > threshold] = -1   # Short extreme winners
                signals.loc[date, daily_z < -threshold] = 1   # Long extreme losers
        
        # Shift to avoid lookahead bias
        signals = signals.shift(1).dropna(how='all')
        
        return signals
    
    def run_backtest(self, returns: pd.DataFrame) -> dict:
        """Run backtest for adaptive volatility strategy."""
        signals = self.generate_adaptive_signals(returns)
        
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

def run_all_volatility_strategies(data_dir: str = 'raw') -> dict:
    """
    Run all volatility-adjusted strategies with different parameters.
    
    Args:
        data_dir: Directory containing commodity data
    
    Returns:
        Dictionary of all results
    """
    print("🚀 Running All Volatility-Adjusted Strategies")
    print("=" * 60)
    
    # Load data
    loader = CommodityDataLoader(data_dir)
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns(prices)
    
    results = {}
    
    # 1. Different z-score thresholds
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    windows = [10, 20, 30, 60]
    
    for window in windows:
        for threshold in thresholds:
            print(f"\n📊 Running volatility strategy (W={window}, T={threshold})...")
            strategy = VolatilityAdjustedStrategy(
                window=window, 
                threshold=threshold, 
                strategy_type='contrarian'
            )
            name = f'vol_w{window}_t{threshold}'
            results[name] = strategy.run_backtest(returns)
    
    # 2. Adaptive volatility strategy
    print(f"\n📊 Running adaptive volatility strategy...")
    adaptive_strategy = AdaptiveVolatilityStrategy()
    results['adaptive_vol'] = adaptive_strategy.run_backtest(returns)
    
    # 3. Compare momentum vs contrarian for best parameters
    best_params = [(20, 2.0), (30, 1.5), (60, 2.5)]
    
    for window, threshold in best_params:
        # Contrarian version
        contrarian_strategy = VolatilityAdjustedStrategy(
            window=window, threshold=threshold, strategy_type='contrarian'
        )
        name = f'contrarian_vol_w{window}_t{threshold}'
        results[name] = contrarian_strategy.run_backtest(returns)
        
        # Momentum version
        momentum_strategy = VolatilityAdjustedStrategy(
            window=window, threshold=threshold, strategy_type='momentum'
        )
        name = f'momentum_vol_w{window}_t{threshold}'
        results[name] = momentum_strategy.run_backtest(returns)
    
    # 4. Compare all results
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
    comparison.to_csv(results_dir / 'volatility_strategies_comparison.csv')
    
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
    # Run all volatility strategies
    results = run_all_volatility_strategies()
    
    # Find best performing strategy
    best_strategy = None
    best_sharpe = -999
    
    for name, result_dict in results.items():
        sharpe = result_dict['result'].metrics['sharpe_ratio']
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_strategy = name
    
    print(f"\n🏆 Best performing strategy: {best_strategy}")
    print(f"🏆 Sharpe ratio: {best_sharpe:.3f}")
    
    # Show parameter analysis for best strategy
    if best_strategy in results and 'parameters' in results[best_strategy]:
        params = results[best_strategy]['parameters']
        print(f"🏆 Best parameters: Window={params['window']}, Threshold={params['threshold']}")
    
    print("\n✅ Volatility-adjusted strategy analysis completed!")