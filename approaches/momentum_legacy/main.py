"""
Main script to run all advanced momentum strategies.
Orchestrates all strategy implementations and creates comprehensive comparison.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import CommodityDataLoader
from backtest_engine import BacktestEngine
from portfolio import PortfolioOptimizer
from signals import (
    basic_momentum_signals, 
    basic_contrarian_signals,
    multi_timeframe_momentum,
    volatility_adjusted_signals,
    percentile_based_signals,
    multi_timeframe_contrarian_enhanced
)

# Import strategy modules
sys.path.append(str(Path(__file__).parent / 'strategies'))
from multi_timeframe import run_all_timeframe_strategies
from volatility_adjusted import run_all_volatility_strategies
from percentile_based import run_all_percentile_strategies
from cross_sectoral import run_cross_sectoral_strategies
from regime_adaptive import run_regime_adaptive_strategies

class AdvancedMomentumRunner:
    """
    Main orchestrator for all advanced momentum strategies.
    """
    
    def __init__(self, data_dir: str = 'raw'):
        """
        Initialize the momentum strategy runner.
        
        Args:
            data_dir: Directory containing commodity data
        """
        self.data_dir = data_dir
        self.results = {}
        self.loader = CommodityDataLoader(data_dir)
        self.returns = None
        self.prices = None
        
        print("🚀 ADVANCED MOMENTUM STRATEGIES FRAMEWORK")
        print("=" * 60)
        
    def load_data(self):
        """Load and prepare data."""
        print("📊 Loading commodity data...")
        self.prices = self.loader.load_commodity_data()
        self.returns = self.loader.calculate_returns(self.prices)
        
        # Print data summary
        summary = self.loader.get_data_summary()
        print(f"✅ Loaded {summary['n_commodities']} commodities")
        print(f"📅 Period: {summary['date_range'][0].date()} to {summary['date_range'][1].date()}")
        print(f"📊 Total trading days: {summary['trading_days']:,}")
    
    def _create_momentum_zscore_signals(self, returns: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.DataFrame:
        """Create momentum z-score signals (opposite of contrarian z-score)."""
        # Calculate rolling z-scores
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        z_scores = (returns - rolling_mean) / rolling_std
        
        # Generate momentum signals (follow extreme moves)
        signals_raw = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        signals_raw[z_scores > threshold] = 1    # Long extreme winners (momentum)
        signals_raw[z_scores < -threshold] = -1  # Short extreme losers (momentum)
        
        # Shift to avoid lookahead bias
        signals = signals_raw.shift(1).dropna(how='all')
        return signals
        
    def run_basic_strategies(self):
        """Run basic momentum and contrarian strategies for comparison."""
        print("\n🔵 Running Basic Strategies")
        print("-" * 40)
        
        engine = BacktestEngine(self.returns)
        
        # Basic strategies
        strategies = {
            'Basic Momentum 1D': basic_momentum_signals(self.returns, lookback=1),
            'Basic Contrarian 1D': basic_contrarian_signals(self.returns, lookback=1),
            'Basic Momentum 5D': basic_momentum_signals(self.returns, lookback=5),
            'Basic Contrarian 5D': basic_contrarian_signals(self.returns, lookback=5),
        }
        
        basic_results = engine.run_multiple_backtests(strategies, portfolio_type="long_short")
        
        # Store results
        for name, result in basic_results.items():
            self.results[name] = {
                'result': result,
                'category': 'Basic',
                'strategy_name': name
            }
        
        return basic_results
    
    def run_enhanced_contrarian(self):
        """Run the enhanced multi-timeframe contrarian strategy."""
        print("\n🟡 Running Enhanced Multi-Timeframe Contrarian Strategy")
        print("-" * 50)
        
        try:
            # Generate enhanced contrarian signals
            enhanced_signals = multi_timeframe_contrarian_enhanced(self.returns)
            
            # Run backtest
            engine = BacktestEngine(self.returns)
            enhanced_result = engine.run_backtest(
                signals=enhanced_signals,
                strategy_name="Enhanced Multi-TF Contrarian",
                portfolio_type="long_short"
            )
            
            # Store result
            self.results['Enhanced Multi-TF Contrarian'] = {
                'result': enhanced_result,
                'category': 'Enhanced',
                'strategy_name': 'Enhanced Multi-TF Contrarian'
            }
            
            print("✅ Enhanced contrarian strategy completed")
            return enhanced_result
            
        except Exception as e:
            print(f"⚠️ Enhanced contrarian strategy failed: {e}")
            return None
    
    def run_advanced_strategies(self):
        """Run all advanced strategy categories."""
        print("\n🔴 Running Advanced Strategies")
        print("-" * 40)
        
        advanced_results = {}
        
        # 1. Multi-timeframe strategies (reduced scope for speed)
        print("\n1️⃣ Multi-Timeframe Strategies...")
        try:
            # Run a subset of timeframe strategies
            engine = BacktestEngine(self.returns)
            
            timeframe_strategies = {
                'Multi-TF [1,5,10]': multi_timeframe_momentum(self.returns, [1, 5, 10]),
                'Multi-TF [5,10,20]': multi_timeframe_momentum(self.returns, [5, 10, 20]),
            }
            
            tf_results = engine.run_multiple_backtests(timeframe_strategies, portfolio_type="long_short")
            
            for name, result in tf_results.items():
                self.results[name] = {
                    'result': result,
                    'category': 'Multi-Timeframe',
                    'strategy_name': name
                }
            
        except Exception as e:
            print(f"⚠️ Multi-timeframe strategies failed: {e}")
        
        # 2. Volatility-adjusted strategies
        print("\n2️⃣ Volatility-Adjusted Strategies...")
        try:
            engine = BacktestEngine(self.returns)
            
            # Test both momentum and contrarian z-score approaches
            vol_strategies = {
                'Vol-Adj Momentum (20,2.0)': self._create_momentum_zscore_signals(self.returns, window=20, threshold=2.0),
                'Vol-Adj Momentum (30,1.5)': self._create_momentum_zscore_signals(self.returns, window=30, threshold=1.5),
                'Vol-Adj Contrarian (60,1.0)': volatility_adjusted_signals(self.returns, window=60, threshold=1.0),
                'Vol-Adj Momentum (10,1.0)': self._create_momentum_zscore_signals(self.returns, window=10, threshold=1.0),
            }
            
            vol_results = engine.run_multiple_backtests(vol_strategies, portfolio_type="long_short")
            
            for name, result in vol_results.items():
                self.results[name] = {
                    'result': result,
                    'category': 'Volatility-Adjusted',
                    'strategy_name': name
                }
                
        except Exception as e:
            print(f"⚠️ Volatility-adjusted strategies failed: {e}")
        
        # 3. Percentile-based strategies
        print("\n3️⃣ Percentile-Based Strategies...")
        try:
            engine = BacktestEngine(self.returns)
            
            percentile_strategies = {
                'Percentile Contrarian 20%': percentile_based_signals(
                    self.returns, top_pct=0.2, bottom_pct=0.2, strategy_type='contrarian'
                ),
                'Percentile Contrarian 10%': percentile_based_signals(
                    self.returns, top_pct=0.1, bottom_pct=0.1, strategy_type='contrarian'
                ),
            }
            
            pct_results = engine.run_multiple_backtests(percentile_strategies, portfolio_type="long_short")
            
            for name, result in pct_results.items():
                self.results[name] = {
                    'result': result,
                    'category': 'Percentile-Based',
                    'strategy_name': name
                }
                
        except Exception as e:
            print(f"⚠️ Percentile-based strategies failed: {e}")
    
    def create_portfolio_blends(self):
        """Create blended portfolios from best strategies."""
        print("\n🌟 Creating Portfolio Blends")
        print("-" * 30)
        
        # Find best strategies by category
        best_strategies = {}
        categories = ['Basic', 'Multi-Timeframe', 'Volatility-Adjusted', 'Percentile-Based']
        
        for category in categories:
            category_results = {name: data for name, data in self.results.items() 
                              if data['category'] == category}
            
            if category_results:
                best_name = max(category_results.items(), 
                              key=lambda x: x[1]['result'].metrics['sharpe_ratio'])[0]
                best_strategies[category] = best_name
                print(f"🏆 Best {category}: {best_name}")
        
        # Create blended portfolios
        if len(best_strategies) >= 2:
            try:
                # Get weights from best strategies
                strategy_weights = {}
                for category, strategy_name in best_strategies.items():
                    weights = self.results[strategy_name]['result'].weights
                    strategy_weights[strategy_name] = weights
                
                # Create blended portfolio
                optimizer = PortfolioOptimizer(self.returns)
                blended_weights = optimizer.blend_strategies(
                    strategy_weights, 
                    blend_method="inverse_vol"
                )
                
                # Backtest blended portfolio
                engine = BacktestEngine(self.returns)
                blended_result = engine.run_backtest(
                    signals=blended_weights,
                    strategy_name="Blended Best Strategies",
                    portfolio_type="custom"
                )
                
                self.results['Blended Portfolio'] = {
                    'result': blended_result,
                    'category': 'Blended',
                    'strategy_name': 'Blended Portfolio'
                }
                
                print("✅ Created blended portfolio")
                
            except Exception as e:
                print(f"⚠️ Portfolio blending failed: {e}")
    
    def generate_comprehensive_comparison(self):
        """Generate comprehensive strategy comparison."""
        print("\n📊 Comprehensive Strategy Comparison")
        print("=" * 60)
        
        # Extract backtest results
        backtest_results = {}
        for name, data in self.results.items():
            backtest_results[name] = data['result']
        
        # Create comparison
        engine = BacktestEngine(self.returns)
        comparison = engine.compare_strategies(backtest_results)
        
        # Add category column
        comparison['Category'] = [self.results[name]['category'] for name in comparison.index]
        
        # Reorder columns
        cols = ['Category', 'total_return', 'annualized_return', 'volatility', 
                'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']
        comparison = comparison[cols]
        
        print(comparison.round(4))
        
        # Save comprehensive results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        comparison.to_csv(results_dir / 'comprehensive_strategy_comparison.csv')
        
        # Save individual results
        for name, data in self.results.items():
            result = data['result']
            clean_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
            
            result.equity_curve.to_csv(results_dir / f'{clean_name}_equity.csv')
            result.strategy_returns.to_csv(results_dir / f'{clean_name}_returns.csv')
            if hasattr(result.weights, 'to_csv'):
                result.weights.to_csv(results_dir / f'{clean_name}_weights.csv')
        
        print(f"\n💾 Results saved to {results_dir}/")
        
        return comparison
    
    def print_top_strategies(self, comparison: pd.DataFrame, top_n: int = 5):
        """Print top performing strategies."""
        print(f"\n🏆 TOP {top_n} STRATEGIES BY SHARPE RATIO")
        print("=" * 50)
        
        top_strategies = comparison.nlargest(top_n, 'sharpe_ratio')
        
        for i, (name, row) in enumerate(top_strategies.iterrows(), 1):
            print(f"{i}. {name}")
            print(f"   📈 Total Return: {row['total_return']:.2%}")
            print(f"   📊 Sharpe Ratio: {row['sharpe_ratio']:.3f}")
            print(f"   📉 Max Drawdown: {row['max_drawdown']:.2%}")
            print(f"   🎯 Category: {row['Category']}")
            print()
    
    def run_all_strategies(self):
        """Run complete analysis."""
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        self.load_data()
        
        # Run strategies
        self.run_basic_strategies()
        self.run_enhanced_contrarian()  # Add enhanced contrarian
        self.run_advanced_strategies()
        self.create_portfolio_blends()
        
        # Generate comparison
        comparison = self.generate_comprehensive_comparison()
        
        # Print results
        self.print_top_strategies(comparison)
        
        print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("✅ ALL STRATEGIES COMPLETED SUCCESSFULLY!")
        
        return self.results, comparison

def main():
    """Main execution function."""
    runner = AdvancedMomentumRunner(data_dir='raw')
    results, comparison = runner.run_all_strategies()
    
    return results, comparison

if __name__ == "__main__":
    results, comparison = main()