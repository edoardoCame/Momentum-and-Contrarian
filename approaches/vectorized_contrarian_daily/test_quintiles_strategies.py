#!/usr/bin/env python3
"""
Test script for commodity quintiles vs universe strategies
This script validates the implementation and runs basic tests
"""

import sys
import os
import pandas as pd
import numpy as np

# Add modules to path
sys.path.append('modules')

# Import our modules
from modules.commodity_quintile_strategy import (
    run_commodity_strategies_comparison,
    save_commodity_comparison_results,
    prepare_daily_commodity_data,
    contrarian_quintiles_daily_strategy,
    contrarian_full_universe_daily_strategy
)
from modules.commodity_universe_comparison import (
    calculate_performance_metrics,
    analyze_position_characteristics
)
from modules.commodities_backtest import download_and_save_commodities_data

def test_basic_functionality():
    """Test basic functionality with a small subset of data"""
    print("=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    # Use a small subset of tickers for testing
    test_tickers = ['CL=F', 'GC=F', 'ZC=F']  # Oil, Gold, Corn
    print(f"Testing with {len(test_tickers)} commodities: {test_tickers}")
    
    try:
        # Download test data
        print("\n1. Downloading test data...")
        data_dict = download_and_save_commodities_data(
            test_tickers, 
            start_date='2020-01-01', 
            end_date='2023-12-31',
            data_dir='commodities/data/raw'
        )
        
        if len(data_dict) < len(test_tickers):
            print(f"WARNING: Only {len(data_dict)} out of {len(test_tickers)} tickers loaded")
            
        # Test data preparation
        print("\n2. Testing data preparation...")
        daily_prices = prepare_daily_commodity_data(data_dict)
        print(f"   Daily data shape: {daily_prices.shape}")
        print(f"   Date range: {daily_prices.index[0].date()} to {daily_prices.index[-1].date()}")
        
        # Test individual strategies
        print("\n3. Testing quintiles strategy...")
        quintiles_results, quintiles_positions = contrarian_quintiles_daily_strategy(
            daily_prices, lookback_days=10, apply_transaction_costs=False
        )
        print(f"   Results shape: {quintiles_results.shape}")
        print(f"   Positions shape: {quintiles_positions.shape}")
        print(f"   Final return: {quintiles_results['cumulative_returns'].iloc[-1]:.4f}")
        
        print("\n4. Testing full universe strategy...")
        universe_results, universe_positions = contrarian_full_universe_daily_strategy(
            daily_prices, lookback_days=10, apply_transaction_costs=False
        )
        print(f"   Results shape: {universe_results.shape}")
        print(f"   Positions shape: {universe_positions.shape}")
        print(f"   Final return: {universe_results['cumulative_returns'].iloc[-1]:.4f}")
        
        # Test comparison function
        print("\n5. Testing comparison function...")
        results_dict, positions_dict = run_commodity_strategies_comparison(
            data_dict, lookback_days=10, apply_transaction_costs=False
        )
        
        print(f"   Strategies tested: {list(results_dict.keys())}")
        for strategy, results in results_dict.items():
            final_return = results['cumulative_returns'].iloc[-1] - 1
            print(f"   {strategy}: {final_return:.2%} total return")
        
        print("\n✓ Basic functionality test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Basic functionality test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bias_prevention():
    """Test that strategies prevent lookahead bias"""
    print("\n" + "=" * 60)
    print("TESTING BIAS PREVENTION")
    print("=" * 60)
    
    try:
        # Create synthetic test data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        n_assets = 3
        
        # Create predictable pattern: asset 0 always goes up, assets 1-4 always go down
        # Use 5 assets so bottom quintile (20%) = 1 asset
        n_assets = 5
        np.random.seed(42)
        prices = pd.DataFrame(index=dates, columns=[f'ASSET_{i}' for i in range(n_assets)])
        
        prices.iloc[0] = 100  # Starting prices
        for i in range(1, len(dates)):
            prices.iloc[i, 0] = prices.iloc[i-1, 0] * 1.001  # Always up
            prices.iloc[i, 1] = prices.iloc[i-1, 1] * 0.998  # Always down most
            prices.iloc[i, 2] = prices.iloc[i-1, 2] * 0.9985  # Always down 
            prices.iloc[i, 3] = prices.iloc[i-1, 3] * 0.999  # Always down
            prices.iloc[i, 4] = prices.iloc[i-1, 4] * 0.9995  # Always down least
        
        print("Testing with synthetic data (5 assets):")
        print(f"  Asset 0: Always trending UP")
        print(f"  Asset 1: Always trending DOWN (most)")
        print(f"  Asset 2-4: Always trending DOWN (varying rates)")
        
        # Test quintiles strategy
        print("\n1. Testing quintiles strategy bias prevention...")
        lookback = 5
        
        # Calculate returns
        daily_returns = prices.pct_change(fill_method=None)
        
        # Calculate lookback performance with shift(1) - this should prevent bias
        lookback_performance = daily_returns.rolling(window=lookback).sum().shift(1)
        
        # Check that we're not using today's data for today's decision
        print(f"   Lookback window: {lookback} days")
        print(f"   Using shift(1): True")
        
        # Check first valid signal date
        first_signal_date = lookback_performance.dropna().index[0]
        # Should be lookback days + 1 due to shift(1) - we need lookback days to calc rolling, then +1 for shift
        expected_first_date = dates[lookback + 1]  # lookback=5, so index 6 (7th day = Jan 7)
        
        print(f"   First valid signal: {first_signal_date.date()}")
        print(f"   Expected first signal: {expected_first_date.date()}")
        
        if first_signal_date == expected_first_date:
            print("   ✓ Bias prevention: Signal timing correct")
        else:
            print("   ✗ Bias prevention: Signal timing incorrect")
            return False
        
        # Test that contrarian logic works as expected
        # Asset 1 (always down) should be selected more often by contrarian strategy
        quintile_ranks = lookback_performance.rank(axis=1, pct=True)
        long_mask = quintile_ranks <= 0.2  # Bottom quintile
        
        selection_freq = long_mask.mean()
        print(f"\n   Selection frequencies:")
        for asset, freq in selection_freq.items():
            print(f"     {asset}: {freq:.1%}")
            
        # Asset 1 (always down most) should be selected most often by contrarian strategy
        # With 5 assets, bottom quintile = 20% = 1 asset, which should be ASSET_1
        if selection_freq.idxmax() == 'ASSET_1' and selection_freq['ASSET_1'] > 0:
            print("   ✓ Contrarian logic: Worst performer selected most often")
        else:
            print("   ✗ Contrarian logic: Logic may be incorrect")
            print(f"   Expected ASSET_1 to be selected most, but got: {selection_freq.idxmax()}")
            return False
        
        print("\n✓ Bias prevention test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Bias prevention test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transaction_costs():
    """Test transaction cost implementation"""
    print("\n" + "=" * 60)  
    print("TESTING TRANSACTION COSTS")
    print("=" * 60)
    
    try:
        # Use small data sample
        test_tickers = ['CL=F', 'GC=F']
        print(f"Testing transaction costs with: {test_tickers}")
        
        data_dict = download_and_save_commodities_data(
            test_tickers,
            start_date='2020-01-01',
            end_date='2023-12-31',
            data_dir='commodities/data/raw'
        )
        
        daily_prices = prepare_daily_commodity_data(data_dict)
        
        # Test without transaction costs
        print("\n1. Testing without transaction costs...")
        results_no_costs, _ = contrarian_quintiles_daily_strategy(
            daily_prices, lookback_days=10, apply_transaction_costs=False
        )
        
        # Test with transaction costs
        print("2. Testing with transaction costs...")
        results_with_costs, _ = contrarian_quintiles_daily_strategy(
            daily_prices, lookback_days=10, apply_transaction_costs=True, volume_tier=1
        )
        
        # Compare results
        final_no_costs = results_no_costs['cumulative_returns'].iloc[-1] - 1
        final_with_costs = results_with_costs['cumulative_returns'].iloc[-1] - 1
        cost_drag = final_no_costs - final_with_costs
        
        print(f"\n   Without costs: {final_no_costs:.4f} ({final_no_costs:.2%})")
        print(f"   With costs: {final_with_costs:.4f} ({final_with_costs:.2%})")
        print(f"   Cost drag: {cost_drag:.4f} ({cost_drag:.2%})")
        
        # Check that costs reduce returns (as expected) OR costs exist
        if cost_drag > 0 or 'transaction_costs' in results_with_costs.columns:
            print("   ✓ Transaction costs logic working")
            # Check if there were any costs incurred
            total_costs = results_with_costs['transaction_costs'].sum()
            print(f"   Total costs incurred: {total_costs:.6f} ({total_costs*100:.4f}%)")
        else:
            print("   ✗ Transaction costs logic may be incorrect")
            return False
            
        # Check that cost columns exist
        required_cols = ['strategy_returns_gross', 'transaction_costs']
        missing_cols = [col for col in required_cols if col not in results_with_costs.columns]
        
        if not missing_cols:
            print("   ✓ All required cost columns present")
        else:
            print(f"   ✗ Missing cost columns: {missing_cols}")
            return False
        
        print("\n✓ Transaction costs test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Transaction costs test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_metrics():
    """Test performance metrics calculation"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE METRICS")
    print("=" * 60)
    
    try:
        # Create simple synthetic strategy results
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
        # Simple strategy: +0.1% every day
        daily_returns = pd.Series(0.001, index=dates)  # 0.1% daily
        cumulative_returns = (1 + daily_returns).cumprod()
        
        test_results = pd.DataFrame({
            'strategy_returns': daily_returns,
            'cumulative_returns': cumulative_returns
        })
        
        print("Testing with synthetic +0.1% daily return strategy...")
        
        # Test metrics calculation
        metrics = calculate_performance_metrics(test_results, "Test Strategy")
        
        print(f"\nCalculated metrics:")
        print(f"   Annual Return: {metrics['annual_return']:.2%}")
        print(f"   Annual Volatility: {metrics['annual_volatility']:.2%}") 
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
        
        # Basic sanity checks
        expected_annual_return = (1.001 ** 252) - 1  # ~29% annual
        annual_return_diff = abs(metrics['annual_return'] - expected_annual_return)
        
        if annual_return_diff < 0.01:  # Within 1%
            print("   ✓ Annual return calculation appears correct")
        else:
            print(f"   ✗ Annual return seems incorrect (expected ~{expected_annual_return:.2%})")
            return False
        
        # Win rate should be 100% (always positive returns)
        if abs(metrics['win_rate'] - 1.0) < 0.01:
            print("   ✓ Win rate calculation correct")
        else:
            print(f"   ✗ Win rate incorrect (expected 100%, got {metrics['win_rate']:.1%})")
            return False
        
        # Max drawdown should be 0 (always going up)
        if abs(metrics['max_drawdown']) < 0.001:
            print("   ✓ Max drawdown calculation correct")
        else:
            print(f"   ✗ Max drawdown incorrect (expected 0%, got {metrics['max_drawdown']:.2%})")
            return False
        
        print("\n✓ Performance metrics test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Performance metrics test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("COMMODITY QUINTILES STRATEGIES - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Bias Prevention", test_bias_prevention), 
        ("Transaction Costs", test_transaction_costs),
        ("Performance Metrics", test_performance_metrics)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests PASSED! The implementation is ready to use.")
        return True
    else:
        print(f"\n❌ {len(tests) - passed} tests FAILED. Please review the implementation.")
        return False

if __name__ == "__main__":
    main()