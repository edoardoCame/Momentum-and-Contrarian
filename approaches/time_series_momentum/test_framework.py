#!/usr/bin/env python3
"""
Test script for TSMOM framework to verify all modules work together.
"""

import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append('modules')

from modules.data_loader import ForexDataLoader
from modules.tsmom_strategy import TSMOMStrategy
from modules.backtest_engine import TSMOMBacktestEngine
from modules.performance_utils import TSMOMPerformanceAnalyzer

def test_tsmom_framework():
    """Test the complete TSMOM framework."""
    
    print("=" * 60)
    print("TESTING TSMOM FRAMEWORK")
    print("=" * 60)
    
    # Test 1: Data Loading
    print("\n1. Testing Data Loading...")
    try:
        loader = ForexDataLoader()
        weekly_returns, monthly_returns = loader.prepare_data_for_backtest()
        print(f"✓ Data loaded successfully!")
        print(f"  Weekly data: {weekly_returns.shape}")
        print(f"  Monthly data: {monthly_returns.shape}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Test 2: Strategy Signal Generation
    print("\n2. Testing Strategy Signals...")
    try:
        tsmom = TSMOMStrategy()
        
        # Test one weekly strategy
        weekly_signals = tsmom.calculate_tsmom_signals(
            weekly_returns, [4], 'weekly'
        )
        
        # Test one monthly strategy  
        monthly_signals = tsmom.calculate_tsmom_signals(
            monthly_returns, [1], 'monthly'
        )
        
        print(f"✓ Signal generation successful!")
        print(f"  Weekly signals: {list(weekly_signals.keys())}")
        print(f"  Monthly signals: {list(monthly_signals.keys())}")
        
    except Exception as e:
        print(f"✗ Signal generation failed: {e}")
        return False
    
    # Test 3: Backtesting
    print("\n3. Testing Backtesting Engine...")
    try:
        engine = TSMOMBacktestEngine(transaction_cost_bps=5.0)
        
        # Test weekly backtest
        weekly_results = engine.run_all_backtests(weekly_signals, weekly_returns)
        
        # Test monthly backtest
        monthly_results = engine.run_all_backtests(monthly_signals, monthly_returns)
        
        # Combine results
        all_results = {**weekly_results, **monthly_results}
        
        print(f"✓ Backtesting successful!")
        print(f"  Total strategies tested: {len(all_results)}")
        
    except Exception as e:
        print(f"✗ Backtesting failed: {e}")
        return False
    
    # Test 4: Performance Analysis
    print("\n4. Testing Performance Analysis...")
    try:
        # Calculate metrics
        metrics = engine.calculate_performance_metrics(all_results)
        
        # Test analyzer
        analyzer = TSMOMPerformanceAnalyzer()
        formatted_table = analyzer.create_performance_table(metrics)
        summary_report = analyzer.generate_summary_report(all_results, metrics)
        
        print(f"✓ Performance analysis successful!")
        print(f"  Performance metrics calculated for {len(metrics)} strategies")
        
        # Display summary
        print(f"\n5. Summary Results:")
        print(f"Best strategy: {metrics['Net_Sharpe_Ratio'].idxmax()}")
        print(f"Best Sharpe ratio: {metrics['Net_Sharpe_Ratio'].max():.3f}")
        
        print(f"\nFormatted Performance Table:")
        print(formatted_table)
        
    except Exception as e:
        print(f"✗ Performance analysis failed: {e}")
        return False
    
    # Test 5: Results Saving
    print("\n6. Testing Results Saving...")
    try:
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save weekly and monthly results separately to avoid index mismatch
        if weekly_results:
            weekly_engine = TSMOMBacktestEngine(transaction_cost_bps=5.0)
            weekly_engine.save_results(weekly_results, "results")
            print(f"  ✓ Weekly results saved")
            
        if monthly_results:
            monthly_engine = TSMOMBacktestEngine(transaction_cost_bps=5.0) 
            monthly_engine.save_results(monthly_results, "results")
            print(f"  ✓ Monthly results saved")
        
        # Save combined performance metrics
        metrics.to_parquet("results/combined_performance_metrics.parquet")
        
        # Save additional files
        formatted_table.to_csv("results/test_performance_summary.csv")
        
        with open("results/test_summary_report.txt", "w") as f:
            f.write(summary_report)
        
        print(f"✓ Results saved successfully!")
        print(f"  Files saved to: {results_dir.absolute()}")
        
    except Exception as e:
        print(f"✗ Results saving failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("FRAMEWORK TEST COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)
    print(f"\nThe TSMOM framework is ready for use.")
    print(f"Run the Jupyter notebook for full analysis:")
    print(f"  jupyter notebook notebooks/tsmom_analysis.ipynb")
    
    return True

if __name__ == "__main__":
    success = test_tsmom_framework()
    if not success:
        print("\n" + "=" * 60)
        print("FRAMEWORK TEST FAILED! ✗")
        print("=" * 60)
        sys.exit(1)