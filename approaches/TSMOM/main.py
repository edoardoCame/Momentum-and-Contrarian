#!/usr/bin/env python3
"""
TSMOM Strategy Main Execution Script

Implements Time Series Momentum strategy following Moskowitz-Ooi-Pedersen (2012).
Orchestrates data loading, signal generation, portfolio construction, and analysis.
"""

import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from data_loader import TSMOMDataLoader
from volatility_estimator import TSMOMVolatilityEstimator
from tsmom_strategy import TSMOMStrategy
from performance_analyzer import TSMOMPerformanceAnalyzer

def main():
    """
    Main execution function for TSMOM strategy
    """
    print("="*80)
    print("TIME SERIES MOMENTUM (TSMOM) STRATEGY")
    print("Based on Moskowitz-Ooi-Pedersen (2012)")
    print("="*80)
    
    # Configuration
    START_DATE = "2000-01-01"
    K_LOOKBACK = 12  # months
    H_HOLDING = 1    # months  
    TARGET_VOL = 0.40  # 40% annualized
    COM_DAYS = 60    # EWMA center of mass
    
    print(f"\nStrategy Parameters:")
    print(f"Start Date: {START_DATE}")
    print(f"Lookback Period (k): {K_LOOKBACK} months")
    print(f"Holding Period (h): {H_HOLDING} months")
    print(f"Target Volatility: {TARGET_VOL:.0%}")
    print(f"EWMA Center of Mass: {COM_DAYS} days")
    
    try:
        # Step 1: Data Loading
        print(f"\n{'='*60}")
        print("STEP 1: DATA LOADING")
        print(f"{'='*60}")
        
        data_loader = TSMOMDataLoader(start_date=START_DATE, cache_dir="data/cached")
        daily_prices, monthly_prices, daily_returns = data_loader.load_all_data(force_refresh=False)
        
        # Calculate monthly returns
        monthly_returns = data_loader.calculate_monthly_returns(monthly_prices, excess_returns=False)
        
        print(f"\nData Summary:")
        print(f"Assets loaded: {len(monthly_prices.columns)}")
        print(f"Date range: {monthly_prices.index[0].date()} to {monthly_prices.index[-1].date()}")
        print(f"Monthly observations: {len(monthly_returns)}")
        
        # Step 2: Volatility Estimation
        print(f"\n{'='*60}")
        print("STEP 2: VOLATILITY ESTIMATION")
        print(f"{'='*60}")
        
        vol_estimator = TSMOMVolatilityEstimator(center_of_mass=COM_DAYS)
        monthly_vol = vol_estimator.calculate_monthly_volatility(daily_returns, lag_periods=1)
        
        # Validate volatility estimates
        vol_validation = vol_estimator.validate_volatility_estimates(daily_returns, monthly_vol)
        
        # Step 3: Strategy Implementation
        print(f"\n{'='*60}")
        print("STEP 3: STRATEGY IMPLEMENTATION")
        print(f"{'='*60}")
        
        strategy = TSMOMStrategy(k=K_LOOKBACK, h=H_HOLDING, target_vol=TARGET_VOL)
        final_weights, portfolio_returns = strategy.run_strategy(monthly_returns, monthly_vol)
        
        # Step 4: Performance Analysis
        print(f"\n{'='*60}")
        print("STEP 4: PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        analyzer = TSMOMPerformanceAnalyzer(results_dir="results")
        summary_df = analyzer.run_full_analysis(portfolio_returns, final_weights)
        
        # Step 5: Parameter Grid Analysis (Optional)
        print(f"\n{'='*60}")
        print("STEP 5: ROBUSTNESS TESTING")
        print(f"{'='*60}")
        
        try:
            grid_results = strategy.parameter_grid_analysis(
                monthly_returns, monthly_vol,
                k_values=[3, 6, 9, 12],
                h_values=[1, 3, 6]
            )
            
            # Save grid results
            grid_path = Path("results") / "parameter_grid_results.csv"
            grid_results.to_csv(grid_path, index=False)
            print(f"\nParameter grid results saved: {grid_path}")
            
        except Exception as e:
            print(f"Parameter grid analysis failed: {e}")
        
        # Final Summary
        print(f"\n{'='*80}")
        print("EXECUTION COMPLETE")
        print(f"{'='*80}")
        
        print(f"\nFiles generated:")
        results_dir = Path("results")
        if results_dir.exists():
            for file in results_dir.glob("*"):
                print(f"  {file.name}")
        
        print(f"\nStrategy successfully implemented!")
        print(f"Results saved in: {results_dir.absolute()}")
        
        # Print key results
        if 'total' in portfolio_returns:
            total_returns = portfolio_returns['total'].dropna()
            if len(total_returns) > 0:
                total_ret = (1 + total_returns).prod() - 1
                ann_ret = total_returns.mean() * 12
                ann_vol = total_returns.std() * np.sqrt(12)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
                
                print(f"\nKey Results (Total Portfolio):")
                print(f"Total Return: {total_ret:.1%}")
                print(f"Annualized Return: {ann_ret:.1%}")
                print(f"Annualized Volatility: {ann_vol:.1%}")
                print(f"Sharpe Ratio: {sharpe:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_test():
    """
    Run a quick test with minimal data for debugging
    """
    print("="*80)
    print("TSMOM QUICK TEST")
    print("="*80)
    
    try:
        # Test data loading with cache
        data_loader = TSMOMDataLoader(start_date="2020-01-01", cache_dir="data/cached")  # Shorter period
        
        # Test with just a few symbols
        test_symbols = ["EURUSD=X", "CL=F", "GC=F"]
        daily_prices = data_loader.download_data(test_symbols)
        
        if daily_prices.empty:
            print("No data loaded - check internet connection")
            return False
        
        print(f"✓ Data loading test passed: {daily_prices.shape}")
        
        # Test volatility estimation
        daily_returns = daily_prices.pct_change().dropna()
        vol_estimator = TSMOMVolatilityEstimator(center_of_mass=30)  # Shorter for test
        
        if len(daily_returns) > 30:
            vol_estimates = vol_estimator.calculate_ewma_variance(daily_returns, min_periods=30)
            print(f"✓ Volatility estimation test passed: {vol_estimates.shape}")
        else:
            print("Insufficient data for volatility test")
            return False
        
        print("✓ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TSMOM Strategy Implementation")
    print("Choose execution mode:")
    print("1. Full execution (default)")
    print("2. Quick test")
    print("3. Clear cache and refresh data")
    
    # Get user input (default to full execution)
    try:
        choice = input("\nEnter choice (1, 2, or 3, default=1): ").strip()
        
        if choice == "2":
            success = run_quick_test()
        elif choice == "3":
            # Clear cache and force refresh
            print("Clearing cache and forcing data refresh...")
            from data_loader import TSMOMDataLoader
            data_loader = TSMOMDataLoader(cache_dir="data/cached")
            data_loader.clear_cache()
            success = main()
        else:
            success = main()
        
        if success:
            print("\n✓ Execution completed successfully!")
        else:
            print("\n✗ Execution failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)