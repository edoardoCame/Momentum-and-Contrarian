#!/usr/bin/env python3
"""
Verification script to ensure all modules work correctly
Run this script to verify the commodity quintiles vs universe implementation
"""

import sys
import os

def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...")
    
    try:
        # Add modules to path
        sys.path.append('modules')
        
        # Test core modules
        from modules.commodity_quintile_strategy import (
            run_commodity_strategies_comparison,
            prepare_daily_commodity_data,
            contrarian_quintiles_daily_strategy,
            contrarian_full_universe_daily_strategy
        )
        print("  ✓ commodity_quintile_strategy imported successfully")
        
        from modules.commodity_universe_comparison import (
            calculate_performance_metrics,
            analyze_position_characteristics,
            print_detailed_comparison_table
        )
        print("  ✓ commodity_universe_comparison imported successfully")
        
        from modules.commodities_backtest import download_and_save_commodities_data
        print("  ✓ commodities_backtest imported successfully")
        
        from modules.strategy_contrarian import strategy
        print("  ✓ strategy_contrarian imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_notebook_compatibility():
    """Test notebook import paths work"""
    print("\nTesting notebook compatibility...")
    
    try:
        # Simulate notebook environment
        current_dir = os.getcwd()
        notebook_dir = os.path.join(current_dir, 'commodities', 'notebooks')
        os.chdir(notebook_dir)
        
        # Add paths like notebook does
        sys.path.insert(0, '../../modules')
        sys.path.insert(0, '../..')
        
        # Test imports
        from commodity_quintile_strategy import run_commodity_strategies_comparison
        from commodity_universe_comparison import calculate_performance_metrics
        from commodities_backtest import download_and_save_commodities_data
        
        print("  ✓ Notebook imports work correctly")
        
        # Restore directory
        os.chdir(current_dir)
        return True
        
    except Exception as e:
        print(f"  ✗ Notebook compatibility failed: {e}")
        # Restore directory even if failed
        try:
            os.chdir(current_dir)
        except:
            pass
        return False

def test_basic_functionality():
    """Test basic functionality with minimal data"""
    print("\nTesting basic functionality...")
    
    try:
        sys.path.append('modules')
        from modules.commodity_quintile_strategy import prepare_daily_commodity_data
        
        # Create minimal test data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        test_data = {
            'TEST1': pd.DataFrame({
                'Close': np.random.uniform(90, 110, len(dates))
            }, index=dates),
            'TEST2': pd.DataFrame({
                'Close': np.random.uniform(45, 55, len(dates))
            }, index=dates)
        }
        
        # Test data preparation
        daily_prices = prepare_daily_commodity_data(test_data)
        
        if daily_prices.shape == (len(dates), 2):
            print("  ✓ Data preparation works correctly")
            return True
        else:
            print(f"  ✗ Unexpected data shape: {daily_prices.shape}")
            return False
            
    except Exception as e:
        print(f"  ✗ Basic functionality failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("COMMODITY QUINTILES vs UNIVERSE - INSTALLATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Notebook Compatibility", test_notebook_compatibility), 
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running {test_name}...")
        print(f"{'-' * 40}")
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'=' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour installation is working correctly.")
        print("\nNext steps:")
        print("1. Open the Jupyter notebook: commodities/notebooks/commodities_quintiles_vs_universe_comparison.ipynb")
        print("2. Run the demo script: python3 demo_quintiles_comparison.py")
        print("3. Execute the test suite: python3 test_quintiles_strategies.py")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed.")
        print("\nPlease check the error messages above and ensure all dependencies are installed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)