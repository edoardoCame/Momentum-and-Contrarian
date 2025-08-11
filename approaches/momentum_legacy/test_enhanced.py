#!/usr/bin/env python3
"""
Test script for enhanced multi-timeframe contrarian strategy.
"""
import sys
sys.path.append('src')

from signals import multi_timeframe_contrarian_enhanced
from data_loader import CommodityDataLoader
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    print('📊 Testing enhanced multi-timeframe contrarian strategy...')

    # Load data
    loader = CommodityDataLoader('raw')
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns(prices)

    print(f'Data loaded: {returns.shape[0]} days, {returns.shape[1]} commodities')

    # Test the enhanced strategy
    try:
        signals = multi_timeframe_contrarian_enhanced(returns, prices)
        print(f'✅ Enhanced signals generated: {signals.shape}')
        print(f'Signal range: {signals.min().min()} to {signals.max().max()}')
        non_zero = (signals != 0).sum().sum()
        print(f'Non-zero signals: {non_zero}')
        
        # Show some sample signals
        print('\n📈 Sample signals (last 5 days):')
        print(signals.tail())
        
        # Signal distribution
        print('\n📊 Signal distribution:')
        print(f'Long signals (1): {(signals == 1).sum().sum()}')
        print(f'Short signals (-1): {(signals == -1).sum().sum()}') 
        print(f'Neutral signals (0): {(signals == 0).sum().sum()}')
        
        print('\n✅ Enhanced contrarian strategy test completed successfully!')
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()