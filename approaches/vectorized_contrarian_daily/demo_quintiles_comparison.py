#!/usr/bin/env python3
"""
Demo script for commodity quintiles vs universe strategies comparison
This script demonstrates the full workflow with a small subset of commodities
"""

import sys
import os
sys.path.append('modules')

from modules.commodity_quintile_strategy import run_commodity_strategies_comparison, save_commodity_comparison_results
from modules.commodity_universe_comparison import (
    calculate_performance_metrics,
    analyze_position_characteristics,
    print_detailed_comparison_table,
    generate_summary_report
)
from modules.commodities_backtest import download_and_save_commodities_data

def main():
    print("=" * 80)
    print("COMMODITY CONTRARIAN STRATEGIES: QUINTILES VS UNIVERSE DEMO")
    print("=" * 80)
    
    # Use a representative subset of commodities for demo
    demo_tickers = [
        'CL=F',  # Energy: Crude Oil
        'GC=F',  # Precious Metal: Gold
        'HG=F',  # Industrial Metal: Copper
        'ZC=F',  # Agriculture: Corn
        'ZS=F',  # Agriculture: Soybeans
        'SB=F'   # Soft Commodity: Sugar
    ]
    
    print(f"Demo using {len(demo_tickers)} representative commodities:")
    for ticker in demo_tickers:
        print(f"  • {ticker}")
    
    # Configuration
    LOOKBACK_DAYS = 20  # 1 month lookback
    APPLY_TRANSACTION_COSTS = True
    VOLUME_TIER = 1  # Retail tier
    
    print(f"\nConfiguration:")
    print(f"  • Lookback period: {LOOKBACK_DAYS} days")
    print(f"  • Transaction costs: {'Enabled' if APPLY_TRANSACTION_COSTS else 'Disabled'}")
    print(f"  • IBKR volume tier: {VOLUME_TIER}")
    
    try:
        # Step 1: Load data
        print(f"\n{'-'*50}")
        print("STEP 1: LOADING DATA")
        print(f"{'-'*50}")
        
        data_dict = download_and_save_commodities_data(
            demo_tickers,
            start_date='2018-01-01',  # Use 6+ years for robust analysis
            end_date='2024-12-31',
            data_dir='commodities/data/raw'
        )
        
        print(f"✓ Loaded data for {len(data_dict)} commodities")
        
        # Step 2: Run strategies comparison
        print(f"\n{'-'*50}")
        print("STEP 2: RUNNING STRATEGIES")
        print(f"{'-'*50}")
        
        results_dict, positions_dict = run_commodity_strategies_comparison(
            data_dict=data_dict,
            lookback_days=LOOKBACK_DAYS,
            apply_transaction_costs=APPLY_TRANSACTION_COSTS,
            volume_tier=VOLUME_TIER
        )
        
        # Step 3: Calculate metrics
        print(f"\n{'-'*50}")
        print("STEP 3: CALCULATING METRICS")
        print(f"{'-'*50}")
        
        metrics_dict = {}
        for strategy_name, results in results_dict.items():
            metrics = calculate_performance_metrics(results, strategy_name)
            metrics_dict[strategy_name] = metrics
            
            total_return = results['cumulative_returns'].iloc[-1] - 1
            days = len(results['strategy_returns'].dropna())
            
            print(f"\n{strategy_name.upper()}:")
            print(f"  • Total Return: {total_return:.2%}")
            print(f"  • Annual Return: {metrics['annual_return']:.2%}")
            print(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  • Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  • Win Rate: {metrics['win_rate']:.1%}")
            print(f"  • Trading Days: {days}")
        
        # Step 4: Position analysis
        print(f"\n{'-'*50}")
        print("STEP 4: POSITION ANALYSIS")
        print(f"{'-'*50}")
        
        positions_analysis = analyze_position_characteristics(positions_dict)
        
        for strategy_name, analysis in positions_analysis.items():
            print(f"\n{strategy_name.upper()}:")
            print(f"  • Avg Active Positions: {analysis['avg_active_positions']:.1f}")
            print(f"  • Max Active Positions: {analysis['max_active_positions']:.0f}")
            print(f"  • Portfolio Concentration: {analysis['avg_concentration']:.3f}")
            print(f"  • Daily Turnover: {analysis['avg_daily_turnover']:.3f}")
        
        # Step 5: Save results
        print(f"\n{'-'*50}")
        print("STEP 5: SAVING RESULTS")
        print(f"{'-'*50}")
        
        save_commodity_comparison_results(results_dict, positions_dict)
        
        # Step 6: Summary
        print(f"\n{'-'*50}")
        print("STEP 6: EXECUTIVE SUMMARY")
        print(f"{'-'*50}")
        
        summary_report = generate_summary_report(
            results_dict=results_dict,
            metrics_dict=metrics_dict,
            positions_analysis=positions_analysis,
            lookback_days=LOOKBACK_DAYS
        )
        
        print(summary_report)
        
        # Step 7: Detailed comparison
        print(f"\n{'-'*50}")
        print("STEP 7: DETAILED COMPARISON")
        print(f"{'-'*50}")
        
        print_detailed_comparison_table(metrics_dict, positions_analysis)
        
        print(f"\n{'='*80}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("='*80")
        print("Next steps:")
        print("• Run the full notebook for comprehensive analysis")
        print("• Experiment with different lookback periods") 
        print("• Test with transaction costs disabled for gross performance")
        print("• Analyze results across different market regimes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)