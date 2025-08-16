#!/usr/bin/env python3
"""
Lookback Period Comparison Script for TSMOM Strategy

Tests different lookback periods and creates overlapping equity curve plots
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from data_loader import TSMOMDataLoader
from volatility_estimator import TSMOMVolatilityEstimator
from tsmom_strategy import TSMOMStrategy

def test_lookback_periods():
    """
    Test different lookback periods and generate comparison
    """
    print("="*80)
    print("TSMOM LOOKBACK PERIOD COMPARISON")
    print("="*80)
    
    # Configuration
    START_DATE = "2000-01-01"
    TARGET_VOL = 0.40
    COM_DAYS = 60
    LOOKBACK_PERIODS = [3, 6, 9, 12, 18, 24]  # months
    H_HOLDING = 1  # months
    
    print(f"\nTesting lookback periods: {LOOKBACK_PERIODS}")
    print(f"Target Volatility: {TARGET_VOL:.0%}")
    print(f"Holding Period: {H_HOLDING} month(s)")
    
    # Step 1: Load data once
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    data_loader = TSMOMDataLoader(start_date=START_DATE, cache_dir="data/cached")
    daily_prices, monthly_prices, daily_returns = data_loader.load_all_data(force_refresh=False)
    monthly_returns = data_loader.calculate_monthly_returns(monthly_prices, excess_returns=False)
    
    # Step 2: Calculate volatility once
    print(f"\n{'='*60}")
    print("CALCULATING VOLATILITY")
    print(f"{'='*60}")
    
    vol_estimator = TSMOMVolatilityEstimator(center_of_mass=COM_DAYS)
    monthly_vol = vol_estimator.calculate_monthly_volatility(daily_returns, lag_periods=1)
    
    # Step 3: Test different lookback periods
    print(f"\n{'='*60}")
    print("TESTING LOOKBACK PERIODS")
    print(f"{'='*60}")
    
    results = {}
    equity_curves = {}
    performance_metrics = []
    
    for k in LOOKBACK_PERIODS:
        print(f"\n--- Testing {k}-month lookback ---")
        
        try:
            # Initialize strategy with current lookback
            strategy = TSMOMStrategy(k=k, h=H_HOLDING, target_vol=TARGET_VOL)
            
            # Run strategy
            final_weights, portfolio_returns = strategy.run_strategy(monthly_returns, monthly_vol)
            
            # Store results
            results[k] = {
                'weights': final_weights,
                'returns': portfolio_returns
            }
            
            # Calculate equity curve
            total_returns = portfolio_returns['total'].dropna()
            if len(total_returns) > 0:
                equity_curve = (1 + total_returns).cumprod()
                equity_curves[k] = equity_curve
                
                # Calculate performance metrics
                total_ret = (1 + total_returns).prod() - 1
                ann_ret = total_returns.mean() * 12
                ann_vol = total_returns.std() * np.sqrt(12)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
                max_dd = calculate_max_drawdown(equity_curve)
                
                performance_metrics.append({
                    'lookback_months': k,
                    'total_return': total_ret,
                    'annual_return': ann_ret,
                    'annual_volatility': ann_vol,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'n_observations': len(total_returns)
                })
                
                print(f"✓ {k}m: {ann_ret:.1%} return, {ann_vol:.1%} vol, {sharpe:.2f} Sharpe, {max_dd:.1%} MDD")
            else:
                print(f"✗ {k}m: No returns data")
                
        except Exception as e:
            print(f"✗ {k}m: Error - {e}")
    
    # Step 4: Create comparison plots
    print(f"\n{'='*60}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*60}")
    
    create_equity_curve_comparison(equity_curves)
    create_performance_summary(performance_metrics)
    
    print(f"\nComparison analysis complete!")
    return results, equity_curves, performance_metrics

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    if len(equity_curve) == 0:
        return np.nan
    
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def create_equity_curve_comparison(equity_curves):
    """Create overlapping equity curve plot"""
    if not equity_curves:
        print("No equity curves to plot")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Define colors for different lookback periods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (k, equity_curve) in enumerate(equity_curves.items()):
        color = colors[i % len(colors)]
        plt.plot(equity_curve.index, equity_curve.values, 
                label=f'{k}-month lookback', 
                linewidth=2, 
                color=color,
                alpha=0.8)
    
    plt.title('TSMOM Strategy: Equity Curves by Lookback Period', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Base = 1)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # Format y-axis as percentages
    from matplotlib.ticker import FuncFormatter
    def format_percent(x, pos):
        return f'{x:.1f}x'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_percent))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plot_path = results_dir / "lookback_comparison_equity_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Equity curve comparison saved: {plot_path}")
    
    plt.show()

def create_performance_summary(performance_metrics):
    """Create performance summary table and plots"""
    if not performance_metrics:
        print("No performance metrics to summarize")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_metrics)
    
    # Save to CSV
    results_dir = Path("results")
    csv_path = results_dir / "lookback_comparison_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Performance metrics saved: {csv_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY BY LOOKBACK PERIOD")
    print(f"{'='*80}")
    print(df.round(4).to_string(index=False))
    
    # Create performance comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sharpe Ratio
    axes[0,0].bar(df['lookback_months'], df['sharpe_ratio'], color='steelblue', alpha=0.7)
    axes[0,0].set_title('Sharpe Ratio by Lookback Period', fontweight='bold')
    axes[0,0].set_xlabel('Lookback Period (months)')
    axes[0,0].set_ylabel('Sharpe Ratio')
    axes[0,0].grid(True, alpha=0.3)
    
    # Annual Return
    axes[0,1].bar(df['lookback_months'], df['annual_return']*100, color='green', alpha=0.7)
    axes[0,1].set_title('Annual Return by Lookback Period', fontweight='bold')
    axes[0,1].set_xlabel('Lookback Period (months)')
    axes[0,1].set_ylabel('Annual Return (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Volatility
    axes[1,0].bar(df['lookback_months'], df['annual_volatility']*100, color='orange', alpha=0.7)
    axes[1,0].set_title('Annual Volatility by Lookback Period', fontweight='bold')
    axes[1,0].set_xlabel('Lookback Period (months)')
    axes[1,0].set_ylabel('Annual Volatility (%)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Max Drawdown
    axes[1,1].bar(df['lookback_months'], df['max_drawdown']*100, color='red', alpha=0.7)
    axes[1,1].set_title('Maximum Drawdown by Lookback Period', fontweight='bold')
    axes[1,1].set_xlabel('Lookback Period (months)')
    axes[1,1].set_ylabel('Max Drawdown (%)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save performance comparison plot
    perf_plot_path = results_dir / "lookback_comparison_metrics.png"
    plt.savefig(perf_plot_path, dpi=300, bbox_inches='tight')
    print(f"Performance comparison plot saved: {perf_plot_path}")
    
    plt.show()

if __name__ == "__main__":
    try:
        results, equity_curves, performance_metrics = test_lookback_periods()
        print(f"\n✓ Lookback comparison analysis completed successfully!")
        
        # Print best performing lookback
        if performance_metrics:
            df = pd.DataFrame(performance_metrics)
            best_sharpe_idx = df['sharpe_ratio'].idxmax()
            best_lookback = df.loc[best_sharpe_idx, 'lookback_months']
            best_sharpe = df.loc[best_sharpe_idx, 'sharpe_ratio']
            print(f"\nBest performing lookback: {best_lookback} months (Sharpe: {best_sharpe:.3f})")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)