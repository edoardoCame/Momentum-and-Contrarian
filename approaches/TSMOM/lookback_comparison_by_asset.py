#!/usr/bin/env python3
"""
Lookback Period Comparison by Asset Class for TSMOM Strategy

Tests different lookback periods separately for Total, Commodities, and Forex
Creates overlapping equity curve plots for each asset class
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

def test_lookback_periods_by_asset():
    """
    Test different lookback periods and generate comparison by asset class
    """
    print("="*80)
    print("TSMOM LOOKBACK PERIOD COMPARISON BY ASSET CLASS")
    print("="*80)
    
    # Configuration
    START_DATE = "2000-01-01"
    TARGET_VOL = 0.40
    COM_DAYS = 60
    LOOKBACK_PERIODS = [3, 6, 9, 12, 18, 24]  # months
    H_HOLDING = 1  # months
    ASSET_CLASSES = ['total', 'commodities', 'forex']
    
    print(f"\nTesting lookback periods: {LOOKBACK_PERIODS}")
    print(f"Asset classes: {ASSET_CLASSES}")
    print(f"Target Volatility: {TARGET_VOL:.0%}")
    print(f"Holding Period: {H_HOLDING} month(s)")
    
    # Step 1: Load data once
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    data_loader = TSMOMDataLoader(start_date=START_DATE, cache_dir="data/cached")
    daily_prices, monthly_prices, daily_returns = data_loader.load_all_data(force_refresh=False)
    monthly_returns = data_loader.calculate_monthly_returns(monthly_prices, excess_returns=False)
    
    # Get universe info
    universe_info = data_loader.get_universe_info()
    print(f"Universe: {universe_info['total_symbols']} total assets")
    print(f"Commodities: {len(universe_info['commodities'])} assets")
    print(f"Forex: {len(universe_info['forex'])} assets")
    
    # Step 2: Calculate volatility once
    print(f"\n{'='*60}")
    print("CALCULATING VOLATILITY")
    print(f"{'='*60}")
    
    vol_estimator = TSMOMVolatilityEstimator(center_of_mass=COM_DAYS)
    monthly_vol = vol_estimator.calculate_monthly_volatility(daily_returns, lag_periods=1)
    
    # Step 3: Test different lookback periods
    print(f"\n{'='*60}")
    print("TESTING LOOKBACK PERIODS BY ASSET CLASS")
    print(f"{'='*60}")
    
    results_by_asset = {}
    equity_curves_by_asset = {}
    performance_metrics_by_asset = {}
    
    for asset_class in ASSET_CLASSES:
        print(f"\n{'='*40}")
        print(f"ASSET CLASS: {asset_class.upper()}")
        print(f"{'='*40}")
        
        results_by_asset[asset_class] = {}
        equity_curves_by_asset[asset_class] = {}
        performance_metrics_by_asset[asset_class] = []
        
        for k in LOOKBACK_PERIODS:
            print(f"\n--- {asset_class.capitalize()}: {k}-month lookback ---")
            
            try:
                # Initialize strategy with current lookback
                strategy = TSMOMStrategy(k=k, h=H_HOLDING, target_vol=TARGET_VOL)
                
                # Run strategy
                final_weights, portfolio_returns = strategy.run_strategy(monthly_returns, monthly_vol)
                
                # Get returns for specific asset class
                if asset_class in portfolio_returns:
                    asset_returns = portfolio_returns[asset_class].dropna()
                else:
                    print(f"Asset class {asset_class} not found in results")
                    continue
                
                # Store results
                results_by_asset[asset_class][k] = {
                    'weights': final_weights,
                    'returns': asset_returns
                }
                
                # Calculate equity curve
                if len(asset_returns) > 0:
                    equity_curve = (1 + asset_returns).cumprod()
                    equity_curves_by_asset[asset_class][k] = equity_curve
                    
                    # Calculate performance metrics
                    total_ret = (1 + asset_returns).prod() - 1
                    ann_ret = asset_returns.mean() * 12
                    ann_vol = asset_returns.std() * np.sqrt(12)
                    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
                    max_dd = calculate_max_drawdown(equity_curve)
                    
                    performance_metrics_by_asset[asset_class].append({
                        'asset_class': asset_class,
                        'lookback_months': k,
                        'total_return': total_ret,
                        'annual_return': ann_ret,
                        'annual_volatility': ann_vol,
                        'sharpe_ratio': sharpe,
                        'max_drawdown': max_dd,
                        'n_observations': len(asset_returns)
                    })
                    
                    print(f"✓ {asset_class} {k}m: {ann_ret:.1%} return, {ann_vol:.1%} vol, {sharpe:.2f} Sharpe, {max_dd:.1%} MDD")
                else:
                    print(f"✗ {asset_class} {k}m: No returns data")
                    
            except Exception as e:
                print(f"✗ {asset_class} {k}m: Error - {e}")
    
    # Step 4: Create comparison plots and analysis
    print(f"\n{'='*60}")
    print("CREATING COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    # Create separate plots for each asset class
    for asset_class in ASSET_CLASSES:
        if asset_class in equity_curves_by_asset and equity_curves_by_asset[asset_class]:
            create_equity_curve_comparison(equity_curves_by_asset[asset_class], asset_class)
            create_performance_summary(performance_metrics_by_asset[asset_class], asset_class)
    
    # Create combined comparison
    create_combined_asset_comparison(equity_curves_by_asset, performance_metrics_by_asset)
    
    print(f"\nAsset class comparison analysis complete!")
    return results_by_asset, equity_curves_by_asset, performance_metrics_by_asset

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    if len(equity_curve) == 0:
        return np.nan
    
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def create_equity_curve_comparison(equity_curves, asset_class):
    """Create overlapping equity curve plot for specific asset class"""
    if not equity_curves:
        print(f"No equity curves to plot for {asset_class}")
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
    
    plt.title(f'TSMOM Strategy: {asset_class.capitalize()} Equity Curves by Lookback Period', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Base = 1)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # Format y-axis
    from matplotlib.ticker import FuncFormatter
    def format_percent(x, pos):
        return f'{x:.1f}x'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_percent))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    results_dir = Path(f"results/{asset_class}")
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / f"lookback_comparison_equity_curves_{asset_class}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"{asset_class.capitalize()} equity curve comparison saved: {plot_path}")
    
    plt.show()

def create_performance_summary(performance_metrics, asset_class):
    """Create performance summary table and plots for specific asset class"""
    if not performance_metrics:
        print(f"No performance metrics to summarize for {asset_class}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_metrics)
    
    # Save to CSV
    results_dir = Path(f"results/{asset_class}")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"lookback_comparison_metrics_{asset_class}.csv"
    df.to_csv(csv_path, index=False)
    print(f"{asset_class.capitalize()} performance metrics saved: {csv_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY: {asset_class.upper()}")
    print(f"{'='*60}")
    print(df.round(4).to_string(index=False))
    
    # Create performance comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sharpe Ratio
    axes[0,0].bar(df['lookback_months'], df['sharpe_ratio'], color='steelblue', alpha=0.7)
    axes[0,0].set_title(f'{asset_class.capitalize()}: Sharpe Ratio by Lookback Period', fontweight='bold')
    axes[0,0].set_xlabel('Lookback Period (months)')
    axes[0,0].set_ylabel('Sharpe Ratio')
    axes[0,0].grid(True, alpha=0.3)
    
    # Annual Return
    axes[0,1].bar(df['lookback_months'], df['annual_return']*100, color='green', alpha=0.7)
    axes[0,1].set_title(f'{asset_class.capitalize()}: Annual Return by Lookback Period', fontweight='bold')
    axes[0,1].set_xlabel('Lookback Period (months)')
    axes[0,1].set_ylabel('Annual Return (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Volatility
    axes[1,0].bar(df['lookback_months'], df['annual_volatility']*100, color='orange', alpha=0.7)
    axes[1,0].set_title(f'{asset_class.capitalize()}: Annual Volatility by Lookback Period', fontweight='bold')
    axes[1,0].set_xlabel('Lookback Period (months)')
    axes[1,0].set_ylabel('Annual Volatility (%)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Max Drawdown
    axes[1,1].bar(df['lookback_months'], df['max_drawdown']*100, color='red', alpha=0.7)
    axes[1,1].set_title(f'{asset_class.capitalize()}: Maximum Drawdown by Lookback Period', fontweight='bold')
    axes[1,1].set_xlabel('Lookback Period (months)')
    axes[1,1].set_ylabel('Max Drawdown (%)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save performance comparison plot
    perf_plot_path = results_dir / f"lookback_comparison_metrics_{asset_class}.png"
    plt.savefig(perf_plot_path, dpi=300, bbox_inches='tight')
    print(f"{asset_class.capitalize()} performance comparison plot saved: {perf_plot_path}")
    
    plt.show()

def create_combined_asset_comparison(equity_curves_by_asset, performance_metrics_by_asset):
    """Create combined comparison across asset classes"""
    print(f"\n{'='*60}")
    print("CREATING COMBINED ASSET CLASS COMPARISON")
    print(f"{'='*60}")
    
    # 1. Create combined equity curves plot (best lookback for each asset class)
    plt.figure(figsize=(16, 10))
    
    # Find best lookbook for each asset class (highest Sharpe)
    best_configs = {}
    colors = {
        'total': '#d62728',  # Red for the new 50/50 balanced portfolio
        'commodities': '#ff7f0e', 
        'forex': '#2ca02c'
    }
    
    for asset_class in ['total', 'commodities', 'forex']:
        if asset_class in performance_metrics_by_asset and performance_metrics_by_asset[asset_class]:
            df = pd.DataFrame(performance_metrics_by_asset[asset_class])
            best_idx = df['sharpe_ratio'].idxmax()
            best_lookback = df.loc[best_idx, 'lookback_months']
            best_sharpe = df.loc[best_idx, 'sharpe_ratio']
            best_configs[asset_class] = {
                'lookback': best_lookback,
                'sharpe': best_sharpe,
                'equity_curve': equity_curves_by_asset[asset_class][best_lookback]
            }
    
    # Plot best configuration for each asset class
    for asset_class, config in best_configs.items():
        equity_curve = config['equity_curve']
        lookback = config['lookback']
        sharpe = config['sharpe']
        color = colors.get(asset_class, '#000000')
        
        label_name = "Balanced 50/50" if asset_class == 'total' else asset_class.capitalize()
        plt.plot(equity_curve.index, equity_curve.values,
                label=f'{label_name} ({lookback}m, Sharpe: {sharpe:.2f})',
                linewidth=3,
                color=color,
                alpha=0.9)
    
    plt.title('TSMOM Strategy: Best Performing Configurations by Asset Class', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Base = 1)', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Format axes
    from matplotlib.ticker import FuncFormatter
    def format_percent(x, pos):
        return f'{x:.1f}x'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_percent))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save combined plot
    results_dir = Path("results/combined")
    results_dir.mkdir(parents=True, exist_ok=True)
    combined_plot_path = results_dir / "best_asset_class_comparison.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"Combined asset class comparison saved: {combined_plot_path}")
    
    plt.show()
    
    # 2. Create performance heatmap across lookbacks and asset classes
    create_performance_heatmap(performance_metrics_by_asset)

def create_performance_heatmap(performance_metrics_by_asset):
    """Create heatmap of performance metrics across asset classes and lookbacks"""
    # Combine all metrics
    all_metrics = []
    for asset_class, metrics in performance_metrics_by_asset.items():
        all_metrics.extend(metrics)
    
    if not all_metrics:
        print("No metrics available for heatmap")
        return
    
    df_all = pd.DataFrame(all_metrics)
    
    # Create Sharpe ratio heatmap
    sharpe_pivot = df_all.pivot(index='asset_class', columns='lookback_months', values='sharpe_ratio')
    
    plt.figure(figsize=(12, 6))
    import seaborn as sns
    sns.heatmap(sharpe_pivot, annot=True, cmap='RdYlGn', center=0, fmt='.2f',
                cbar_kws={'label': 'Sharpe Ratio'})
    plt.title('TSMOM Strategy: Sharpe Ratio Heatmap by Asset Class and Lookback Period', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Lookback Period (months)', fontsize=12)
    plt.ylabel('Asset Class', fontsize=12)
    plt.tight_layout()
    
    # Save heatmap
    results_dir = Path("results/combined")
    heatmap_path = results_dir / "sharpe_ratio_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Sharpe ratio heatmap saved: {heatmap_path}")
    
    plt.show()
    
    # Save combined metrics CSV
    combined_csv_path = results_dir / "combined_asset_class_metrics.csv"
    df_all.to_csv(combined_csv_path, index=False)
    print(f"Combined metrics CSV saved: {combined_csv_path}")

if __name__ == "__main__":
    try:
        results, equity_curves, performance_metrics = test_lookback_periods_by_asset()
        print(f"\n✓ Asset class lookback comparison analysis completed successfully!")
        
        # Print summary of best configurations
        print(f"\n{'='*80}")
        print("BEST CONFIGURATIONS SUMMARY")
        print(f"{'='*80}")
        
        for asset_class in ['total', 'commodities', 'forex']:
            if asset_class in performance_metrics and performance_metrics[asset_class]:
                df = pd.DataFrame(performance_metrics[asset_class])
                best_idx = df['sharpe_ratio'].idxmax()
                best_row = df.loc[best_idx]
                display_name = "BALANCED 50/50" if asset_class == 'total' else asset_class.upper()
                print(f"{display_name}:")
                print(f"  Best lookback: {best_row['lookback_months']} months")
                print(f"  Sharpe ratio: {best_row['sharpe_ratio']:.3f}")
                print(f"  Annual return: {best_row['annual_return']:.1%}")
                print(f"  Max drawdown: {best_row['max_drawdown']:.1%}")
                print()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)