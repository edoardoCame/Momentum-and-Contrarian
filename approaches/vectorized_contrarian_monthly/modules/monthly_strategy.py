import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from pathlib import Path
import os

def prepare_monthly_data(commodity_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert daily commodity data to monthly close prices"""
    monthly_closes = []
    
    for ticker, data in commodity_data.items():
        if 'Close' in data.columns:
            # Resample to month-end, take last close price
            monthly = data['Close'].resample('ME').last()
            monthly.name = ticker
            monthly_closes.append(monthly)
    
    # Combine all commodities into single DataFrame
    if monthly_closes:
        monthly_df = pd.concat(monthly_closes, axis=1)
        
        # Drop rows where all commodities are NaN
        monthly_df = monthly_df.dropna(how='all')
        
        print(f"Monthly data: {len(monthly_df)} months, {len(monthly_df.columns)} commodities")
        return monthly_df
    else:
        print("No valid commodity data found")
        return pd.DataFrame()

def monthly_contrarian_strategy(monthly_prices: pd.DataFrame, lookback_months: int = 6) -> pd.DataFrame:
    """
    Vectorized monthly contrarian strategy using quintiles
    - Look back N months to rank performance
    - Long bottom quintile (losers), short top quintile (winners)  
    - Equal weight within quintiles
    """
    
    # Calculate monthly returns
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    
    # Calculate lookback performance (avoid lookahead bias with shift)
    lookback_performance = monthly_returns.rolling(window=lookback_months).sum().shift(1)
    
    # Create quintile rankings for each month (0 to 1, where 0 = worst performer)
    quintile_ranks = lookback_performance.rank(axis=1, pct=True)
    
    # Generate positions vectorized
    positions = pd.DataFrame(0.0, index=quintile_ranks.index, columns=quintile_ranks.columns)
    
    # Long bottom quintile (worst performers = contrarian long)
    long_mask = quintile_ranks <= 0.2
    positions[long_mask] = 1.0
    
    # Short top quintile (best performers = contrarian short) 
    short_mask = quintile_ranks >= 0.8
    positions[short_mask] = -1.0
    
    # Equal weight within each quintile
    long_counts = long_mask.sum(axis=1)
    short_counts = short_mask.sum(axis=1)
    
    # Normalize weights so each quintile sums to 1 or -1
    for i in range(len(positions)):
        if long_counts.iloc[i] > 0:
            long_positions = positions.iloc[i] == 1.0
            positions.iloc[i, long_positions] = 1.0 / long_counts.iloc[i]
            
        if short_counts.iloc[i] > 0:
            short_positions = positions.iloc[i] == -1.0
            positions.iloc[i, short_positions] = -1.0 / short_counts.iloc[i]
    
    # Calculate strategy returns (use next month's returns with this month's positions)
    strategy_returns = (positions.shift(1) * monthly_returns).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    results_df = pd.DataFrame({
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns
    })
    
    return results_df, positions

def run_parameter_sweep(monthly_prices: pd.DataFrame, 
                       lookback_periods: list = [3, 6, 9, 12]) -> pd.DataFrame:
    """Run strategy across different lookback periods"""
    
    results = []
    
    for lookback in lookback_periods:
        print(f"Testing lookback = {lookback} months...")
        
        try:
            strategy_results, _ = monthly_contrarian_strategy(monthly_prices, lookback)
            
            # Remove NaN values for metrics calculation
            clean_returns = strategy_results['strategy_returns'].dropna()
            
            if len(clean_returns) > 12:  # Need at least 1 year of data
                # Calculate performance metrics
                total_return = strategy_results['cumulative_returns'].iloc[-1] - 1
                annual_return = (1 + total_return) ** (12 / len(clean_returns)) - 1
                annual_vol = clean_returns.std() * np.sqrt(12)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                
                # Max drawdown
                cumulative = strategy_results['cumulative_returns'].dropna()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                results.append({
                    'lookback_months': lookback,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'annual_volatility': annual_vol,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'months_data': len(clean_returns)
                })
                
        except Exception as e:
            print(f"Error with lookback {lookback}: {e}")
    
    return pd.DataFrame(results)

def save_optimization_results(top_strategies_results: Dict[str, pd.DataFrame], 
                            optimization_df: pd.DataFrame,
                            data_dir: str = '../data') -> None:
    """
    Save optimization results and equity curves to data directory
    
    Args:
        top_strategies_results: Dictionary with strategy names and their results DataFrames
        optimization_df: DataFrame with optimization parameter results
        data_dir: Directory to save results
    """
    # Ensure data directory exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Save optimization parameters summary
    optimization_df.to_csv(os.path.join(data_dir, 'optimization_parameters.csv'), index=False)
    print(f"✓ Saved optimization parameters to: {data_dir}/optimization_parameters.csv")
    
    # Save individual strategy equity curves
    for strategy_name, results in top_strategies_results.items():
        # Clean strategy name for filename
        clean_name = strategy_name.replace(' ', '_').replace('-', '').lower()
        
        # Save full results (returns + cumulative)
        results_file = os.path.join(data_dir, f'equity_curve_{clean_name}.csv')
        results.to_csv(results_file)
        print(f"✓ Saved {strategy_name} equity curve to: {results_file}")
        
        # Save just the cumulative returns series for easy plotting
        cumulative_file = os.path.join(data_dir, f'cumulative_returns_{clean_name}.csv')
        cumulative_series = results['cumulative_returns'].to_frame()
        cumulative_series.columns = [strategy_name]
        cumulative_series.to_csv(cumulative_file)
    
    # Save combined cumulative returns for all top strategies
    combined_cumulative = pd.DataFrame()
    for strategy_name, results in top_strategies_results.items():
        combined_cumulative[strategy_name] = results['cumulative_returns']
    
    combined_file = os.path.join(data_dir, 'all_top_strategies_cumulative_returns.csv')
    combined_cumulative.to_csv(combined_file)
    print(f"✓ Saved combined equity curves to: {combined_file}")

def create_daily_equity_curves(commodity_data: Dict[str, pd.DataFrame],
                             strategy_params: Dict[str, int],
                             monthly_prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate daily equity curves while maintaining monthly rebalancing logic
    
    Args:
        commodity_data: Daily commodity price data
        strategy_params: Dictionary with strategy names and their lookback periods
        monthly_prices: Monthly price data for signal generation
        
    Returns:
        Dictionary with strategy names and their daily equity curve DataFrames
    """
    daily_results = {}
    
    # Get the date range for daily data (intersection of all commodities)
    date_ranges = [data.index for data in commodity_data.values()]
    common_start = max(dr.min() for dr in date_ranges)
    common_end = min(dr.max() for dr in date_ranges)
    
    print(f"Daily equity curves date range: {common_start.date()} to {common_end.date()}")
    
    for strategy_name, lookback in strategy_params.items():
        print(f"Generating daily equity curve for {strategy_name} (lookback={lookback})...")
        
        # Generate monthly positions using existing logic
        _, monthly_positions = monthly_contrarian_strategy(monthly_prices, lookback)
        
        # Create daily price DataFrame
        daily_prices = pd.DataFrame()
        for ticker in monthly_positions.columns:
            if ticker in commodity_data and 'Close' in commodity_data[ticker].columns:
                daily_prices[ticker] = commodity_data[ticker]['Close']
        
        # Filter to common date range
        daily_prices = daily_prices.loc[common_start:common_end]
        
        # Calculate daily returns
        daily_returns = daily_prices.pct_change(fill_method=None)
        
        # Forward-fill monthly positions to daily frequency
        # Resample monthly positions to daily (forward fill within each month)
        monthly_positions_reindexed = monthly_positions.reindex(daily_returns.index, method='ffill')
        
        # Calculate daily strategy returns
        daily_strategy_returns = (monthly_positions_reindexed.shift(1) * daily_returns).sum(axis=1)
        
        # Calculate cumulative returns
        daily_cumulative = (1 + daily_strategy_returns.fillna(0)).cumprod()
        
        # Create results DataFrame
        daily_results[strategy_name] = pd.DataFrame({
            'strategy_returns': daily_strategy_returns,
            'cumulative_returns': daily_cumulative
        })
        
        print(f"  ✓ Generated {len(daily_cumulative)} daily points")
    
    return daily_results

def enhanced_optimization_charts(optimization_df: pd.DataFrame, 
                               strategy_results: Dict[str, pd.DataFrame],
                               daily_results: Dict[str, pd.DataFrame] = None,
                               holding_period: str = "1 month") -> None:
    """
    Create enhanced optimization visualization charts with holding period information
    
    Args:
        optimization_df: DataFrame with optimization results
        strategy_results: Dictionary with monthly strategy results
        daily_results: Optional dictionary with daily strategy results
        holding_period: String description of holding period
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. Enhanced Parameter Optimization Charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Monthly Contrarian Strategy Optimization\n(Holding Period: {holding_period})', 
                 fontsize=16)
    
    # Sharpe Ratio
    bars1 = axes[0,0].bar(optimization_df['lookback_months'], optimization_df['sharpe_ratio'], 
                         color='steelblue', alpha=0.8, edgecolor='navy')
    axes[0,0].set_title('Sharpe Ratio by Lookback Period', fontsize=12)
    axes[0,0].set_xlabel('Lookback Period (Months)')
    axes[0,0].set_ylabel('Sharpe Ratio')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, optimization_df['sharpe_ratio']):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Annual Return
    bars2 = axes[0,1].bar(optimization_df['lookback_months'], optimization_df['annual_return']*100, 
                         color='green', alpha=0.8, edgecolor='darkgreen')
    axes[0,1].set_title('Annual Return by Lookback Period', fontsize=12)
    axes[0,1].set_xlabel('Lookback Period (Months)')
    axes[0,1].set_ylabel('Annual Return (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, optimization_df['annual_return']*100):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                      f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Max Drawdown
    bars3 = axes[1,0].bar(optimization_df['lookback_months'], optimization_df['max_drawdown']*100, 
                         color='red', alpha=0.8, edgecolor='darkred')
    axes[1,0].set_title('Maximum Drawdown by Lookback Period', fontsize=12)
    axes[1,0].set_xlabel('Lookback Period (Months)')
    axes[1,0].set_ylabel('Maximum Drawdown (%)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, optimization_df['max_drawdown']*100):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
                      f'{value:.1f}%', ha='center', va='top', fontsize=10)
    
    # Win Rate
    bars4 = axes[1,1].bar(optimization_df['lookback_months'], optimization_df['win_rate']*100, 
                         color='orange', alpha=0.8, edgecolor='darkorange')
    axes[1,1].set_title('Win Rate by Lookback Period', fontsize=12)
    axes[1,1].set_xlabel('Lookback Period (Months)')
    axes[1,1].set_ylabel('Win Rate (%)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, optimization_df['win_rate']*100):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                      f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Equity Curves Comparison (Daily if available, otherwise monthly)
    results_to_plot = daily_results if daily_results else strategy_results
    frequency_label = "Daily" if daily_results else "Monthly"
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = ['steelblue', 'green', 'red', 'orange', 'purple']
    for i, (strategy_name, results) in enumerate(results_to_plot.items()):
        ax.plot(results.index, results['cumulative_returns'], 
                linewidth=2.5, color=colors[i % len(colors)], 
                label=f"{strategy_name} (Holding: {holding_period})", alpha=0.8)
    
    ax.set_title(f'Top 5 Monthly Contrarian Strategies - {frequency_label} Equity Curves Comparison\n'
                f'(Monthly Rebalancing, Holding Period: {holding_period})', 
                fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Enhanced Results Table
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RESULTS SUMMARY (Holding Period: {holding_period})")
    print(f"{'='*80}")
    
    display_df = optimization_df.copy()
    display_df = display_df.sort_values('sharpe_ratio', ascending=False)
    
    print(f"{'Rank':<4} {'Lookback':<10} {'Holding':<10} {'Total Ret':<10} {'Annual Ret':<11} {'Sharpe':<8} {'Max DD':<10} {'Win Rate':<10}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(display_df.iterrows(), 1):
        print(f"{i:<4} {int(row['lookback_months'])} months{'':<2} {holding_period:<10} "
              f"{row['total_return']:<10.1%} {row['annual_return']:<11.1%} "
              f"{row['sharpe_ratio']:<8.3f} {row['max_drawdown']:<10.1%} {row['win_rate']:<10.1%}")

if __name__ == "__main__":
    # Example usage - would need to import data_loader
    from data_loader import load_commodity_data
    
    print("Loading commodity data...")
    commodity_data = load_commodity_data()
    
    print("Preparing monthly data...")
    monthly_prices = prepare_monthly_data(commodity_data)
    
    print("Running parameter sweep...")
    results = run_parameter_sweep(monthly_prices)
    
    print("\nParameter Sweep Results:")
    print(results.round(4))