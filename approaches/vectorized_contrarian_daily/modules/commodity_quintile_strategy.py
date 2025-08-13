import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Handle imports for both notebook and script execution
try:
    from .strategy_contrarian import calculate_futures_transaction_costs_ibkr
except (ImportError, ValueError):
    try:
        from strategy_contrarian import calculate_futures_transaction_costs_ibkr
    except ImportError:
        import sys
        import os
        current_dir = os.path.dirname(__file__)
        sys.path.append(current_dir)
        from strategy_contrarian import calculate_futures_transaction_costs_ibkr

def prepare_daily_commodity_data(data_dict):
    """
    Convert commodity data dictionary to unified DataFrame with daily data
    
    Parameters:
    - data_dict: Dictionary with commodity data {ticker: DataFrame}
    
    Returns:
    - combined_df: DataFrame with Close prices for all commodities
    """
    daily_closes = []
    
    for ticker, data in data_dict.items():
        if 'Close' in data.columns:
            close_prices = data['Close'].copy()
            close_prices.name = ticker
            daily_closes.append(close_prices)
    
    if daily_closes:
        combined_df = pd.concat(daily_closes, axis=1)
        # Drop rows where all commodities are NaN
        combined_df = combined_df.dropna(how='all')
        
        print(f"Daily commodity data: {len(combined_df)} days, {len(combined_df.columns)} commodities")
        return combined_df
    else:
        print("No valid commodity data found")
        return pd.DataFrame()

def contrarian_quintiles_daily_strategy(daily_prices, lookback_days=20, apply_transaction_costs=True, volume_tier=1):
    """
    Daily contrarian strategy using only bottom quintile (worst performers)
    - Look back N days to rank performance
    - Long only bottom quintile (worst 20% performers)
    - Equal weight within quintile
    - Prevents lookahead bias with shift(1)
    
    Parameters:
    - daily_prices: DataFrame with daily commodity prices
    - lookback_days: Days for lookback performance calculation
    - apply_transaction_costs: Whether to apply IBKR transaction costs
    - volume_tier: IBKR volume tier (1-4)
    
    Returns:
    - results_df: DataFrame with strategy returns and cumulative returns
    - positions_df: DataFrame with daily positions
    """
    
    # Calculate daily returns
    daily_returns = daily_prices.pct_change(fill_method=None)
    
    # Calculate lookback performance (avoid lookahead bias with shift)
    # The shift(1) ensures we use t-1 data for t decisions
    lookback_performance = daily_returns.rolling(window=lookback_days).sum().shift(1)
    
    # Create quintile rankings for each day (0 to 1, where 0 = worst performer)
    quintile_ranks = lookback_performance.rank(axis=1, pct=True)
    
    # Generate positions (only long bottom quintile)
    positions = pd.DataFrame(0.0, index=quintile_ranks.index, columns=quintile_ranks.columns)
    
    # Long bottom quintile only (worst performers = contrarian long)
    long_mask = quintile_ranks <= 0.2
    positions[long_mask] = 1.0
    
    # Equal weight within quintile
    long_counts = long_mask.sum(axis=1)
    
    # Normalize weights so quintile sums to 1
    for i in range(len(positions)):
        if long_counts.iloc[i] > 0:
            long_positions = positions.iloc[i] == 1.0
            positions.iloc[i, long_positions] = 1.0 / long_counts.iloc[i]
    
    # Calculate gross strategy returns (use next day's returns with today's positions)
    gross_strategy_returns = (positions.shift(1) * daily_returns).sum(axis=1)
    
    # Apply transaction costs if requested
    if apply_transaction_costs:
        total_transaction_costs = pd.Series(0.0, index=daily_returns.index)
        
        for ticker in daily_prices.columns:
            if ticker in positions.columns:
                ticker_positions = positions[ticker]
                ticker_returns = daily_returns[ticker]
                
                # Create fake series that mimics strategy_returns for cost calculation
                ticker_strategy_returns = ticker_positions.shift(1) * ticker_returns
                
                # Calculate transaction costs for this ticker
                ticker_costs = calculate_futures_transaction_costs_ibkr(
                    ticker_strategy_returns, ticker, volume_tier
                )
                
                # Weight the costs by position size
                weighted_costs = ticker_costs * ticker_positions.abs()
                total_transaction_costs += weighted_costs
        
        # Net strategy returns after costs
        net_strategy_returns = gross_strategy_returns - total_transaction_costs
        
        # Store both for analysis
        results_df = pd.DataFrame({
            'strategy_returns': net_strategy_returns,
            'strategy_returns_gross': gross_strategy_returns,
            'transaction_costs': total_transaction_costs,
            'cumulative_returns': (1 + net_strategy_returns).cumprod()
        })
    else:
        # No transaction costs
        results_df = pd.DataFrame({
            'strategy_returns': gross_strategy_returns,
            'cumulative_returns': (1 + gross_strategy_returns).cumprod()
        })
    
    return results_df, positions

def contrarian_full_universe_daily_strategy(daily_prices, lookback_days=20, apply_transaction_costs=True, volume_tier=1):
    """
    Daily contrarian strategy using full universe
    - Look back N days to rank performance  
    - Long all commodities when they have negative performance
    - Equal weight allocation
    - Prevents lookahead bias with shift(1)
    
    Parameters:
    - daily_prices: DataFrame with daily commodity prices
    - lookback_days: Days for lookback performance calculation
    - apply_transaction_costs: Whether to apply IBKR transaction costs
    - volume_tier: IBKR volume tier (1-4)
    
    Returns:
    - results_df: DataFrame with strategy returns and cumulative returns
    - positions_df: DataFrame with daily positions
    """
    
    # Calculate daily returns
    daily_returns = daily_prices.pct_change(fill_method=None)
    
    # Calculate lookback performance (avoid lookahead bias with shift)
    lookback_performance = daily_returns.rolling(window=lookback_days).sum().shift(1)
    
    # Generate positions (long when lookback performance is negative)
    positions = pd.DataFrame(0.0, index=lookback_performance.index, columns=lookback_performance.columns)
    
    # Long when past performance is negative (contrarian logic)
    long_mask = lookback_performance < 0
    positions[long_mask] = 1.0
    
    # Equal weight within active positions
    long_counts = long_mask.sum(axis=1)
    
    # Normalize weights so active positions sum to 1
    for i in range(len(positions)):
        if long_counts.iloc[i] > 0:
            long_positions = positions.iloc[i] == 1.0
            positions.iloc[i, long_positions] = 1.0 / long_counts.iloc[i]
    
    # Calculate gross strategy returns (use next day's returns with today's positions)
    gross_strategy_returns = (positions.shift(1) * daily_returns).sum(axis=1)
    
    # Apply transaction costs if requested
    if apply_transaction_costs:
        total_transaction_costs = pd.Series(0.0, index=daily_returns.index)
        
        for ticker in daily_prices.columns:
            if ticker in positions.columns:
                ticker_positions = positions[ticker]
                ticker_returns = daily_returns[ticker]
                
                # Create fake series that mimics strategy_returns for cost calculation
                ticker_strategy_returns = ticker_positions.shift(1) * ticker_returns
                
                # Calculate transaction costs for this ticker
                ticker_costs = calculate_futures_transaction_costs_ibkr(
                    ticker_strategy_returns, ticker, volume_tier
                )
                
                # Weight the costs by position size
                weighted_costs = ticker_costs * ticker_positions.abs()
                total_transaction_costs += weighted_costs
        
        # Net strategy returns after costs
        net_strategy_returns = gross_strategy_returns - total_transaction_costs
        
        # Store both for analysis
        results_df = pd.DataFrame({
            'strategy_returns': net_strategy_returns,
            'strategy_returns_gross': gross_strategy_returns,
            'transaction_costs': total_transaction_costs,
            'cumulative_returns': (1 + net_strategy_returns).cumprod()
        })
    else:
        # No transaction costs
        results_df = pd.DataFrame({
            'strategy_returns': gross_strategy_returns,
            'cumulative_returns': (1 + gross_strategy_returns).cumprod()
        })
    
    return results_df, positions

def run_commodity_strategies_comparison(data_dict, lookback_days=20, apply_transaction_costs=True, volume_tier=1):
    """
    Run both quintile and full universe strategies on commodity data
    
    Parameters:
    - data_dict: Dictionary with commodity data {ticker: DataFrame}
    - lookback_days: Days for lookback performance calculation
    - apply_transaction_costs: Whether to apply IBKR transaction costs
    - volume_tier: IBKR volume tier (1-4)
    
    Returns:
    - results_dict: Dictionary with results for both strategies
    - positions_dict: Dictionary with positions for both strategies
    """
    
    print(f"Running commodity strategies comparison...")
    print(f"Lookback days: {lookback_days}")
    print(f"Apply transaction costs: {apply_transaction_costs}")
    print(f"Volume tier: {volume_tier}")
    
    # Prepare unified daily data
    daily_prices = prepare_daily_commodity_data(data_dict)
    
    if daily_prices.empty:
        print("ERROR: No valid daily commodity data available")
        return {}, {}
    
    # Run quintiles strategy
    print("\nRunning quintiles strategy (bottom 20% only)...")
    quintiles_results, quintiles_positions = contrarian_quintiles_daily_strategy(
        daily_prices, lookback_days, apply_transaction_costs, volume_tier
    )
    
    # Run full universe strategy  
    print("Running full universe strategy (all negative performers)...")
    universe_results, universe_positions = contrarian_full_universe_daily_strategy(
        daily_prices, lookback_days, apply_transaction_costs, volume_tier
    )
    
    # Store results
    results_dict = {
        'quintiles': quintiles_results,
        'full_universe': universe_results
    }
    
    positions_dict = {
        'quintiles': quintiles_positions,
        'full_universe': universe_positions
    }
    
    print(f"\nStrategies completed successfully!")
    print(f"Quintiles final return: {quintiles_results['cumulative_returns'].iloc[-1]:.4f}")
    print(f"Full universe final return: {universe_results['cumulative_returns'].iloc[-1]:.4f}")
    
    return results_dict, positions_dict

def save_commodity_comparison_results(results_dict, positions_dict, results_dir='../commodities/data/results/quintiles_comparison'):
    """
    Save comparison results to disk
    
    Parameters:
    - results_dict: Dictionary with strategy results
    - positions_dict: Dictionary with strategy positions  
    - results_dir: Directory to save results
    """
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save strategy results
    for strategy_name, results in results_dict.items():
        results_file = os.path.join(results_dir, f'{strategy_name}_results.parquet')
        results.to_parquet(results_file)
        print(f"✓ Saved {strategy_name} results to: {results_file}")
    
    # Save positions
    for strategy_name, positions in positions_dict.items():
        positions_file = os.path.join(results_dir, f'{strategy_name}_positions.parquet')
        positions.to_parquet(positions_file)
        print(f"✓ Saved {strategy_name} positions to: {positions_file}")
    
    # Save combined equity curves for easy comparison
    combined_equity = pd.DataFrame()
    for strategy_name, results in results_dict.items():
        combined_equity[strategy_name] = results['cumulative_returns']
    
    equity_file = os.path.join(results_dir, 'combined_equity_curves.parquet')
    combined_equity.to_parquet(equity_file)
    print(f"✓ Saved combined equity curves to: {equity_file}")
    
    print(f"\nAll results saved to: {results_dir}")