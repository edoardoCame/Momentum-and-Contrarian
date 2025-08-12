import pandas as pd
import numpy as np
from typing import Dict, Tuple

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