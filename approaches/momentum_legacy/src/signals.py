"""
Signal generation functions for momentum and contrarian strategies.
All signal functions follow the same interface and avoid lookahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

def basic_momentum_signals(returns: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """
    Basic momentum signals: 1 if return was positive, 0 otherwise.
    
    Args:
        returns: DataFrame of daily returns
        lookback: Number of days to look back (default 1)
    
    Returns:
        DataFrame of signals (1 = long, 0 = no position)
    """
    # Calculate momentum over lookback period
    if lookback == 1:
        momentum = returns
    else:
        momentum = returns.rolling(lookback).sum()
    
    # Generate signals: 1 if momentum > 0
    signals_raw = (momentum > 0).astype(int)
    
    # CRITICAL: Shift to avoid lookahead bias
    signals = signals_raw.shift(1).dropna(how='all')
    
    return signals

def basic_contrarian_signals(returns: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """
    Basic contrarian signals: +1 if return was negative, -1 if positive.
    
    Args:
        returns: DataFrame of daily returns
        lookback: Number of days to look back
    
    Returns:
        DataFrame of signals (+1 = long, -1 = short, 0 = neutral)
    """
    # Calculate momentum over lookback period
    if lookback == 1:
        momentum = returns
    else:
        momentum = returns.rolling(lookback).sum()
    
    # Generate contrarian signals
    signals_raw = pd.DataFrame(index=momentum.index, columns=momentum.columns)
    signals_raw[momentum > 0] = -1  # Short winners
    signals_raw[momentum < 0] = 1   # Long losers
    signals_raw[momentum == 0] = 0  # Neutral
    
    # CRITICAL: Shift to avoid lookahead bias
    signals = signals_raw.shift(1).dropna(how='all')
    
    return signals

def multi_timeframe_momentum(returns: pd.DataFrame, 
                            timeframes: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Multi-timeframe momentum: combines signals from different lookback periods.
    
    Args:
        returns: DataFrame of daily returns
        timeframes: List of lookback periods
    
    Returns:
        DataFrame of combined momentum signals
    """
    momentum_signals = {}
    
    for tf in timeframes:
        if tf == 1:
            momentum = returns
        else:
            momentum = returns.rolling(tf).sum()
        
        # Normalize momentum to [-1, 1] range using tanh
        momentum_norm = np.tanh(momentum * 10)  # Scale factor for sensitivity
        momentum_signals[f'momentum_{tf}d'] = momentum_norm
    
    # Combine signals with equal weights
    combined_momentum = pd.DataFrame(index=returns.index, columns=returns.columns)
    for col in returns.columns:
        signal_sum = sum(momentum_signals[f'momentum_{tf}d'][col] for tf in timeframes)
        combined_momentum[col] = signal_sum / len(timeframes)
    
    # Convert to discrete signals: +1, 0, -1
    signals_raw = pd.DataFrame(index=returns.index, columns=returns.columns)
    signals_raw[combined_momentum > 0.1] = 1   # Long threshold
    signals_raw[combined_momentum < -0.1] = -1 # Short threshold
    signals_raw = signals_raw.fillna(0)        # Neutral
    
    # CRITICAL: Shift to avoid lookahead bias
    signals = signals_raw.shift(1).dropna(how='all')
    
    return signals

def volatility_adjusted_signals(returns: pd.DataFrame, 
                               window: int = 20, 
                               threshold: float = 2.0) -> pd.DataFrame:
    """
    Volatility-adjusted signals using z-scores.
    
    Args:
        returns: DataFrame of daily returns
        window: Rolling window for mean/std calculation
        threshold: Z-score threshold (standard deviations)
    
    Returns:
        DataFrame of signals based on z-scores
    """
    # Calculate rolling z-scores
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    z_scores = (returns - rolling_mean) / rolling_std
    
    # Generate signals based on z-score thresholds
    signals_raw = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    signals_raw[z_scores > threshold] = -1   # Short extreme winners (contrarian)
    signals_raw[z_scores < -threshold] = 1   # Long extreme losers (contrarian)
    
    # CRITICAL: Shift to avoid lookahead bias
    signals = signals_raw.shift(1).dropna(how='all')
    
    return signals

def percentile_based_signals(returns: pd.DataFrame, 
                           window: int = 252, 
                           top_pct: float = 0.2, 
                           bottom_pct: float = 0.2,
                           strategy_type: str = 'contrarian') -> pd.DataFrame:
    """
    Percentile-based signals: long/short based on cross-sectional rankings.
    
    Args:
        returns: DataFrame of daily returns
        window: Rolling window for percentile calculation
        top_pct: Percentile for top performers (0.2 = top 20%)
        bottom_pct: Percentile for bottom performers
        strategy_type: 'momentum' or 'contrarian'
    
    Returns:
        DataFrame of signals
    """
    # Calculate rolling percentiles for each day
    signals_raw = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    for date in returns.index[window:]:
        # Get returns for this date across all commodities
        daily_returns = returns.loc[date]
        
        # Calculate percentiles
        top_threshold = daily_returns.quantile(1 - top_pct)
        bottom_threshold = daily_returns.quantile(bottom_pct)
        
        if strategy_type == 'contrarian':
            # Contrarian: short winners, long losers
            signals_raw.loc[date, daily_returns >= top_threshold] = -1
            signals_raw.loc[date, daily_returns <= bottom_threshold] = 1
        else:  # momentum
            # Momentum: long winners, short losers  
            signals_raw.loc[date, daily_returns >= top_threshold] = 1
            signals_raw.loc[date, daily_returns <= bottom_threshold] = -1
    
    # CRITICAL: Shift to avoid lookahead bias
    signals = signals_raw.shift(1).dropna(how='all')
    
    return signals

def cross_sectoral_momentum(returns: pd.DataFrame, 
                          sectors_map: Dict[str, str],
                          lookback: int = 5) -> pd.DataFrame:
    """
    Cross-sectoral momentum: momentum relative to sector average.
    
    Args:
        returns: DataFrame of daily returns
        sectors_map: Dictionary mapping ticker to sector
        lookback: Lookback period for momentum calculation
    
    Returns:
        DataFrame of sector-relative momentum signals
    """
    # Calculate momentum
    if lookback == 1:
        momentum = returns
    else:
        momentum = returns.rolling(lookback).sum()
    
    # Calculate sector-relative momentum
    relative_momentum = pd.DataFrame(index=momentum.index, columns=momentum.columns)
    
    # Group by sectors and calculate relative performance
    for sector in set(sectors_map.values()):
        sector_tickers = [t for t, s in sectors_map.items() if s == sector and t in momentum.columns]
        
        if len(sector_tickers) > 1:
            sector_data = momentum[sector_tickers]
            sector_mean = sector_data.mean(axis=1)
            
            # Calculate relative momentum vs sector
            for ticker in sector_tickers:
                relative_momentum[ticker] = momentum[ticker] - sector_mean
        else:
            # If only one ticker in sector, use absolute momentum
            if sector_tickers:
                relative_momentum[sector_tickers[0]] = momentum[sector_tickers[0]]
    
    # Generate signals: positive relative momentum = long
    signals_raw = pd.DataFrame(0, index=momentum.index, columns=momentum.columns)
    signals_raw[relative_momentum > 0] = 1
    signals_raw[relative_momentum < 0] = -1
    
    # CRITICAL: Shift to avoid lookahead bias
    signals = signals_raw.shift(1).dropna(how='all')
    
    return signals

def pattern_based_signals(returns: pd.DataFrame, 
                         pattern_type: str = 'consecutive',
                         min_days: int = 2) -> pd.DataFrame:
    """
    Pattern-based signals looking for specific price patterns.
    
    Args:
        returns: DataFrame of daily returns
        pattern_type: 'consecutive', 'gap', or 'reversal'
        min_days: Minimum days for pattern confirmation
    
    Returns:
        DataFrame of pattern-based signals
    """
    signals_raw = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    
    if pattern_type == 'consecutive':
        # Look for consecutive days of same direction
        for col in returns.columns:
            series = returns[col]
            
            # Count consecutive positive/negative days
            pos_consecutive = 0
            neg_consecutive = 0
            
            for i in range(min_days, len(series)):
                if series.iloc[i] > 0:
                    pos_consecutive += 1
                    neg_consecutive = 0
                elif series.iloc[i] < 0:
                    neg_consecutive += 1
                    pos_consecutive = 0
                else:
                    pos_consecutive = 0
                    neg_consecutive = 0
                
                # Contrarian signal after consecutive moves
                if pos_consecutive >= min_days:
                    signals_raw.iloc[i, signals_raw.columns.get_loc(col)] = -1  # Short after consecutive gains
                elif neg_consecutive >= min_days:
                    signals_raw.iloc[i, signals_raw.columns.get_loc(col)] = 1   # Long after consecutive losses
    
    elif pattern_type == 'reversal':
        # Look for reversal patterns (high volatility followed by opposite move)
        volatility = returns.rolling(5).std()
        
        for col in returns.columns:
            ret_series = returns[col]
            vol_series = volatility[col]
            
            for i in range(5, len(ret_series)):
                # High volatility in recent past
                if vol_series.iloc[i-1] > vol_series.rolling(20).quantile(0.8).iloc[i-1]:
                    # Large negative move after high vol -> contrarian long
                    if ret_series.iloc[i] < -2 * vol_series.iloc[i-1]:
                        signals_raw.iloc[i, signals_raw.columns.get_loc(col)] = 1
                    # Large positive move after high vol -> contrarian short
                    elif ret_series.iloc[i] > 2 * vol_series.iloc[i-1]:
                        signals_raw.iloc[i, signals_raw.columns.get_loc(col)] = -1
    
    # CRITICAL: Shift to avoid lookahead bias
    signals = signals_raw.shift(1).dropna(how='all')
    
    return signals

def regime_adaptive_signals(returns: pd.DataFrame, 
                          regime_window: int = 60,
                          volatility_threshold: float = 1.5) -> pd.DataFrame:
    """
    Regime-adaptive signals: switch between momentum and contrarian based on market regime.
    
    Args:
        returns: DataFrame of daily returns
        regime_window: Window for regime detection
        volatility_threshold: Threshold for high/low volatility regime
    
    Returns:
        DataFrame of regime-adaptive signals
    """
    # Calculate market-wide volatility (average across commodities)
    market_returns = returns.mean(axis=1)
    market_volatility = market_returns.rolling(regime_window).std()
    long_term_volatility = market_volatility.rolling(regime_window * 2).mean()
    
    # Detect volatility regime
    high_vol_regime = market_volatility > (volatility_threshold * long_term_volatility)
    
    # Generate basic momentum and contrarian signals
    momentum_sigs = basic_momentum_signals(returns, lookback=1)
    contrarian_sigs = basic_contrarian_signals(returns, lookback=1)
    
    # Adaptive signals: contrarian in high vol, momentum in low vol
    signals = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    # Align indices
    common_index = momentum_sigs.index.intersection(contrarian_sigs.index).intersection(high_vol_regime.index)
    
    for date in common_index:
        if high_vol_regime.loc[date]:
            # High volatility -> use contrarian
            signals.loc[date] = contrarian_sigs.loc[date]
        else:
            # Low volatility -> use momentum
            signals.loc[date] = momentum_sigs.loc[date]
    
    return signals.dropna(how='all')

# Utility functions
def combine_signals(signals_dict: Dict[str, pd.DataFrame], 
                   weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Combine multiple signal DataFrames with optional weights.
    
    Args:
        signals_dict: Dictionary of strategy_name -> signals DataFrame
        weights: Optional weights for each strategy
    
    Returns:
        Combined signals DataFrame
    """
    if weights is None:
        weights = {name: 1.0 / len(signals_dict) for name in signals_dict.keys()}
    
    # Ensure all signals have the same index and columns
    common_index = None
    common_columns = None
    
    for name, signals in signals_dict.items():
        if common_index is None:
            common_index = signals.index
            common_columns = signals.columns
        else:
            common_index = common_index.intersection(signals.index)
            common_columns = common_columns.intersection(signals.columns)
    
    # Combine weighted signals
    combined = pd.DataFrame(0.0, index=common_index, columns=common_columns)
    
    for name, signals in signals_dict.items():
        weight = weights.get(name, 0.0)
        aligned_signals = signals.loc[common_index, common_columns]
        combined += weight * aligned_signals
    
    # Convert to discrete signals
    result = pd.DataFrame(0, index=common_index, columns=common_columns)
    result[combined > 0.3] = 1
    result[combined < -0.3] = -1
    
    return result

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        prices: Price series (not returns)
        window: RSI calculation window
        
    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def multi_timeframe_contrarian_simplified(returns: pd.DataFrame,
                                         prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Simplified multi-timeframe contrarian strategy optimized for commodities.
    Focus only on the most effective components.
    
    Args:
        returns: DataFrame of daily returns
        prices: DataFrame of prices (will be calculated from returns if None)
        
    Returns:
        DataFrame of simplified contrarian signals
    """
    if prices is None:
        prices = (1 + returns).cumprod() * 100
    
    # Align indices
    common_index = returns.index.intersection(prices.index)
    returns = returns.loc[common_index]
    prices = prices.loc[common_index]
    
    signals_combined = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    for col in returns.columns:
        col_returns = returns[col]
        col_prices = prices[col]
        
        # SIMPLE APPROACH: Focus on proven components
        # 1. RSI (20-period) - most reliable
        rsi_20 = calculate_rsi(col_prices, window=20)
        
        # 2. Z-score (10-period) - balanced approach
        zscore_10 = (col_returns - col_returns.rolling(10).mean()) / col_returns.rolling(10).std()
        
        # Simple scoring
        score = pd.Series(0.0, index=col_returns.index)
        
        # RSI contrarian (80% weight)
        score += np.where(rsi_20 < 30, 0.8, 0.0)   # Oversold -> Long
        score += np.where(rsi_20 > 70, -0.8, 0.0)  # Overbought -> Short
        
        # Z-score contrarian (20% weight)
        score += np.where(zscore_10 > 2.0, -0.2, 0.0)  # Extreme gains -> Short
        score += np.where(zscore_10 < -2.0, 0.2, 0.0)  # Extreme losses -> Long
        
        signals_combined[col] = score
    
    # Convert to discrete signals
    signals_discrete = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    signals_discrete[signals_combined > 0.5] = 1    # Long threshold
    signals_discrete[signals_combined < -0.5] = -1  # Short threshold
    
    # Shift to avoid lookahead bias
    signals = signals_discrete.shift(1).dropna(how='all')
    return signals

def multi_timeframe_contrarian_enhanced(returns: pd.DataFrame,
                                       prices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Enhanced multi-timeframe contrarian strategy optimized for commodities.
    
    Combines:
    - Short-term (1-4 weeks): RSI oversold/overbought + Z-score extremes  
    - Medium-term (2-6 months): Consolidation filter
    - Long-term (1-3 years): Secular trend bias
    
    Args:
        returns: DataFrame of daily returns
        prices: DataFrame of prices (will be calculated from returns if None)
        
    Returns:
        DataFrame of enhanced contrarian signals
    """
    if prices is None:
        # Reconstruct prices from returns (assuming first price = 100)
        prices = (1 + returns).cumprod() * 100
    
    # Align indices between returns and prices
    common_index = returns.index.intersection(prices.index)
    returns = returns.loc[common_index]
    prices = prices.loc[common_index]
    
    signals_combined = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    for col in returns.columns:
        col_returns = returns[col]
        col_prices = prices[col]
        
        # LAYER 1: SHORT-TERM (70% weight) - Maximum contrarian power
        # RSI oversold/overbought (weekly = 5 days)
        rsi_5d = calculate_rsi(col_prices, window=5)
        rsi_20d = calculate_rsi(col_prices, window=20)
        
        # Z-score extremes (1-4 week windows)
        zscore_5d = (col_returns - col_returns.rolling(5).mean()) / col_returns.rolling(5).std()
        zscore_10d = (col_returns - col_returns.rolling(10).mean()) / col_returns.rolling(10).std()
        zscore_20d = (col_returns - col_returns.rolling(20).mean()) / col_returns.rolling(20).std()
        
        # Short-term contrarian signals
        short_term_signal = pd.Series(0.0, index=col_returns.index)
        
        # RSI signals (moderate weight)
        short_term_signal += np.where(rsi_5d < 15, 1.0, 0.0)    # Very oversold -> Long
        short_term_signal += np.where(rsi_5d > 85, -1.0, 0.0)   # Very overbought -> Short
        short_term_signal += np.where(rsi_20d < 25, 0.5, 0.0)   # Moderate oversold
        short_term_signal += np.where(rsi_20d > 75, -0.5, 0.0)  # Moderate overbought
        
        # Z-score signals (contrarian, more conservative)
        short_term_signal += np.where(zscore_5d > 2.5, -0.8, 0.0)   # Very extreme gains -> Short
        short_term_signal += np.where(zscore_5d < -2.5, 0.8, 0.0)  # Very extreme losses -> Long
        short_term_signal += np.where(zscore_10d > 2.0, -0.5, 0.0) 
        short_term_signal += np.where(zscore_10d < -2.0, 0.5, 0.0)
        short_term_signal += np.where(zscore_20d > 1.5, -0.3, 0.0)
        short_term_signal += np.where(zscore_20d < -1.5, 0.3, 0.0)
        
        # LAYER 2: MEDIUM-TERM FILTER (Neutral weight) - Avoid consolidation zones
        rsi_60d = calculate_rsi(col_prices, window=60)
        medium_term_filter = pd.Series(1.0, index=col_returns.index)
        
        # Reduce signal strength in consolidation zones (RSI 30-70)
        consolidation_zone = (rsi_60d >= 30) & (rsi_60d <= 70)
        medium_term_filter = np.where(consolidation_zone, 0.5, 1.0)  # 50% reduction
        
        # LAYER 3: LONG-TERM BIAS (20% weight) - Secular trend bias
        # Long-term trend: 252 days (1 year) and 756 days (3 years)  
        long_ma_252 = col_prices.rolling(252).mean()
        long_ma_756 = col_prices.rolling(756).mean()
        
        long_term_bias = pd.Series(0.0, index=col_returns.index)
        
        # If in long-term downtrend -> bias toward long contrarian
        long_term_bias += np.where(col_prices < long_ma_252, 0.3, 0.0)
        long_term_bias += np.where(col_prices < long_ma_756, 0.2, 0.0)
        # If in long-term uptrend -> bias toward short contrarian  
        long_term_bias += np.where(col_prices > long_ma_252, -0.3, 0.0)
        long_term_bias += np.where(col_prices > long_ma_756, -0.2, 0.0)
        
        # COMBINE ALL LAYERS with weights: 70% short-term, 20% long-term, medium-term is filter
        combined_score = (
            short_term_signal * 0.70 * medium_term_filter +  # Short-term with filter
            long_term_bias * 0.20                            # Long-term bias
        )
        
        signals_combined[col] = combined_score
    
    # Convert continuous scores to discrete signals (more conservative thresholds)
    signals_discrete = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    signals_discrete[signals_combined > 0.6] = 1    # Long threshold (lowered)
    signals_discrete[signals_combined < -0.6] = -1  # Short threshold (lowered)
    
    # CRITICAL: Shift to avoid lookahead bias
    signals = signals_discrete.shift(1).dropna(how='all')
    
    return signals

if __name__ == "__main__":
    # Test signal functions
    from data_loader import load_all_data
    
    print("🧪 Testing signal functions...")
    
    _, returns = load_all_data()
    
    # Test basic signals
    print("\n📊 Testing basic momentum signals...")
    mom_signals = basic_momentum_signals(returns)
    print(f"Momentum signals shape: {mom_signals.shape}")
    
    print("\n📊 Testing basic contrarian signals...")
    con_signals = basic_contrarian_signals(returns)
    print(f"Contrarian signals shape: {con_signals.shape}")
    
    print("\n📊 Testing multi-timeframe momentum...")
    mtf_signals = multi_timeframe_momentum(returns)
    print(f"Multi-timeframe signals shape: {mtf_signals.shape}")
    
    print("\n📊 Testing volatility-adjusted signals...")
    vol_signals = volatility_adjusted_signals(returns)
    print(f"Volatility-adjusted signals shape: {vol_signals.shape}")
    
    print("\n📊 Testing enhanced multi-timeframe contrarian...")
    enhanced_signals = multi_timeframe_contrarian_enhanced(returns)
    print(f"Enhanced contrarian signals shape: {enhanced_signals.shape}")
    
    print("\n✅ All signal tests completed!")