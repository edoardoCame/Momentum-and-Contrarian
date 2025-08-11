#!/usr/bin/env python3
"""
Cross-Sectional Momentum & Contrarian Strategies on Commodity Futures
Replicates Jegadeesh-Titman (1993) methodology for commodity markets

Implementation of cross-sectional momentum and contrarian strategies on commodity futures
following academic literature methodology. Supports overlapping portfolios, quintile-based
ranking, and comprehensive performance evaluation with Newey-West statistics.

Features:
- Yahoo Finance data download with yfinance
- Cross-sectional quintile formation (no gap months)
- Overlapping portfolio construction
- Momentum (R ∈ {1,3,6,12}) and Contrarian (R ∈ {24,36,60}) strategies  
- Multiple holding periods H (1-60 months)
- Long-short, long-only, and benchmark variants
- Newey-West t-statistics for overlapping returns
- Comprehensive performance metrics and visualization

Author: Generated with Claude Code
Version: 1.0
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from scipy import stats
# from statsmodels.tsa.stattools import acf
# from statsmodels.stats.diagnostic import acorr_ljungbox
import sys

# Import our cache manager
from data_cache_manager import DataCacheManager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Default commodity universe (Yahoo Finance tickers)
DEFAULT_UNIVERSE = {
    'Energy': ['CL=F', 'BZ=F', 'NG=F', 'RB=F', 'HO=F'],
    'Metals': ['GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F'],  
    'Grain_Oilseeds': ['ZC=F', 'ZW=F', 'KE=F', 'ZS=F', 'ZM=F', 'ZL=F', 'ZO=F'],
    'Softs': ['KC=F', 'SB=F', 'CC=F', 'CT=F', 'OJ=F'],
    'Livestock_Dairy': ['LE=F', 'HE=F', 'GF=F', 'DC=F'],
    'Other': ['LBS=F']  # Lumber - will try LBR=F as fallback
}

# Flatten universe for easy access
ALL_TICKERS = [ticker for category in DEFAULT_UNIVERSE.values() for ticker in category]

# Contract month codes for stitching mode
CONTRACT_MONTHS = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_logging(output_dir: Path) -> None:
    """Setup logging configuration."""
    log_file = output_dir / 'backtest.log'
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_output_dirs(base_dir: Path) -> Dict[str, Path]:
    """Create output directory structure."""
    dirs = {
        'results': base_dir / 'results',
        'series': base_dir / 'results' / 'series', 
        'figures': base_dir / 'results' / 'figures',
        'summary': base_dir / 'results' / 'summary'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

# =============================================================================
# DATA DOWNLOAD AND PREPROCESSING
# =============================================================================

def download_prices(tickers: List[str], start: str, end: str, freq: str = 'D', 
                   use_cache: bool = True, cache_manager: DataCacheManager = None) -> pd.DataFrame:
    """
    Download commodity prices from Yahoo Finance with caching support.
    
    Args:
        tickers: List of Yahoo Finance ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        freq: Frequency ('D' for daily)
        use_cache: Whether to use cached data if available
        cache_manager: DataCacheManager instance
    
    Returns:
        DataFrame with adjusted close prices
    """
    # Try to load from cache first if requested
    if use_cache and cache_manager is not None:
        cached_data = cache_manager.load_price_data()
        if cached_data[0] is not None:  # daily_prices exists
            daily_prices, monthly_prices, monthly_returns, metadata = cached_data
            
            # Check if cached data matches our requirements
            cached_tickers = set(metadata['tickers_successful'])
            requested_tickers = set(tickers)
            
            if (cached_tickers.issuperset(requested_tickers) and 
                metadata['start_date'] <= start and 
                metadata['end_date'] >= end):
                
                print("✅ Using cached price data")
                return daily_prices[tickers]  # Return only requested tickers
    print(f"📥 Downloading prices for {len(tickers)} commodities...")
    print(f"📅 Period: {start} to {end}")
    
    price_data = pd.DataFrame()
    failed_tickers = []
    successful_tickers = []
    
    for ticker in tqdm(tickers, desc="Downloading"):
        try:
            # Download data
            data = yf.download(ticker, start=start, end=end, progress=False)
            
            if data.empty:
                print(f"⚠️  No data available for {ticker}")
                failed_tickers.append(ticker)
                continue
                
            # Use Adj Close if available, otherwise Close
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']
            
            # Handle potential MultiIndex from yfinance
            if isinstance(prices, pd.DataFrame) and len(prices.columns) == 1:
                prices = prices.iloc[:, 0]
            
            price_data[ticker] = prices
            successful_tickers.append(ticker)
            
        except Exception as e:
            print(f"❌ Failed to download {ticker}: {str(e)}")
            failed_tickers.append(ticker)
            continue
    
    # Clean data
    price_data = price_data.dropna(how='all')
    
    print(f"✅ Successfully downloaded {len(successful_tickers)} commodities")
    print(f"❌ Failed to download {len(failed_tickers)} commodities")
    if failed_tickers:
        print(f"   Failed tickers: {failed_tickers}")
    
    print(f"📊 Final dataset: {len(price_data)} observations x {len(price_data.columns)} commodities")
    
    # Save to cache if cache manager is provided
    if cache_manager is not None:
        # Also calculate monthly prices and returns for caching
        monthly_prices = to_monthly(price_data)
        monthly_returns = compute_monthly_returns(monthly_prices)
        
        cache_manager.save_price_data(
            daily_prices=price_data,
            monthly_prices=monthly_prices, 
            monthly_returns=monthly_returns,
            tickers=tickers,
            start_date=start,
            end_date=end
        )
    
    return price_data

def to_monthly(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily prices to monthly (month-end) prices.
    
    Args:
        df_prices: DataFrame with daily prices
    
    Returns:
        DataFrame with month-end prices
    """
    print("📅 Converting daily prices to month-end...")
    
    # Group by month-end and take last available price
    monthly_prices = df_prices.groupby(pd.Grouper(freq='M')).last()
    
    # Remove months with no data
    monthly_prices = monthly_prices.dropna(how='all')
    
    print(f"📊 Monthly data: {len(monthly_prices)} months x {len(monthly_prices.columns)} commodities")
    
    return monthly_prices

def compute_monthly_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly log returns from monthly prices.
    
    Args:
        df_prices: DataFrame with monthly prices
    
    Returns:
        DataFrame with monthly returns
    """
    print("🔢 Computing monthly log returns...")
    
    # Calculate log returns: ln(P_t / P_{t-1})
    returns = np.log(df_prices / df_prices.shift(1))
    
    # Remove first row (NaN returns)
    returns = returns.dropna(how='all')
    
    print(f"📊 Returns data: {len(returns)} months x {len(returns.columns)} commodities")
    print(f"📈 Returns period: {returns.index.min().date()} to {returns.index.max().date()}")
    
    return returns

# =============================================================================
# CROSS-SECTIONAL STRATEGY FUNCTIONS  
# =============================================================================

def form_quintiles(signal_scores: pd.Series, n_quintiles: int = 5) -> Dict[int, List[str]]:
    """
    Form quintiles based on cross-sectional ranking of signals.
    
    Args:
        signal_scores: Series with asset signals (higher = better momentum)
        n_quintiles: Number of quintiles to form
    
    Returns:
        Dictionary mapping quintile number to list of tickers
        Q1 = Losers (lowest signal), Q5 = Winners (highest signal)
    """
    # Remove NaN values
    valid_scores = signal_scores.dropna()
    
    if len(valid_scores) < n_quintiles:
        return {q: [] for q in range(1, n_quintiles + 1)}
    
    # Rank scores (1 = lowest, n = highest) 
    ranks = valid_scores.rank(method='first')
    
    # Determine quintile breakpoints
    quintile_size = len(valid_scores) / n_quintiles
    
    quintiles = {}
    for q in range(1, n_quintiles + 1):
        if q == n_quintiles:
            # Last quintile gets remaining assets
            lower_bound = (q - 1) * quintile_size
            mask = ranks > lower_bound
        else:
            lower_bound = (q - 1) * quintile_size
            upper_bound = q * quintile_size
            mask = (ranks > lower_bound) & (ranks <= upper_bound)
        
        quintiles[q] = valid_scores[mask].index.tolist()
    
    return quintiles

def calculate_ranking_signal(returns: pd.DataFrame, date: pd.Timestamp, R: int, 
                           min_hist: int = 12) -> pd.Series:
    """
    Calculate ranking signal for cross-sectional strategy at given date.
    
    Args:
        returns: DataFrame of monthly returns
        date: Current date for signal calculation
        R: Ranking window in months
        min_hist: Minimum history required in months
    
    Returns:
        Series of signals for eligible assets
    """
    # Find position of current date
    try:
        date_idx = returns.index.get_loc(date)
    except KeyError:
        return pd.Series(dtype=float)
    
    # Check if we have enough history
    if date_idx < max(R, min_hist):
        return pd.Series(dtype=float)
    
    # Get ranking period returns (exclude current month)
    ranking_start = date_idx - R
    ranking_end = date_idx  # Exclusive, so this excludes current month
    
    ranking_returns = returns.iloc[ranking_start:ranking_end]
    
    # Calculate signal as arithmetic mean of monthly returns over R months
    signals = ranking_returns.mean()
    
    # Filter assets with sufficient history
    valid_assets = []
    for asset in signals.index:
        asset_history = returns.loc[:date, asset].dropna()
        if len(asset_history) >= min_hist:
            valid_assets.append(asset)
    
    return signals[valid_assets]

def run_cs_strategy(returns: pd.DataFrame, R: int, H: int, mode: str = 'momentum',
                   costs_bps: float = 0, min_hist: int = 12, 
                   n_quintiles: int = 5) -> Dict[str, pd.Series]:
    """
    Run cross-sectional strategy with overlapping portfolios.
    
    Args:
        returns: Monthly returns DataFrame
        R: Ranking/formation period in months
        H: Holding period in months  
        mode: 'momentum' or 'contrarian'
        costs_bps: Transaction costs in basis points
        min_hist: Minimum history required
        n_quintiles: Number of quintiles
    
    Returns:
        Dictionary with strategy return series
    """
    print(f"🚀 Running {mode} strategy R={R}, H={H}")
    
    # Initialize results storage
    cohort_returns = {}  # Store returns for each cohort
    portfolio_returns = {
        'long_short': [],
        'long_only_q5': [],
        'equal_weight': []
    }
    
    rebalance_dates = []
    n_cohorts_active = []
    
    # Main loop over rebalancing dates
    for i, date in enumerate(returns.index):
        # Skip if not enough history
        if i < max(R, min_hist):
            continue
            
        rebalance_dates.append(date)
        
        # Calculate ranking signals
        signals = calculate_ranking_signal(returns, date, R, min_hist)
        
        if len(signals) < n_quintiles:
            # Not enough assets - skip this period
            for strategy in portfolio_returns:
                portfolio_returns[strategy].append(0.0)
            n_cohorts_active.append(0)
            continue
        
        # Form quintiles
        quintiles = form_quintiles(signals, n_quintiles)
        
        # Get winners and losers based on strategy mode
        if mode == 'momentum':
            winners = quintiles[5]  # Q5 - highest signals
            losers = quintiles[1]   # Q1 - lowest signals  
        else:  # contrarian
            winners = quintiles[1]  # Q1 - lowest signals (buy losers)
            losers = quintiles[5]   # Q5 - highest signals (sell winners)
        
        # Create new cohort for this rebalancing date
        cohort_key = f"{date.strftime('%Y-%m')}_R{R}H{H}"
        cohort_returns[cohort_key] = {
            'start_date': date,
            'winners': winners,
            'losers': losers,
            'returns': [],
            'active': True
        }
        
        # Apply transaction costs when cohort is formed
        transaction_cost = costs_bps / 10000.0  # Convert bps to decimal
        
        # Calculate portfolio returns for this month
        ls_return = 0.0    # Long-short return
        lo_return = 0.0    # Long-only Q5 return  
        ew_return = 0.0    # Equal-weight return
        
        active_cohorts = 0
        
        # Loop through all active cohorts
        for cohort_key, cohort_data in cohort_returns.items():
            if not cohort_data['active']:
                continue
                
            # Check if cohort should be closed
            months_active = len(cohort_data['returns'])
            if months_active >= H:
                cohort_data['active'] = False
                continue
            
            active_cohorts += 1
            
            # Calculate cohort returns for this month
            if i < len(returns):
                current_month_returns = returns.iloc[i]
                
                # Long-short portfolio
                winner_return = current_month_returns[cohort_data['winners']].mean() if cohort_data['winners'] else 0
                loser_return = current_month_returns[cohort_data['losers']].mean() if cohort_data['losers'] else 0
                
                cohort_ls_return = winner_return - loser_return
                
                # Apply transaction costs only in first month
                if months_active == 0:
                    cohort_ls_return -= 2 * transaction_cost  # Cost for both long and short legs
                
                # Long-only Q5 portfolio  
                cohort_lo_return = winner_return
                if months_active == 0:
                    cohort_lo_return -= transaction_cost
                
                # Equal-weight portfolio (all eligible assets)
                all_assets = list(set(cohort_data['winners'] + cohort_data['losers']))
                cohort_ew_return = current_month_returns[all_assets].mean() if all_assets else 0
                if months_active == 0:
                    cohort_ew_return -= transaction_cost
                
                # Store cohort returns
                cohort_data['returns'].append({
                    'long_short': cohort_ls_return,
                    'long_only_q5': cohort_lo_return, 
                    'equal_weight': cohort_ew_return
                })
                
                # Add to aggregate portfolio returns
                ls_return += cohort_ls_return
                lo_return += cohort_lo_return  
                ew_return += cohort_ew_return
        
        # Average across active cohorts (overlapping portfolio methodology)
        if active_cohorts > 0:
            portfolio_returns['long_short'].append(ls_return / active_cohorts)
            portfolio_returns['long_only_q5'].append(lo_return / active_cohorts)
            portfolio_returns['equal_weight'].append(ew_return / active_cohorts)
        else:
            portfolio_returns['long_short'].append(0.0)
            portfolio_returns['long_only_q5'].append(0.0)
            portfolio_returns['equal_weight'].append(0.0)
        
        n_cohorts_active.append(active_cohorts)
    
    # Convert to pandas Series
    result_index = pd.DatetimeIndex(rebalance_dates)
    results = {}
    
    for strategy_type, returns_list in portfolio_returns.items():
        results[strategy_type] = pd.Series(returns_list, index=result_index)
    
    # Add diagnostic information
    results['n_cohorts_active'] = pd.Series(n_cohorts_active, index=result_index)
    results['rebalance_dates'] = result_index
    
    print(f"✅ Completed {mode} R={R}, H={H}: {len(result_index)} periods")
    
    return results

# =============================================================================
# PERFORMANCE EVALUATION
# =============================================================================

def nw_tstat(returns: pd.Series, lags: int = None) -> float:
    """
    Calculate Newey-West t-statistic for testing if mean return is zero.
    
    Args:
        returns: Series of returns
        lags: Number of lags for Newey-West correction (default: max(1, sqrt(T)))
    
    Returns:
        Newey-West t-statistic
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 2:
        return np.nan
    
    n = len(returns_clean)
    mean_ret = returns_clean.mean()
    
    if lags is None:
        lags = max(1, int(np.sqrt(n)))
    
    # Calculate Newey-West variance estimator
    gamma_0 = returns_clean.var(ddof=1)  # Sample variance
    
    nw_var = gamma_0
    
    # Add autocovariance terms
    for lag in range(1, min(lags + 1, n)):
        if n - lag > 0:
            gamma_lag = np.cov(returns_clean.iloc[:-lag], returns_clean.iloc[lag:])[0, 1]
            weight = 1 - lag / (lags + 1)  # Bartlett weights
            nw_var += 2 * weight * gamma_lag
    
    # Calculate t-statistic  
    nw_se = np.sqrt(nw_var / n)
    
    if nw_se == 0:
        return np.nan
        
    t_stat = mean_ret / nw_se
    
    return t_stat

def evaluate(returns: pd.Series, rf_rate: pd.Series = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a return series.
    
    Args:
        returns: Monthly return series
        rf_rate: Risk-free rate series (optional)
    
    Returns:
        Dictionary of performance metrics
    """
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 2:
        return {metric: np.nan for metric in ['ann_return', 'ann_vol', 'sharpe', 
                                             'hit_ratio', 't_stat', 'max_dd', 'calmar']}
    
    # Basic statistics
    n_months = len(returns_clean)
    ann_return = returns_clean.mean() * 12
    ann_vol = returns_clean.std() * np.sqrt(12)
    
    # Risk-adjusted metrics
    if rf_rate is not None and len(rf_rate) > 0:
        # Align risk-free rate with returns
        rf_aligned = rf_rate.reindex(returns_clean.index, method='ffill')
        excess_returns = returns_clean - rf_aligned.fillna(0) / 12  # Monthly RF rate
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(12) if excess_returns.std() > 0 else np.nan
    else:
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    
    # Hit ratio
    hit_ratio = (returns_clean > 0).mean()
    
    # Newey-West t-statistic  
    t_stat = nw_tstat(returns_clean)
    
    # Drawdown analysis
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    
    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol, 
        'sharpe': sharpe,
        'hit_ratio': hit_ratio,
        't_stat': t_stat,
        'max_dd': max_dd,
        'calmar': calmar,
        'n_months': n_months
    }

# =============================================================================
# GRID BACKTESTING
# =============================================================================

def grid_backtest(returns: pd.DataFrame, R_list: List[int], H_list: List[int], 
                 modes: List[str], **kwargs) -> pd.DataFrame:
    """
    Run backtests across grid of R and H parameters.
    
    Args:
        returns: Monthly returns DataFrame
        R_list: List of ranking periods
        H_list: List of holding periods  
        modes: List of strategy modes ('momentum', 'contrarian')
        **kwargs: Additional arguments passed to run_cs_strategy
    
    Returns:
        DataFrame with summary results
    """
    print("🔄 Running grid backtest...")
    
    results = []
    total_combinations = len(R_list) * len(H_list) * len(modes) * 3  # 3 portfolio types
    
    with tqdm(total=total_combinations, desc="Grid Backtest") as pbar:
        for mode in modes:
            for R in R_list:
                for H in H_list:
                    try:
                        # Run strategy
                        strategy_results = run_cs_strategy(
                            returns, R, H, mode=mode, **kwargs
                        )
                        
                        # Evaluate each portfolio type
                        for portfolio_type in ['long_short', 'long_only_q5', 'equal_weight']:
                            if portfolio_type in strategy_results:
                                metrics = evaluate(strategy_results[portfolio_type])
                                
                                result_row = {
                                    'Strategy': mode.title(),
                                    'R': R,
                                    'H': H, 
                                    'Portfolio_Type': portfolio_type,
                                    'Strategy_Key': f"{mode}_R{R}_H{H}_{portfolio_type}",
                                    **metrics
                                }
                                
                                results.append(result_row)
                            
                            pbar.update(1)
                        
                    except Exception as e:
                        print(f"⚠️  Error in {mode} R={R} H={H}: {str(e)}")
                        # Add empty results to maintain structure
                        for portfolio_type in ['long_short', 'long_only_q5', 'equal_weight']:
                            result_row = {
                                'Strategy': mode.title(),
                                'R': R,
                                'H': H,
                                'Portfolio_Type': portfolio_type, 
                                'Strategy_Key': f"{mode}_R{R}_H{H}_{portfolio_type}",
                                **{metric: np.nan for metric in ['ann_return', 'ann_vol', 'sharpe', 
                                                                'hit_ratio', 't_stat', 'max_dd', 'calmar', 'n_months']}
                            }
                            results.append(result_row)
                            pbar.update(1)
    
    results_df = pd.DataFrame(results)
    
    print(f"✅ Grid backtest completed: {len(results_df)} combinations")
    
    return results_df

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_heatmap(summary_df: pd.DataFrame, metric: str = 'sharpe', 
                strategy: str = 'Momentum', portfolio_type: str = 'long_short',
                save_path: Path = None) -> None:
    """Plot heatmap of performance metric across R and H dimensions."""
    
    # Filter data
    data = summary_df[
        (summary_df['Strategy'] == strategy) & 
        (summary_df['Portfolio_Type'] == portfolio_type)
    ].copy()
    
    if data.empty:
        print(f"No data for {strategy} {portfolio_type}")
        return
    
    # Pivot for heatmap
    heatmap_data = data.pivot(index='R', columns='H', values=metric)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Determine color map based on metric
    if metric in ['sharpe', 'ann_return', 'calmar', 't_stat']:
        cmap = 'RdYlGn'
    else:  # max_dd, ann_vol
        cmap = 'RdYlGn_r'
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        center=0 if metric in ['ann_return', 't_stat'] else None,
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    
    plt.title(f'{strategy} Strategy: {metric.replace("_", " ").title()}\n({portfolio_type.replace("_", " ").title()})')
    plt.xlabel('Holding Period (H) - Months')
    plt.ylabel('Ranking Period (R) - Months')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / f'{strategy.lower()}_{portfolio_type}_{metric}_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_equity_curves(strategy_results: Dict[str, pd.Series], save_path: Path = None) -> None:
    """Plot equity curves for multiple strategies."""
    
    plt.figure(figsize=(15, 10))
    
    for i, (strategy_name, returns) in enumerate(strategy_results.items()):
        if 'n_cohorts' in strategy_name or 'rebalance' in strategy_name:
            continue  # Skip diagnostic series
            
        equity_curve = (1 + returns.fillna(0)).cumprod()
        
        plt.plot(equity_curve.index, equity_curve.values, 
                label=strategy_name, linewidth=2)
    
    plt.title('Commodity Cross-Sectional Strategy Equity Curves', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'equity_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_returns_distribution(returns: pd.Series, title: str = "Returns Distribution", 
                            save_path: Path = None) -> None:
    """Plot distribution of monthly returns."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(returns.dropna(), bins=30, alpha=0.7, density=True)
    ax1.set_title(f'{title} - Histogram')
    ax1.set_xlabel('Monthly Return')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot against normal distribution
    stats.probplot(returns.dropna(), dist="norm", plot=ax2)
    ax2.set_title(f'{title} - Q-Q Plot vs Normal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# =============================================================================
# CONTRACT STITCHING (OPTIONAL ADVANCED FEATURE)
# =============================================================================

def generate_contract_list(root_symbol: str, start_year: int, end_year: int) -> List[str]:
    """Generate list of futures contracts for stitching mode."""
    contracts = []
    
    # Simplified contract generation - would need exchange-specific logic for production
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_code = CONTRACT_MONTHS[month]
            
            # Use 2-digit year
            year_code = str(year)[-2:]
            
            # Different exchanges have different formats
            if root_symbol in ['CL', 'NG', 'RB', 'HO']:  # NYMEX
                contract = f"{root_symbol}{month_code}{year_code}.NYM"
            elif root_symbol in ['ZC', 'ZW', 'ZS', 'ZM', 'ZL', 'ZO', 'LE', 'HE', 'GF']:  # CME
                contract = f"{root_symbol}{month_code}{year_code}.CME"
            elif root_symbol in ['KC', 'SB', 'CC', 'CT', 'OJ']:  # ICE
                contract = f"{root_symbol}{month_code}{year_code}.ICE"
            else:  # Default format
                contract = f"{root_symbol}{month_code}{year_code}"
            
            contracts.append(contract)
    
    return contracts

def stitch_contracts(root_symbol: str, start_date: str, end_date: str) -> pd.Series:
    """
    Stitch individual futures contracts to create continuous series.
    This is a simplified implementation - production would need more sophisticated logic.
    """
    print(f"⚠️  Contract stitching is experimental for {root_symbol}")
    
    # For now, fall back to continuous contract
    continuous_ticker = f"{root_symbol}=F"
    
    try:
        data = yf.download(continuous_ticker, start=start_date, end=end_date, progress=False)
        
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        else:
            return data['Close']
            
    except Exception as e:
        print(f"❌ Stitching failed for {root_symbol}, using continuous: {e}")
        return pd.Series(dtype=float)

# =============================================================================
# MAIN EXECUTION AND CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-Sectional Momentum & Contrarian Strategies on Commodity Futures",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--start', type=str, default='1985-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-07-31', 
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--universe', type=str, default='default',
                       help='Universe: "default" or path to ticker file')
    
    # Strategy parameters
    parser.add_argument('--ranking_windows', type=str, default='1,3,6,12,24,36,60',
                       help='Ranking windows R (comma-separated)')
    parser.add_argument('--holding_windows', type=str, default='1,3,6,12,24,36,60',
                       help='Holding windows H (comma-separated)')
    parser.add_argument('--quintiles', type=int, default=5,
                       help='Number of quintiles')
    parser.add_argument('--strategy', type=str, default='both',
                       choices=['momentum', 'contrarian', 'both'],
                       help='Strategy type')
    
    # Technical parameters
    parser.add_argument('--no_gap_month', type=bool, default=True,
                       help='No gap month between ranking and holding')
    parser.add_argument('--stitch_mode', type=bool, default=False,
                       help='Use contract stitching instead of continuous')
    parser.add_argument('--min_history_months', type=int, default=12,
                       help='Minimum history required')
    parser.add_argument('--min_liquidity_median_volume', type=float, default=0,
                       help='Minimum liquidity filter')
    parser.add_argument('--transaction_cost_bps', type=float, default=0,
                       help='Transaction costs in basis points')
    parser.add_argument('--risk_free', type=str, default='^IRX',
                       help='Risk-free rate ticker (^IRX for 3M Treasury)')
    
    # Output parameters  
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    
    # Cache parameters
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use cached data if available (default: True)')
    parser.add_argument('--force_refresh', action='store_true', default=False,
                       help='Force refresh data ignoring cache')
    parser.add_argument('--cache_status', action='store_true', default=False,
                       help='Show cache status and exit')
    parser.add_argument('--clear_cache', type=str, choices=['all', 'data', 'strategy'], 
                       help='Clear cache: "all", "data", or "strategy"')
    
    return parser.parse_args()

def load_custom_universe(file_path: str) -> List[str]:
    """Load custom ticker universe from file."""
    with open(file_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return tickers

def download_risk_free_rate(ticker: str, start: str, end: str) -> pd.Series:
    """Download risk-free rate data."""
    try:
        print(f"📥 Downloading risk-free rate: {ticker}")
        rf_data = yf.download(ticker, start=start, end=end, progress=False)
        
        if rf_data.empty:
            print(f"⚠️  No risk-free rate data available")
            return pd.Series(dtype=float)
        
        # Convert to monthly and decimal format
        rf_monthly = rf_data['Close'].groupby(pd.Grouper(freq='M')).last() / 100
        
        print(f"✅ Risk-free rate downloaded: {len(rf_monthly)} months")
        return rf_monthly
        
    except Exception as e:
        print(f"⚠️  Risk-free rate download failed: {e}")
        return pd.Series(dtype=float)

def save_results(summary_df: pd.DataFrame, strategy_results: Dict, 
                output_dirs: Dict[str, Path]) -> None:
    """Save all results to files."""
    print("💾 Saving results...")
    
    # Save summary table
    summary_df.to_csv(output_dirs['summary'] / 'strategy_summary.csv', index=False)
    
    # Save individual strategy time series
    for strategy_key, returns in strategy_results.items():
        if 'n_cohorts' in strategy_key or 'rebalance' in strategy_key:
            continue
            
        clean_key = strategy_key.replace(' ', '_').replace('-', '_')
        returns.to_csv(output_dirs['series'] / f'{clean_key}_returns.csv')
        
        # Calculate and save equity curve
        equity_curve = (1 + returns.fillna(0)).cumprod()
        equity_curve.to_csv(output_dirs['series'] / f'{clean_key}_equity.csv')
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'total_strategies': len([k for k in strategy_results.keys() if 'n_cohorts' not in k]),
        'best_strategy': summary_df.loc[summary_df['sharpe'].idxmax(), 'Strategy_Key'] if not summary_df['sharpe'].isna().all() else 'N/A',
        'data_coverage': {
            'start_date': summary_df['Strategy_Key'].iloc[0] if len(summary_df) > 0 else 'N/A',
            'end_date': summary_df['Strategy_Key'].iloc[-1] if len(summary_df) > 0 else 'N/A'
        }
    }
    
    with open(output_dirs['summary'] / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Results saved to {output_dirs['results']}")

def main():
    """Main execution function."""
    print("🚀 COMMODITY CROSS-SECTIONAL MOMENTUM BACKTEST")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize cache manager
    cache_manager = DataCacheManager(".")
    
    # Handle cache-related commands
    if args.cache_status:
        cache_manager.print_cache_status()
        return 0
    
    if args.clear_cache:
        if args.clear_cache == 'all':
            cache_manager.clear_cache()
        elif args.clear_cache == 'data':
            cache_manager.clear_cache(data_only=True)
        elif args.clear_cache == 'strategy':
            cache_manager.clear_cache(strategy_only=True)
        return 0
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dirs = create_output_dirs(output_dir)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Parse parameter lists
    R_list = [int(x) for x in args.ranking_windows.split(',')]
    H_list = [int(x) for x in args.holding_windows.split(',')]
    
    # Determine strategy modes
    if args.strategy == 'both':
        modes = ['momentum', 'contrarian']
    else:
        modes = [args.strategy]
    
    print(f"📊 Configuration:")
    print(f"   Date range: {args.start} to {args.end}")
    print(f"   Ranking periods (R): {R_list}")
    print(f"   Holding periods (H): {H_list}")
    print(f"   Strategies: {modes}")
    print(f"   Transaction costs: {args.transaction_cost_bps} bps")
    print(f"   Use cache: {args.use_cache and not args.force_refresh}")
    
    # Show cache status
    if args.use_cache and not args.force_refresh:
        print(f"\n📋 Cache Status:")
        cache_manager.print_cache_status()
    
    try:
        # Step 1: Load ticker universe
        if args.universe == 'default':
            tickers = ALL_TICKERS
        else:
            tickers = load_custom_universe(args.universe)
        
        print(f"📈 Universe: {len(tickers)} commodities")
        
        # Step 2: Download price data (with cache support)
        use_cache = args.use_cache and not args.force_refresh
        
        if args.stitch_mode:
            print("⚠️  Contract stitching mode is experimental")
            # Implement stitching logic here if needed
            daily_prices = download_prices(tickers, args.start, args.end, 
                                         use_cache=use_cache, cache_manager=cache_manager)
        else:
            daily_prices = download_prices(tickers, args.start, args.end,
                                         use_cache=use_cache, cache_manager=cache_manager)
        
        if daily_prices.empty:
            raise ValueError("No price data downloaded")
        
        # Step 3: Convert to monthly and calculate returns (or load from cache)
        if use_cache:
            cached_data = cache_manager.load_price_data()
            if cached_data[0] is not None:
                _, monthly_prices, monthly_returns, _ = cached_data
                print("✅ Using cached monthly prices and returns")
            else:
                monthly_prices = to_monthly(daily_prices)
                monthly_returns = compute_monthly_returns(monthly_prices)
        else:
            monthly_prices = to_monthly(daily_prices)
            monthly_returns = compute_monthly_returns(monthly_prices)
        
        # Step 4: Download risk-free rate
        rf_rate = download_risk_free_rate(args.risk_free, args.start, args.end)
        
        # Step 5: Run grid backtest
        print(f"\n🔄 Starting grid backtest...")
        summary_results = grid_backtest(
            returns=monthly_returns,
            R_list=R_list,
            H_list=H_list,
            modes=modes,
            costs_bps=args.transaction_cost_bps,
            min_hist=args.min_history_months,
            n_quintiles=args.quintiles
        )
        
        # Step 6: Generate detailed strategy results for best combinations
        print("\n📊 Generating detailed results for key strategies...")
        detailed_results = {}
        
        # Select top strategies for detailed analysis
        top_strategies = summary_results.nlargest(10, 'sharpe')
        
        for _, row in top_strategies.iterrows():
            strategy_key = row['Strategy_Key']
            
            strategy_results = run_cs_strategy(
                monthly_returns, 
                row['R'], 
                row['H'], 
                mode=row['Strategy'].lower(),
                costs_bps=args.transaction_cost_bps,
                min_hist=args.min_history_months,
                n_quintiles=args.quintiles
            )
            
            detailed_results[strategy_key] = strategy_results[row['Portfolio_Type']]
        
        # Step 7: Generate visualizations
        print("\n📈 Creating visualizations...")
        
        # Heatmaps for each strategy type
        for mode in modes:
            for portfolio_type in ['long_short', 'long_only_q5']:
                plot_heatmap(
                    summary_results, 
                    metric='sharpe',
                    strategy=mode.title(), 
                    portfolio_type=portfolio_type,
                    save_path=output_dirs['figures']
                )
                
                plot_heatmap(
                    summary_results,
                    metric='t_stat', 
                    strategy=mode.title(),
                    portfolio_type=portfolio_type,
                    save_path=output_dirs['figures']
                )
        
        # Equity curves for top strategies
        if detailed_results:
            plot_equity_curves(detailed_results, output_dirs['figures'])
            
            # Distribution plots for best strategy
            best_strategy_key = summary_results.loc[summary_results['sharpe'].idxmax(), 'Strategy_Key']
            if best_strategy_key in detailed_results:
                plot_returns_distribution(
                    detailed_results[best_strategy_key],
                    title=f"Best Strategy: {best_strategy_key}",
                    save_path=output_dirs['figures'] / f'{best_strategy_key}_distribution.png'
                )
        
        # Step 8: Save results
        save_results(summary_results, detailed_results, output_dirs)
        
        # Step 9: Print summary
        print("\n🏆 TOP 10 STRATEGIES BY SHARPE RATIO")
        print("=" * 60)
        
        top_10 = summary_results.nlargest(10, 'sharpe')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:2d}. {row['Strategy_Key']}")
            print(f"     📊 Ann Return: {row['ann_return']:.2%}")
            print(f"     📈 Sharpe: {row['sharpe']:.3f}")
            print(f"     🧮 t-stat: {row['t_stat']:.2f}")
            print(f"     📉 Max DD: {row['max_dd']:.2%}")
            print()
        
        # Data coverage report  
        print("📋 DATA COVERAGE REPORT")
        print("-" * 30)
        print(f"📅 Analysis period: {monthly_returns.index.min().date()} to {monthly_returns.index.max().date()}")
        print(f"📊 Total months: {len(monthly_returns)}")
        print(f"🎯 Final universe: {len(monthly_returns.columns)} commodities")
        print(f"📈 Successful strategies: {len(summary_results[~summary_results['sharpe'].isna()])}")
        
        # Asset coverage
        print(f"\n📊 Asset coverage:")
        for ticker in monthly_returns.columns:
            non_null_pct = (1 - monthly_returns[ticker].isna().mean()) * 100
            print(f"   {ticker}: {non_null_pct:.1f}% coverage")
        
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("✅ BACKTEST COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ BACKTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)