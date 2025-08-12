import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from pathlib import Path

def contrarian_quintiles_strategy(monthly_prices: pd.DataFrame, lookback_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Traditional contrarian strategy using extreme quintiles only
    - Long bottom quintile (worst 20% performers)
    - Short top quintile (best 20% performers)
    - Neutral on middle 60%
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

def contrarian_full_spectrum_strategy(monthly_prices: pd.DataFrame, lookback_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full spectrum contrarian strategy
    - Long ALL losers (bottom 50% performers)
    - Short ALL winners (top 50% performers)
    - No neutral positions
    """
    
    # Calculate monthly returns
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    
    # Calculate lookback performance (avoid lookahead bias with shift)
    lookback_performance = monthly_returns.rolling(window=lookback_months).sum().shift(1)
    
    # Create percentile rankings for each month (0 to 1, where 0 = worst performer)
    percentile_ranks = lookback_performance.rank(axis=1, pct=True)
    
    # Generate positions vectorized
    positions = pd.DataFrame(0.0, index=percentile_ranks.index, columns=percentile_ranks.columns)
    
    # Long bottom half (all losers = contrarian long)
    long_mask = percentile_ranks < 0.5
    positions[long_mask] = 1.0
    
    # Short top half (all winners = contrarian short) 
    short_mask = percentile_ranks >= 0.5
    positions[short_mask] = -1.0
    
    # Equal weight within each half
    long_counts = long_mask.sum(axis=1)
    short_counts = short_mask.sum(axis=1)
    
    # Normalize weights so each half sums to 1 or -1
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

def calculate_performance_metrics(strategy_results: pd.DataFrame, strategy_name: str) -> Dict:
    """Calculate comprehensive performance metrics for a strategy"""
    
    clean_returns = strategy_results['strategy_returns'].dropna()
    
    if len(clean_returns) == 0:
        return {}
    
    # Basic metrics
    total_return = strategy_results['cumulative_returns'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (12 / len(clean_returns)) - 1
    annual_vol = clean_returns.std() * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Drawdown analysis
    cumulative = strategy_results['cumulative_returns'].dropna()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (clean_returns > 0).mean()
    
    # Additional metrics
    avg_win = clean_returns[clean_returns > 0].mean() if (clean_returns > 0).any() else 0
    avg_loss = clean_returns[clean_returns < 0].mean() if (clean_returns < 0).any() else 0
    profit_factor = abs(avg_win * (clean_returns > 0).sum()) / abs(avg_loss * (clean_returns < 0).sum()) if avg_loss != 0 else np.inf
    
    # Skewness and kurtosis
    skewness = clean_returns.skew()
    kurtosis = clean_returns.kurtosis()
    
    return {
        'strategy_name': strategy_name,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'months_data': len(clean_returns)
    }

def analyze_position_characteristics(positions_quintiles: pd.DataFrame, 
                                   positions_full: pd.DataFrame) -> Dict:
    """Analyze position characteristics for both strategies"""
    
    # Count positions for each strategy
    quintiles_long = (positions_quintiles > 0).sum(axis=1)
    quintiles_short = (positions_quintiles < 0).sum(axis=1)
    quintiles_total = quintiles_long + quintiles_short
    
    full_long = (positions_full > 0).sum(axis=1)
    full_short = (positions_full < 0).sum(axis=1)
    full_total = full_long + full_short
    
    # Concentration measures (Herfindahl index)
    def herfindahl_index(weights):
        """Calculate concentration index for position weights"""
        abs_weights = weights.abs()
        normalized = abs_weights.div(abs_weights.sum(axis=1), axis=0)
        return (normalized ** 2).sum(axis=1)
    
    quintiles_concentration = herfindahl_index(positions_quintiles)
    full_concentration = herfindahl_index(positions_full)
    
    return {
        'quintiles': {
            'avg_long_positions': quintiles_long.mean(),
            'avg_short_positions': quintiles_short.mean(),
            'avg_total_positions': quintiles_total.mean(),
            'avg_concentration': quintiles_concentration.mean(),
            'position_stability': quintiles_total.std()
        },
        'full_spectrum': {
            'avg_long_positions': full_long.mean(),
            'avg_short_positions': full_short.mean(),
            'avg_total_positions': full_total.mean(),
            'avg_concentration': full_concentration.mean(),
            'position_stability': full_total.std()
        }
    }

def run_strategy_comparison(monthly_prices: pd.DataFrame, 
                          lookback_months: int = 6) -> Tuple[Dict, Dict, Dict]:
    """
    Run comprehensive comparison between quintiles and full spectrum strategies
    
    Returns:
        - results_dict: Dictionary with strategy results
        - metrics_dict: Dictionary with performance metrics
        - positions_analysis: Dictionary with position analysis
    """
    
    print(f"Running strategy comparison with {lookback_months} months lookback...")
    
    # Run both strategies
    quintiles_results, quintiles_positions = contrarian_quintiles_strategy(monthly_prices, lookback_months)
    full_results, full_positions = contrarian_full_spectrum_strategy(monthly_prices, lookback_months)
    
    # Store results
    results_dict = {
        'quintiles': quintiles_results,
        'full_spectrum': full_results
    }
    
    # Calculate performance metrics
    quintiles_metrics = calculate_performance_metrics(quintiles_results, "Quintiles Strategy")
    full_metrics = calculate_performance_metrics(full_results, "Full Spectrum Strategy")
    
    metrics_dict = {
        'quintiles': quintiles_metrics,
        'full_spectrum': full_metrics
    }
    
    # Analyze positions
    positions_analysis = analyze_position_characteristics(quintiles_positions, full_positions)
    
    return results_dict, metrics_dict, positions_analysis

def create_comparison_visualizations(results_dict: Dict, 
                                   metrics_dict: Dict, 
                                   positions_analysis: Dict,
                                   lookback_months: int) -> None:
    """Create comprehensive comparison visualizations"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. Equity Curves Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Linear scale
    ax1.plot(results_dict['quintiles'].index, results_dict['quintiles']['cumulative_returns'], 
             linewidth=2.5, color='steelblue', label='Quintiles Strategy (20% extremes)', alpha=0.8)
    ax1.plot(results_dict['full_spectrum'].index, results_dict['full_spectrum']['cumulative_returns'], 
             linewidth=2.5, color='red', label='Full Spectrum Strategy (50/50 split)', alpha=0.8)
    
    ax1.set_title(f'Contrarian Strategies Comparison - Equity Curves\n({lookback_months} Months Lookback)', fontsize=14)
    ax1.set_ylabel('Cumulative Return (Linear Scale)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.plot(results_dict['quintiles'].index, results_dict['quintiles']['cumulative_returns'], 
             linewidth=2.5, color='steelblue', label='Quintiles Strategy (20% extremes)', alpha=0.8)
    ax2.plot(results_dict['full_spectrum'].index, results_dict['full_spectrum']['cumulative_returns'], 
             linewidth=2.5, color='red', label='Full Spectrum Strategy (50/50 split)', alpha=0.8)
    
    ax2.set_title('Equity Curves - Log Scale', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Performance Metrics Comparison ({lookback_months} Months Lookback)', fontsize=16)
    
    strategies = ['Quintiles', 'Full Spectrum']
    colors = ['steelblue', 'red']
    
    # Sharpe Ratio
    sharpe_values = [metrics_dict['quintiles']['sharpe_ratio'], metrics_dict['full_spectrum']['sharpe_ratio']]
    bars1 = axes[0,0].bar(strategies, sharpe_values, color=colors, alpha=0.8)
    axes[0,0].set_title('Sharpe Ratio', fontsize=12)
    axes[0,0].set_ylabel('Sharpe Ratio')
    axes[0,0].grid(True, alpha=0.3)
    for bar, value in zip(bars1, sharpe_values):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Annual Return
    annual_returns = [metrics_dict['quintiles']['annual_return']*100, metrics_dict['full_spectrum']['annual_return']*100]
    bars2 = axes[0,1].bar(strategies, annual_returns, color=colors, alpha=0.8)
    axes[0,1].set_title('Annual Return', fontsize=12)
    axes[0,1].set_ylabel('Annual Return (%)')
    axes[0,1].grid(True, alpha=0.3)
    for bar, value in zip(bars2, annual_returns):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                      f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Max Drawdown
    max_dd = [metrics_dict['quintiles']['max_drawdown']*100, metrics_dict['full_spectrum']['max_drawdown']*100]
    bars3 = axes[0,2].bar(strategies, max_dd, color=colors, alpha=0.8)
    axes[0,2].set_title('Maximum Drawdown', fontsize=12)
    axes[0,2].set_ylabel('Max Drawdown (%)')
    axes[0,2].grid(True, alpha=0.3)
    for bar, value in zip(bars3, max_dd):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
                      f'{value:.1f}%', ha='center', va='top', fontsize=10)
    
    # Win Rate
    win_rates = [metrics_dict['quintiles']['win_rate']*100, metrics_dict['full_spectrum']['win_rate']*100]
    bars4 = axes[1,0].bar(strategies, win_rates, color=colors, alpha=0.8)
    axes[1,0].set_title('Win Rate', fontsize=12)
    axes[1,0].set_ylabel('Win Rate (%)')
    axes[1,0].grid(True, alpha=0.3)
    for bar, value in zip(bars4, win_rates):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                      f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Average Positions
    avg_positions = [positions_analysis['quintiles']['avg_total_positions'], 
                    positions_analysis['full_spectrum']['avg_total_positions']]
    bars5 = axes[1,1].bar(strategies, avg_positions, color=colors, alpha=0.8)
    axes[1,1].set_title('Average Total Positions', fontsize=12)
    axes[1,1].set_ylabel('Number of Positions')
    axes[1,1].grid(True, alpha=0.3)
    for bar, value in zip(bars5, avg_positions):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Concentration Index
    concentration = [positions_analysis['quintiles']['avg_concentration'], 
                    positions_analysis['full_spectrum']['avg_concentration']]
    bars6 = axes[1,2].bar(strategies, concentration, color=colors, alpha=0.8)
    axes[1,2].set_title('Portfolio Concentration', fontsize=12)
    axes[1,2].set_ylabel('Herfindahl Index')
    axes[1,2].grid(True, alpha=0.3)
    for bar, value in zip(bars6, concentration):
        axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def print_detailed_comparison_table(metrics_dict: Dict, positions_analysis: Dict) -> None:
    """Print detailed comparison table"""
    
    print(f"\n{'='*100}")
    print(f"DETAILED STRATEGY COMPARISON")
    print(f"{'='*100}")
    
    print(f"{'Metric':<25} {'Quintiles':<20} {'Full Spectrum':<20} {'Difference':<15} {'Winner':<10}")
    print("-" * 100)
    
    # Performance metrics comparison
    metrics = [
        ('Total Return', 'total_return', '%'),
        ('Annual Return', 'annual_return', '%'),
        ('Annual Volatility', 'annual_volatility', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Win Rate', 'win_rate', '%'),
        ('Profit Factor', 'profit_factor', ''),
        ('Skewness', 'skewness', ''),
        ('Kurtosis', 'kurtosis', '')
    ]
    
    for metric_name, metric_key, unit in metrics:
        q_val = metrics_dict['quintiles'][metric_key]
        f_val = metrics_dict['full_spectrum'][metric_key]
        diff = f_val - q_val
        
        if unit == '%':
            q_str = f"{q_val:.1%}"
            f_str = f"{f_val:.1%}"
            diff_str = f"{diff:.1%}"
        else:
            q_str = f"{q_val:.3f}"
            f_str = f"{f_val:.3f}"
            diff_str = f"{diff:.3f}"
        
        # Determine winner (higher is better except for volatility, max drawdown)
        if metric_key in ['annual_volatility', 'max_drawdown']:
            winner = "Quintiles" if q_val < f_val else "Full Spectrum"
        else:
            winner = "Quintiles" if q_val > f_val else "Full Spectrum"
            
        print(f"{metric_name:<25} {q_str:<20} {f_str:<20} {diff_str:<15} {winner:<10}")
    
    print("\n" + "="*100)
    print("POSITION ANALYSIS")
    print("="*100)
    
    print(f"{'Metric':<25} {'Quintiles':<20} {'Full Spectrum':<20}")
    print("-" * 70)
    
    pos_metrics = [
        ('Avg Long Positions', 'avg_long_positions'),
        ('Avg Short Positions', 'avg_short_positions'),
        ('Avg Total Positions', 'avg_total_positions'),
        ('Portfolio Concentration', 'avg_concentration'),
        ('Position Stability', 'position_stability')
    ]
    
    for metric_name, metric_key in pos_metrics:
        q_val = positions_analysis['quintiles'][metric_key]
        f_val = positions_analysis['full_spectrum'][metric_key]
        
        print(f"{metric_name:<25} {q_val:<20.2f} {f_val:<20.2f}")