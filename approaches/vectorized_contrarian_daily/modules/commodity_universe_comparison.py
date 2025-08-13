import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

def calculate_performance_metrics(strategy_results: pd.DataFrame, strategy_name: str) -> Dict:
    """
    Calculate comprehensive performance metrics for a strategy
    
    Parameters:
    - strategy_results: DataFrame with strategy returns and cumulative returns
    - strategy_name: Name of the strategy for identification
    
    Returns:
    - metrics_dict: Dictionary with calculated metrics
    """
    
    clean_returns = strategy_results['strategy_returns'].dropna()
    
    if len(clean_returns) == 0:
        print(f"WARNING: No valid returns data for {strategy_name}")
        return {'strategy_name': strategy_name}
    
    # Basic metrics
    total_return = strategy_results['cumulative_returns'].iloc[-1] - 1
    
    # Annualized metrics (252 trading days per year)
    annual_return = (1 + total_return) ** (252 / len(clean_returns)) - 1
    annual_vol = clean_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Drawdown analysis
    cumulative = strategy_results['cumulative_returns'].dropna()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Current drawdown
    current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
    
    # Win rate and profit metrics
    win_rate = (clean_returns > 0).mean()
    loss_rate = (clean_returns < 0).mean()
    
    # Average win/loss
    winning_returns = clean_returns[clean_returns > 0]
    losing_returns = clean_returns[clean_returns < 0]
    
    avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
    avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
    
    # Profit factor
    total_wins = winning_returns.sum() if len(winning_returns) > 0 else 0
    total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
    
    # Additional statistical metrics
    skewness = clean_returns.skew()
    kurtosis = clean_returns.kurtosis()
    
    # Calmar ratio (annual return / max drawdown)
    calmar = abs(annual_return / max_drawdown) if max_drawdown < 0 else 0
    
    # VaR and CVaR (95% confidence)
    var_95 = clean_returns.quantile(0.05)
    cvar_95 = clean_returns[clean_returns <= var_95].mean() if len(clean_returns[clean_returns <= var_95]) > 0 else 0
    
    return {
        'strategy_name': strategy_name,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'current_drawdown': current_drawdown,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'trading_days': len(clean_returns)
    }

def analyze_position_characteristics(positions_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Analyze position characteristics for both strategies
    
    Parameters:
    - positions_dict: Dictionary with strategy positions
    
    Returns:
    - analysis_dict: Dictionary with position analysis metrics
    """
    
    analysis = {}
    
    for strategy_name, positions in positions_dict.items():
        
        # Position counts over time
        active_positions = (positions > 0).sum(axis=1)
        
        # Position concentration (Herfindahl index)
        position_weights = positions.abs()
        row_sums = position_weights.sum(axis=1)
        
        # Avoid division by zero
        normalized_weights = position_weights.div(row_sums, axis=0)
        normalized_weights = normalized_weights.fillna(0)
        
        # Herfindahl index (concentration measure)
        herfindahl = (normalized_weights ** 2).sum(axis=1)
        
        # Turnover analysis (how often positions change)
        position_changes = positions.diff().abs().sum(axis=1)
        
        analysis[strategy_name] = {
            'avg_active_positions': active_positions.mean(),
            'max_active_positions': active_positions.max(),
            'min_active_positions': active_positions.min(),
            'position_stability': active_positions.std(),
            'avg_concentration': herfindahl.mean(),
            'max_concentration': herfindahl.max(),
            'avg_daily_turnover': position_changes.mean(),
            'max_daily_turnover': position_changes.max()
        }
    
    return analysis

def calculate_strategy_correlations(results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate correlations between strategy returns
    
    Parameters:
    - results_dict: Dictionary with strategy results
    
    Returns:
    - correlation_matrix: DataFrame with strategy return correlations
    """
    
    returns_df = pd.DataFrame()
    for strategy_name, results in results_dict.items():
        returns_df[strategy_name] = results['strategy_returns']
    
    return returns_df.corr()

def create_comparison_visualizations(results_dict: Dict[str, pd.DataFrame], 
                                   metrics_dict: Dict[str, Dict], 
                                   positions_analysis: Dict,
                                   lookback_days: int = 20) -> None:
    """
    Create comprehensive comparison visualizations
    
    Parameters:
    - results_dict: Dictionary with strategy results
    - metrics_dict: Dictionary with performance metrics
    - positions_analysis: Dictionary with position analysis
    - lookback_days: Lookback period used in strategies
    """
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. Equity Curves Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Linear scale equity curves
    colors = ['steelblue', 'red']
    strategy_names = list(results_dict.keys())
    
    for i, (strategy_name, results) in enumerate(results_dict.items()):
        ax1.plot(results.index, results['cumulative_returns'], 
                linewidth=2.5, color=colors[i], 
                label=f'{strategy_name.replace("_", " ").title()}', alpha=0.8)
    
    ax1.set_title(f'Daily Contrarian Strategies - Equity Curves Comparison\n({lookback_days} Days Lookback)', 
                 fontsize=14)
    ax1.set_ylabel('Cumulative Return (Linear Scale)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Log scale equity curves
    for i, (strategy_name, results) in enumerate(results_dict.items()):
        ax2.plot(results.index, results['cumulative_returns'], 
                linewidth=2.5, color=colors[i], 
                label=f'{strategy_name.replace("_", " ").title()}', alpha=0.8)
    
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
    fig.suptitle(f'Performance Metrics Comparison ({lookback_days} Days Lookback)', fontsize=16)
    
    strategies = [name.replace('_', ' ').title() for name in strategy_names]
    
    # Sharpe Ratio
    sharpe_values = [metrics_dict[name]['sharpe_ratio'] for name in strategy_names]
    bars1 = axes[0,0].bar(strategies, sharpe_values, color=colors, alpha=0.8)
    axes[0,0].set_title('Sharpe Ratio', fontsize=12)
    axes[0,0].set_ylabel('Sharpe Ratio')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    for bar, value in zip(bars1, sharpe_values):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Annual Return
    annual_returns = [metrics_dict[name]['annual_return']*100 for name in strategy_names]
    bars2 = axes[0,1].bar(strategies, annual_returns, color=colors, alpha=0.8)
    axes[0,1].set_title('Annual Return', fontsize=12)
    axes[0,1].set_ylabel('Annual Return (%)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    for bar, value in zip(bars2, annual_returns):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Max Drawdown
    max_dd = [metrics_dict[name]['max_drawdown']*100 for name in strategy_names]
    bars3 = axes[0,2].bar(strategies, max_dd, color=colors, alpha=0.8)
    axes[0,2].set_title('Maximum Drawdown', fontsize=12)
    axes[0,2].set_ylabel('Max Drawdown (%)')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].tick_params(axis='x', rotation=45)
    for bar, value in zip(bars3, max_dd):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
                      f'{value:.1f}%', ha='center', va='top', fontsize=10)
    
    # Win Rate
    win_rates = [metrics_dict[name]['win_rate']*100 for name in strategy_names]
    bars4 = axes[1,0].bar(strategies, win_rates, color=colors, alpha=0.8)
    axes[1,0].set_title('Win Rate', fontsize=12)
    axes[1,0].set_ylabel('Win Rate (%)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    for bar, value in zip(bars4, win_rates):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Average Active Positions
    avg_positions = [positions_analysis[name]['avg_active_positions'] for name in strategy_names]
    bars5 = axes[1,1].bar(strategies, avg_positions, color=colors, alpha=0.8)
    axes[1,1].set_title('Average Active Positions', fontsize=12)
    axes[1,1].set_ylabel('Number of Positions')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    for bar, value in zip(bars5, avg_positions):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Calmar Ratio
    calmar_ratios = [metrics_dict[name]['calmar_ratio'] for name in strategy_names]
    bars6 = axes[1,2].bar(strategies, calmar_ratios, color=colors, alpha=0.8)
    axes[1,2].set_title('Calmar Ratio', fontsize=12)
    axes[1,2].set_ylabel('Calmar Ratio')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].tick_params(axis='x', rotation=45)
    for bar, value in zip(bars6, calmar_ratios):
        axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Rolling Performance Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Rolling Sharpe Ratio (252-day window)
    window = min(252, len(results_dict[strategy_names[0]]) // 4)
    
    for i, (strategy_name, results) in enumerate(results_dict.items()):
        rolling_returns = results['strategy_returns']
        rolling_sharpe = (rolling_returns.rolling(window).mean() / rolling_returns.rolling(window).std() * np.sqrt(252))
        ax1.plot(rolling_sharpe.index, rolling_sharpe, 
                color=colors[i], linewidth=2, 
                label=f'{strategy_name.replace("_", " ").title()}', alpha=0.8)
    
    ax1.set_title(f'Rolling Sharpe Ratio ({window}-Day Window)', fontsize=14)
    ax1.set_ylabel('Rolling Sharpe Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rolling Drawdown
    for i, (strategy_name, results) in enumerate(results_dict.items()):
        cumulative = results['cumulative_returns']
        running_max = cumulative.rolling(window).max()
        rolling_dd = (cumulative - running_max) / running_max * 100
        ax2.fill_between(rolling_dd.index, rolling_dd, 0, 
                        color=colors[i], alpha=0.3, 
                        label=f'{strategy_name.replace("_", " ").title()}')
    
    ax2.set_title(f'Rolling Drawdown ({window}-Day Max)', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_detailed_comparison_table(metrics_dict: Dict[str, Dict], positions_analysis: Dict) -> None:
    """
    Print detailed comparison table with all metrics
    
    Parameters:
    - metrics_dict: Dictionary with performance metrics for each strategy
    - positions_analysis: Dictionary with position analysis metrics
    """
    
    strategy_names = list(metrics_dict.keys())
    
    print(f"\n{'='*120}")
    print(f"DETAILED COMMODITY STRATEGIES COMPARISON")
    print(f"{'='*120}")
    
    print(f"{'Metric':<25} {'Quintiles':<20} {'Full Universe':<20} {'Difference':<15} {'Winner':<15}")
    print("-" * 120)
    
    # Performance metrics comparison
    metrics_to_compare = [
        ('Total Return', 'total_return', '%'),
        ('Annual Return', 'annual_return', '%'),
        ('Annual Volatility', 'annual_volatility', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Calmar Ratio', 'calmar_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Current Drawdown', 'current_drawdown', '%'),
        ('Win Rate', 'win_rate', '%'),
        ('Profit Factor', 'profit_factor', ''),
        ('Skewness', 'skewness', ''),
        ('Kurtosis', 'kurtosis', ''),
        ('VaR (95%)', 'var_95', '%'),
        ('CVaR (95%)', 'cvar_95', '%')
    ]
    
    for metric_name, metric_key, unit in metrics_to_compare:
        if metric_key in metrics_dict[strategy_names[0]] and metric_key in metrics_dict[strategy_names[1]]:
            val1 = metrics_dict[strategy_names[0]][metric_key]
            val2 = metrics_dict[strategy_names[1]][metric_key]
            diff = val2 - val1
            
            if unit == '%':
                val1_str = f"{val1:.1%}" if not np.isinf(val1) else "Inf"
                val2_str = f"{val2:.1%}" if not np.isinf(val2) else "Inf" 
                diff_str = f"{diff:.1%}" if not np.isinf(diff) else "Inf"
            else:
                val1_str = f"{val1:.3f}" if not np.isinf(val1) else "Inf"
                val2_str = f"{val2:.3f}" if not np.isinf(val2) else "Inf"
                diff_str = f"{diff:.3f}" if not np.isinf(diff) else "Inf"
            
            # Determine winner (higher is better except for volatility, drawdown, VaR, CVaR)
            if metric_key in ['annual_volatility', 'max_drawdown', 'current_drawdown', 'var_95', 'cvar_95']:
                winner = strategy_names[0].title() if val1 > val2 else strategy_names[1].title()
            else:
                winner = strategy_names[0].title() if val1 > val2 else strategy_names[1].title()
                
            print(f"{metric_name:<25} {val1_str:<20} {val2_str:<20} {diff_str:<15} {winner:<15}")
    
    print("\n" + "="*120)
    print("POSITION ANALYSIS")
    print("="*120)
    
    print(f"{'Metric':<25} {'Quintiles':<20} {'Full Universe':<20}")
    print("-" * 70)
    
    pos_metrics = [
        ('Avg Active Positions', 'avg_active_positions'),
        ('Max Active Positions', 'max_active_positions'),
        ('Position Stability', 'position_stability'),
        ('Avg Concentration', 'avg_concentration'),
        ('Avg Daily Turnover', 'avg_daily_turnover')
    ]
    
    for metric_name, metric_key in pos_metrics:
        val1 = positions_analysis[strategy_names[0]][metric_key]
        val2 = positions_analysis[strategy_names[1]][metric_key]
        
        print(f"{metric_name:<25} {val1:<20.2f} {val2:<20.2f}")
    
    print(f"\n{'='*120}")

def generate_summary_report(results_dict: Dict[str, pd.DataFrame], 
                          metrics_dict: Dict[str, Dict],
                          positions_analysis: Dict,
                          lookback_days: int) -> str:
    """
    Generate a summary report of the comparison
    
    Parameters:
    - results_dict: Dictionary with strategy results
    - metrics_dict: Dictionary with performance metrics
    - positions_analysis: Dictionary with position analysis
    - lookback_days: Lookback period used
    
    Returns:
    - summary_text: String with summary report
    """
    
    strategy_names = list(results_dict.keys())
    
    # Determine better performing strategy based on Sharpe ratio
    best_strategy = strategy_names[0] if metrics_dict[strategy_names[0]]['sharpe_ratio'] > metrics_dict[strategy_names[1]]['sharpe_ratio'] else strategy_names[1]
    
    summary = f"""
COMMODITY CONTRARIAN STRATEGIES COMPARISON SUMMARY
==================================================

Analysis Period: {results_dict[strategy_names[0]].index[0].date()} to {results_dict[strategy_names[0]].index[-1].date()}
Lookback Period: {lookback_days} days
Best Performer (by Sharpe): {best_strategy.replace('_', ' ').title()}

KEY FINDINGS:
------------

1. PERFORMANCE OVERVIEW:
   • Quintiles Strategy: {metrics_dict['quintiles']['annual_return']:.1%} annual return, {metrics_dict['quintiles']['sharpe_ratio']:.3f} Sharpe
   • Full Universe Strategy: {metrics_dict['full_universe']['annual_return']:.1%} annual return, {metrics_dict['full_universe']['sharpe_ratio']:.3f} Sharpe
   • Performance Difference: {(metrics_dict['full_universe']['annual_return'] - metrics_dict['quintiles']['annual_return']):.1%}

2. RISK METRICS:
   • Max Drawdown - Quintiles: {metrics_dict['quintiles']['max_drawdown']:.1%}
   • Max Drawdown - Full Universe: {metrics_dict['full_universe']['max_drawdown']:.1%}
   • Volatility Difference: {(metrics_dict['full_universe']['annual_volatility'] - metrics_dict['quintiles']['annual_volatility']):.1%}

3. POSITION CHARACTERISTICS:
   • Quintiles Avg Positions: {positions_analysis['quintiles']['avg_active_positions']:.1f}
   • Full Universe Avg Positions: {positions_analysis['full_universe']['avg_active_positions']:.1f}
   • Concentration Difference: {(positions_analysis['full_universe']['avg_concentration'] - positions_analysis['quintiles']['avg_concentration']):.3f}

RECOMMENDATION:
--------------
Based on the analysis, the {best_strategy.replace('_', ' ').title()} strategy shows superior risk-adjusted returns.
    """
    
    return summary