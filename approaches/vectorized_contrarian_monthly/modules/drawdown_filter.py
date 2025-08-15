import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


def calculate_vectorized_drawdown(equity_curve: pd.Series, 
                                threshold: float = -0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized calculation of drawdown filter with no lookahead bias
    
    Args:
        equity_curve: Series of cumulative returns (equity curve)
        threshold: Drawdown threshold for exiting positions (default: -5%)
        
    Returns:
        Tuple of (drawdown, investment_state, previous_max) arrays
    """
    equity = equity_curve.astype(float).values
    eps = 1e-12
    
    # Calculate maximum PREVIOUS value using cummax() shifted to avoid lookahead bias
    roll_prev = pd.Series(equity).cummax().shift(1).fillna(equity[0]).values
    roll_prev = np.where(roll_prev == 0, eps, roll_prev)
    
    # Calculate drawdown relative to previous maximum (vectorized)
    drawdown = (equity - roll_prev) / roll_prev
    
    # Apply threshold: exit when drawdown <= threshold
    state_at_t = (drawdown > threshold).astype(int)
    
    # Investment state for next period (no lookahead)
    state = np.empty_like(state_at_t)
    state[0] = 1  # Always invested on first day
    state[1:] = state_at_t[:-1]
    
    return drawdown, state, roll_prev


def apply_drawdown_filter(equity_data: pd.DataFrame,
                         threshold: float = -0.05,
                         equity_column: str = 'cumulative_returns') -> pd.DataFrame:
    """
    Apply drawdown filter to equity curve data
    
    Args:
        equity_data: DataFrame containing equity curve
        threshold: Drawdown threshold for filter (default: -5%)
        equity_column: Name of column containing cumulative returns
        
    Returns:
        DataFrame with filter applied and additional analysis columns
    """
    # Copy input data
    result_df = equity_data.copy()
    
    # Apply vectorized drawdown calculation
    drawdown, state, roll_prev = calculate_vectorized_drawdown(
        result_df[equity_column], threshold
    )
    
    # Add filter results to dataframe
    result_df['drawdown'] = drawdown
    result_df['invested'] = state
    result_df['previous_max'] = roll_prev
    
    # Calculate filtered equity curve
    filtered_curve = result_df[equity_column].copy()
    filtered_curve[result_df['invested'] == 0] = np.nan
    filtered_curve = filtered_curve.ffill()
    result_df['filtered_cumulative_returns'] = filtered_curve
    
    # Calculate filter statistics
    exit_points = result_df.index[(result_df['invested'].shift(1) == 1) & (result_df['invested'] == 0)].tolist()
    entry_points = result_df.index[(result_df['invested'].shift(1) == 0) & (result_df['invested'] == 1)].tolist()
    
    filter_stats = {
        'threshold': threshold,
        'exit_points': len(exit_points),
        'entry_points': len(entry_points),
        'time_invested_pct': result_df['invested'].mean() * 100,
        'exit_dates': exit_points,
        'entry_dates': entry_points
    }
    
    # Verify no lookahead bias
    mismatch = verify_no_lookahead_bias(equity_data[equity_column], threshold)
    filter_stats['lookahead_bias_detected'] = len(mismatch) > 0
    filter_stats['mismatch_count'] = len(mismatch)
    
    return result_df, filter_stats


def verify_no_lookahead_bias(equity_curve: pd.Series, threshold: float = -0.05) -> np.ndarray:
    """
    Verify that no lookahead bias exists in the filter implementation
    
    Args:
        equity_curve: Series of cumulative returns
        threshold: Drawdown threshold used in filter
        
    Returns:
        Array of indices where mismatches are detected (empty if no bias)
    """
    equity = equity_curve.astype(float).values
    eps = 1e-12
    
    # Verification calculation (should match main calculation)
    roll_prev_check = pd.Series(equity).cummax().shift(1).fillna(equity[0]).values
    roll_prev_check = np.where(roll_prev_check == 0, eps, roll_prev_check)
    dd_check = (equity - roll_prev_check) / roll_prev_check
    state_check_at_t = (dd_check > threshold).astype(int)
    state_check = np.empty_like(state_check_at_t)
    state_check[0] = 1
    state_check[1:] = state_check_at_t[:-1]
    
    # Compare with original calculation
    _, state_original, _ = calculate_vectorized_drawdown(equity_curve, threshold)
    
    mismatch = np.where(state_original != state_check)[0]
    return mismatch


def analyze_filter_performance(original_equity: pd.Series,
                              filtered_equity: pd.Series,
                              periods_per_year: int = 252) -> Dict[str, float]:
    """
    Comprehensive performance analysis of filtered vs original strategy
    
    Args:
        original_equity: Original equity curve
        filtered_equity: Filtered equity curve
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Dictionary with performance metrics comparison
    """
    def calculate_metrics(series: pd.Series) -> Dict[str, float]:
        """Calculate standard performance metrics for a series"""
        series = series.astype(float).dropna()
        returns = series.pct_change().dropna()
        years = len(returns) / periods_per_year
        
        # Core metrics
        total_return = (series.iloc[-1] / series.iloc[0]) - 1 if len(series) > 0 else np.nan
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
        annual_vol = returns.std() * np.sqrt(periods_per_year) if len(returns) > 0 else np.nan
        sharpe = (returns.mean() * periods_per_year) / annual_vol if annual_vol != 0 else np.nan
        
        # Drawdown calculation
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'years': years
        }
    
    # Calculate metrics for both strategies
    original_metrics = calculate_metrics(original_equity)
    filtered_metrics = calculate_metrics(filtered_equity)
    
    # Create comparison dictionary
    comparison = {}
    for metric in original_metrics.keys():
        comparison[f'original_{metric}'] = original_metrics[metric]
        comparison[f'filtered_{metric}'] = filtered_metrics[metric]
        
        # Calculate improvement (filtered vs original)
        if metric in ['total_return', 'cagr', 'sharpe_ratio']:
            # Higher is better
            improvement = filtered_metrics[metric] - original_metrics[metric]
        elif metric in ['annual_volatility', 'max_drawdown']:
            # Lower is better (for max_drawdown, less negative is better)
            improvement = original_metrics[metric] - filtered_metrics[metric]
        else:
            improvement = np.nan
            
        comparison[f'{metric}_improvement'] = improvement
    
    return comparison


def plot_filter_comparison(equity_data: pd.DataFrame,
                         filter_stats: Dict,
                         performance_metrics: Dict,
                         figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Create comprehensive visualization of drawdown filter analysis
    
    Args:
        equity_data: DataFrame with original and filtered equity curves
        filter_stats: Filter statistics from apply_drawdown_filter
        performance_metrics: Performance comparison from analyze_filter_performance
        figsize: Figure size for plots
    """
    # Prepare data for visualization
    if not np.issubdtype(equity_data.index.dtype, np.datetime64):
        try:
            equity_data.index = pd.to_datetime(equity_data.index)
        except:
            print("Warning: Could not convert index to datetime")
    
    # Optimize plotting for large datasets
    step = max(1, len(equity_data) // 2000)
    plot_data = equity_data.iloc[::step].copy()
    
    # Calculate drawdown of FILTERED strategy for the drawdown plot
    filtered_series = plot_data['filtered_cumulative_returns'].dropna()
    filtered_rolling_max = filtered_series.cummax()
    filtered_drawdown = (filtered_series - filtered_rolling_max) / filtered_rolling_max
    
    # Create subplot layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
    
    # Main equity curve comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(plot_data.index, plot_data['cumulative_returns'], 
             label='Original Strategy', alpha=0.7, color='steelblue', linewidth=1.5)
    ax1.plot(plot_data.index, plot_data['filtered_cumulative_returns'], 
             label=f'Drawdown Filtered ({filter_stats["threshold"]:.1%})', 
             linewidth=2, color='darkorange')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns')
    ax1.set_title('Strategy Performance: Original vs Drawdown Filtered')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Drawdown plot - NOW SHOWING FILTERED STRATEGY DRAWDOWN
    ax2 = fig.add_subplot(gs[1, :])
    # Plot original strategy drawdown for reference (lighter)
    ax2.plot(plot_data.index, plot_data['drawdown'], color='lightblue', alpha=0.5, 
             linewidth=1, label='Original Strategy DD')
    # Plot filtered strategy drawdown (main focus)
    ax2.plot(filtered_drawdown.index, filtered_drawdown, color='red', alpha=0.8, linewidth=2)
    ax2.fill_between(filtered_drawdown.index, filtered_drawdown, 0, color='red', alpha=0.3)
    ax2.axhline(filter_stats['threshold'], color='black', linestyle='--', 
                label=f'Filter Threshold ({filter_stats["threshold"]:.1%})')
    ax2.set_title('Drawdown Analysis (Filtered Strategy Focus)')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add text annotation showing max DD of filtered strategy
    max_filtered_dd = filtered_drawdown.min()
    ax2.text(0.02, 0.95, f'Filtered Strategy Max DD: {max_filtered_dd:.1%}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             verticalalignment='top', fontweight='bold')
    
    # Investment state
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(plot_data.index, plot_data['invested'], color='green', linewidth=1)
    ax3.fill_between(plot_data.index, plot_data['invested'], 0, 
                     color='green', alpha=0.3)
    ax3.set_title(f'Investment State\n({filter_stats["time_invested_pct"]:.1f}% invested)')
    ax3.set_ylabel('Invested (1/0)')
    ax3.grid(True, alpha=0.3)
    
    # Performance metrics comparison
    ax4 = fig.add_subplot(gs[2, 1])
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
    original_vals = [performance_metrics[f'original_{m}'] for m in metrics]
    filtered_vals = [performance_metrics[f'filtered_{m}'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, original_vals, width, label='Original', alpha=0.7)
    bars2 = ax4.bar(x + width/2, filtered_vals, width, label='Filtered', alpha=0.7)
    
    ax4.set_title('Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Total Ret', 'Sharpe', 'Max DD'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Filter statistics summary
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    stats_text = f"""Filter Statistics:
Threshold: {filter_stats['threshold']:.1%}
Exit Events: {filter_stats['exit_points']}
Entry Events: {filter_stats['entry_points']}
Time Invested: {filter_stats['time_invested_pct']:.1f}%

Performance Improvement:
CAGR: {performance_metrics['cagr_improvement']:.2%}
Sharpe: {performance_metrics['sharpe_ratio_improvement']:.3f}
Max DD: {performance_metrics['max_drawdown_improvement']:.2%}

Bias Check: {'✓ Pass' if not filter_stats['lookahead_bias_detected'] else '⚠ Fail'}"""
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()


def calculate_rolling_performance_metrics(equity_data: pd.DataFrame,
                                       window_days: int = 252,
                                       step_days: int = 63) -> pd.DataFrame:
    """
    Calculate rolling performance metrics for filtered vs original strategy
    
    Args:
        equity_data: DataFrame with original and filtered equity curves
        window_days: Rolling window size in days (default: 1 year)
        step_days: Step size for rolling calculation (default: quarterly)
        
    Returns:
        DataFrame with rolling performance metrics
    """
    # Ensure we have the required columns
    required_cols = ['cumulative_returns', 'filtered_cumulative_returns']
    if not all(col in equity_data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Create rolling windows
    rolling_metrics = []
    
    for i in range(window_days, len(equity_data), step_days):
        start_idx = i - window_days
        end_idx = i
        
        window_data = equity_data.iloc[start_idx:end_idx]
        window_date = equity_data.index[end_idx-1]
        
        # Calculate metrics for both strategies in this window
        original_window = window_data['cumulative_returns']
        filtered_window = window_data['filtered_cumulative_returns']
        
        # Calculate returns
        orig_ret = (original_window.iloc[-1] / original_window.iloc[0]) - 1
        filt_ret = (filtered_window.iloc[-1] / filtered_window.iloc[0]) - 1
        
        # Calculate volatility
        orig_vol = original_window.pct_change().std() * np.sqrt(252)
        filt_vol = filtered_window.pct_change().std() * np.sqrt(252)
        
        # Calculate Sharpe ratios
        orig_sharpe = (orig_ret * 252/window_days) / orig_vol if orig_vol > 0 else 0
        filt_sharpe = (filt_ret * 252/window_days) / filt_vol if filt_vol > 0 else 0
        
        rolling_metrics.append({
            'date': window_date,
            'original_return': orig_ret,
            'filtered_return': filt_ret,
            'original_volatility': orig_vol,
            'filtered_volatility': filt_vol,
            'original_sharpe': orig_sharpe,
            'filtered_sharpe': filt_sharpe,
            'return_improvement': filt_ret - orig_ret,
            'sharpe_improvement': filt_sharpe - orig_sharpe
        })
    
    return pd.DataFrame(rolling_metrics).set_index('date')


if __name__ == "__main__":
    # Example usage and testing
    print("Testing drawdown filter module...")
    
    # Create sample equity curve data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    returns = np.random.normal(0.0005, 0.02, 1000)  # Daily returns
    equity_curve = pd.Series((1 + returns).cumprod(), index=dates)
    
    # Create test DataFrame
    test_data = pd.DataFrame({
        'strategy_returns': returns,
        'cumulative_returns': equity_curve
    })
    
    print("Applying drawdown filter...")
    filtered_data, stats = apply_drawdown_filter(test_data, threshold=-0.05)
    
    print("Analyzing performance...")
    performance = analyze_filter_performance(
        test_data['cumulative_returns'], 
        filtered_data['filtered_cumulative_returns']
    )
    
    print("\nFilter Statistics:")
    for key, value in stats.items():
        if key not in ['exit_dates', 'entry_dates']:
            print(f"{key}: {value}")
    
    print("\nPerformance Comparison:")
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    
    print("\nDrawdown filter module test completed successfully!")