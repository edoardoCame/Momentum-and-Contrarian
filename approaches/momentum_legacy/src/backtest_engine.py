"""
Backtesting engine for momentum and contrarian strategies.
Provides a unified framework for testing different signal generation methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestResults:
    """Container for backtest results."""
    strategy_name: str
    strategy_returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    positions: pd.DataFrame
    metrics: Dict[str, float]
    
class BacktestEngine:
    """
    Unified backtesting engine for commodity momentum strategies.
    """
    
    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize backtest engine.
        
        Args:
            returns_data: DataFrame of commodity daily returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        
    def run_backtest(self, 
                    signals: pd.DataFrame,
                    strategy_name: str = "Unknown Strategy",
                    portfolio_type: str = "long_short",
                    rebalance_freq: str = "daily") -> BacktestResults:
        """
        Run backtest for given signals.
        
        Args:
            signals: DataFrame of signals (-1, 0, 1)
            strategy_name: Name of the strategy
            portfolio_type: "long_only", "long_short", or "dollar_neutral"
            rebalance_freq: "daily", "weekly", "monthly"
        
        Returns:
            BacktestResults object
        """
        print(f"🚀 Running backtest for: {strategy_name}")
        print(f"📊 Portfolio type: {portfolio_type}")
        
        # Align returns and signals
        common_index = self.returns_data.index.intersection(signals.index)
        common_columns = self.returns_data.columns.intersection(signals.columns)
        
        returns_aligned = self.returns_data.loc[common_index, common_columns]
        signals_aligned = signals.loc[common_index, common_columns]
        
        # Calculate weights based on signals and portfolio type
        weights = self._calculate_weights(signals_aligned, portfolio_type)
        
        # Apply rebalancing frequency
        if rebalance_freq != "daily":
            weights = self._apply_rebalancing(weights, rebalance_freq)
        
        # Calculate strategy returns
        strategy_returns = (weights * returns_aligned).sum(axis=1)
        
        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod()
        
        # Calculate positions (for analysis)
        positions = signals_aligned.copy()
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(strategy_returns, equity_curve)
        
        print(f"✅ Backtest completed - Total Return: {metrics['total_return']:.2%}")
        
        return BacktestResults(
            strategy_name=strategy_name,
            strategy_returns=strategy_returns,
            equity_curve=equity_curve,
            weights=weights,
            positions=positions,
            metrics=metrics
        )
    
    def _calculate_weights(self, signals: pd.DataFrame, portfolio_type: str) -> pd.DataFrame:
        """
        Calculate portfolio weights from signals.
        
        Args:
            signals: DataFrame of signals (-1, 0, 1)
            portfolio_type: Portfolio construction method
        
        Returns:
            DataFrame of portfolio weights
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        
        if portfolio_type == "long_only":
            # Only long positions
            long_signals = (signals > 0).astype(int)
            n_long = long_signals.sum(axis=1)
            
            # Equal weight among long positions
            for date in signals.index:
                if n_long.loc[date] > 0:
                    weights.loc[date] = long_signals.loc[date] / n_long.loc[date]
                    
        elif portfolio_type == "long_short":
            # Both long and short positions, but not necessarily dollar neutral
            n_long = (signals == 1).sum(axis=1)
            n_short = (signals == -1).sum(axis=1)
            
            for date in signals.index:
                # Long positions
                if n_long.loc[date] > 0:
                    long_mask = signals.loc[date] == 1
                    weights.loc[date, long_mask] = 0.5 / n_long.loc[date]
                
                # Short positions
                if n_short.loc[date] > 0:
                    short_mask = signals.loc[date] == -1
                    weights.loc[date, short_mask] = -0.5 / n_short.loc[date]
                    
        elif portfolio_type == "dollar_neutral":
            # Strict dollar neutral: long exposure = short exposure
            n_long = (signals == 1).sum(axis=1)
            n_short = (signals == -1).sum(axis=1)
            
            for date in signals.index:
                # Only create positions if we have both long and short
                if n_long.loc[date] > 0 and n_short.loc[date] > 0:
                    # Long positions: 50% of capital
                    long_mask = signals.loc[date] == 1
                    weights.loc[date, long_mask] = 0.5 / n_long.loc[date]
                    
                    # Short positions: 50% of capital
                    short_mask = signals.loc[date] == -1
                    weights.loc[date, short_mask] = -0.5 / n_short.loc[date]
                    
        elif portfolio_type == "risk_parity":
            # Risk parity: weight inversely proportional to volatility
            vol_window = 20
            rolling_vol = self.returns_data.rolling(vol_window).std()
            
            for date in signals.index[vol_window:]:
                active_positions = signals.loc[date] != 0
                
                if active_positions.sum() > 0:
                    # Get volatilities for active positions
                    active_vols = rolling_vol.loc[date, active_positions]
                    
                    # Calculate inverse volatility weights
                    inv_vols = 1 / active_vols
                    risk_weights = inv_vols / inv_vols.sum()
                    
                    # Apply to signals
                    for ticker in active_positions.index[active_positions]:
                        signal_direction = signals.loc[date, ticker]
                        weights.loc[date, ticker] = signal_direction * risk_weights[ticker]
        
        return weights
    
    def _apply_rebalancing(self, weights: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Apply rebalancing frequency to weights.
        
        Args:
            weights: Daily weights DataFrame
            freq: Rebalancing frequency ("weekly", "monthly")
        
        Returns:
            Rebalanced weights DataFrame
        """
        if freq == "weekly":
            # Rebalance on Mondays (weekday == 0)
            rebalance_weights = weights.copy()
            last_weights = pd.Series(0.0, index=weights.columns)
            
            for date in weights.index:
                if date.weekday() == 0:  # Monday
                    last_weights = weights.loc[date]
                else:
                    rebalance_weights.loc[date] = last_weights
            
            return rebalance_weights
            
        elif freq == "monthly":
            # Rebalance on first trading day of month
            rebalance_weights = weights.copy()
            last_weights = pd.Series(0.0, index=weights.columns)
            last_month = None
            
            for date in weights.index:
                current_month = date.month
                if current_month != last_month:
                    last_weights = weights.loc[date]
                    last_month = current_month
                else:
                    rebalance_weights.loc[date] = last_weights
            
            return rebalance_weights
        
        return weights  # Daily rebalancing
    
    def _calculate_metrics(self, returns: pd.Series, equity: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Series of strategy returns
            equity: Series of equity curve
        
        Returns:
            Dictionary of performance metrics
        """
        # Basic return metrics
        total_return = equity.iloc[-1] - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Volatility metrics
        volatility = returns.std() * np.sqrt(252)
        downside_vol = returns[returns < 0].std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown metrics
        rolling_max = equity.expanding().max()
        drawdown = (equity / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            drawdown_periods = []
            start_dd = None
            
            for date, is_dd in in_drawdown.items():
                if is_dd and start_dd is None:
                    start_dd = date
                elif not is_dd and start_dd is not None:
                    duration = (date - start_dd).days
                    drawdown_periods.append(duration)
                    start_dd = None
            
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        else:
            avg_drawdown_duration = 0
            max_drawdown_duration = 0
        
        # Win rate and profit metrics
        positive_days = (returns > 0).sum()
        win_rate = positive_days / len(returns)
        
        avg_win = returns[returns > 0].mean() if positive_days > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns) - positive_days > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
        
        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else np.inf
        
        # Information ratio (monthly)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        information_ratio = monthly_returns.mean() / monthly_returns.std() * np.sqrt(12) if len(monthly_returns) > 1 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'downside_volatility': downside_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'n_trades_per_year': len(returns) * (252 / len(returns)),  # Approximation
            'years_tested': n_years
        }
    
    def run_multiple_backtests(self, 
                              signals_dict: Dict[str, pd.DataFrame],
                              portfolio_type: str = "long_short") -> Dict[str, BacktestResults]:
        """
        Run backtests for multiple strategies.
        
        Args:
            signals_dict: Dictionary of strategy_name -> signals DataFrame
            portfolio_type: Portfolio construction method
        
        Returns:
            Dictionary of strategy_name -> BacktestResults
        """
        print(f"🚀 Running {len(signals_dict)} backtests...")
        
        results = {}
        for strategy_name, signals in signals_dict.items():
            results[strategy_name] = self.run_backtest(
                signals=signals,
                strategy_name=strategy_name,
                portfolio_type=portfolio_type
            )
        
        print("✅ All backtests completed!")
        return results
    
    def compare_strategies(self, results: Dict[str, BacktestResults]) -> pd.DataFrame:
        """
        Compare multiple strategy results.
        
        Args:
            results: Dictionary of BacktestResults
        
        Returns:
            DataFrame comparing all strategies
        """
        comparison_data = []
        
        for strategy_name, result in results.items():
            metrics = result.metrics.copy()
            metrics['strategy'] = strategy_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data).set_index('strategy')
        
        # Sort by Sharpe ratio descending
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        
        return comparison_df

if __name__ == "__main__":
    # Test the backtest engine
    from data_loader import load_all_data
    from signals import basic_momentum_signals, basic_contrarian_signals
    
    print("🧪 Testing backtest engine...")
    
    # Load data
    _, returns = load_all_data()
    
    # Generate test signals
    mom_signals = basic_momentum_signals(returns)
    con_signals = basic_contrarian_signals(returns)
    
    # Initialize backtest engine
    engine = BacktestEngine(returns)
    
    # Run backtests
    results = engine.run_multiple_backtests({
        'Basic Momentum': mom_signals,
        'Basic Contrarian': con_signals
    })
    
    # Compare results
    comparison = engine.compare_strategies(results)
    print("\n📊 Strategy Comparison:")
    print(comparison[['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']].round(4))
    
    print("\n✅ Backtest engine test completed!")