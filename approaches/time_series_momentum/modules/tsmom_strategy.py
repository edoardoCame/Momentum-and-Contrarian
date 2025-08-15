import pandas as pd
import numpy as np
from typing import Dict


class SimpleTSMOM:
    """
    Simplified Time Series Momentum strategy implementation.
    
    Clean, vectorized momentum signals with unified interface for all frequencies.
    Core logic: Long if past return > 0, Short if past return < 0.
    """
    
    def __init__(self, lookbacks_monthly=[1, 3, 6, 12]):
        self.lookbacks = {
            'monthly': lookbacks_monthly
        }
        
    def generate_all_signals(self, monthly_returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate TSMOM signals for monthly frequency only.
        
        Args:
            monthly_returns: DataFrame with monthly returns
            
        Returns:
            Dictionary with all strategy signals
        """
        all_signals = {}
        
        # Generate monthly signals only
        for lookback in self.lookbacks['monthly']:
            strategy_name = f"TSMOM_{lookback}M"
            signals = self._momentum_signal(monthly_returns, lookback)
            all_signals[strategy_name] = signals
            
        return all_signals
    
    def _momentum_signal(self, returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Simple, clean momentum signal calculation.
        
        Args:
            returns: DataFrame with returns data
            lookback: Number of periods for momentum calculation
            
        Returns:
            DataFrame with signals (-1, 0, +1)
        """
        # Calculate cumulative return over lookback period
        # Use shift(1) to prevent lookahead bias
        momentum = returns.rolling(window=lookback).sum().shift(1)
        
        # Generate signals: +1 for positive momentum, -1 for negative, 0 for NaN
        signals = np.sign(momentum).fillna(0)
        
        return signals
    
    def calculate_portfolio_returns(self, signals: pd.DataFrame, 
                                  returns: pd.DataFrame) -> pd.Series:
        """
        Calculate equal-weighted portfolio returns from signals.
        
        Args:
            signals: DataFrame with position signals (-1, 0, +1)
            returns: DataFrame with asset returns
            
        Returns:
            Series with portfolio returns
        """
        # Calculate weighted returns
        asset_returns = signals * returns
        
        # Equal weight across active positions
        active_positions = np.abs(signals).sum(axis=1)
        portfolio_returns = asset_returns.sum(axis=1) / np.maximum(active_positions, 1)
        
        # Set returns to 0 when no active positions
        portfolio_returns = np.where(active_positions > 0, portfolio_returns, 0)
        
        return pd.Series(portfolio_returns, index=returns.index, name='portfolio_returns')
    
    def get_strategy_summary(self, signals_dict: Dict[str, pd.DataFrame], 
                           monthly_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create quick summary of signal characteristics.
        
        Args:
            signals_dict: Dictionary of strategy signals
            monthly_returns: DataFrame with monthly returns
            
        Returns:
            DataFrame with signal summary statistics
        """
        summary_stats = {}
        
        for strategy_name, signals in signals_dict.items():
            # Calculate portfolio returns for this strategy
            portfolio_returns = self.calculate_portfolio_returns(signals, monthly_returns)
            
            # Basic statistics
            total_signals = np.abs(signals).sum().sum()
            long_signals = (signals == 1).sum().sum()
            short_signals = (signals == -1).sum().sum()
            avg_active = np.abs(signals).sum(axis=1).mean()
            
            # Performance metrics (monthly frequency)
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = portfolio_returns.mean() * 12
            annual_vol = portfolio_returns.std() * np.sqrt(12)
            sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
            
            summary_stats[strategy_name] = {
                'Total_Signals': total_signals,
                'Long_Pct': long_signals / total_signals * 100 if total_signals > 0 else 0,
                'Short_Pct': short_signals / total_signals * 100 if total_signals > 0 else 0,
                'Avg_Active_Positions': avg_active,
                'Total_Return': total_return,
                'Annual_Return': annual_return,
                'Annual_Vol': annual_vol,
                'Sharpe_Ratio': sharpe
            }
        
        return pd.DataFrame(summary_stats).T


