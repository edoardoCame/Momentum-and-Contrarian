import pandas as pd
import numpy as np
from typing import Dict


class WeeklyTSMOM:
    """
    Time Series Momentum strategy implementation for weekly rebalancing.
    
    Core logic: Long if past return > 0, Short if past return < 0 (momentum).
    Uses weekly frequency for higher granularity testing.
    """
    
    def __init__(self, lookbacks_weekly=[1, 2, 4, 8]):
        self.lookbacks = {
            'weekly': lookbacks_weekly
        }
        
    def generate_all_signals(self, weekly_returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate TSMOM signals for weekly frequency.
        
        Args:
            weekly_returns: DataFrame with weekly returns
            
        Returns:
            Dictionary with all TSMOM strategy signals
        """
        all_signals = {}
        
        # Generate weekly TSMOM signals
        for lookback in self.lookbacks['weekly']:
            strategy_name = f"TSMOM_{lookback}W"
            signals = self._momentum_signal(weekly_returns, lookback)
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
        past_performance = returns.rolling(window=lookback).sum().shift(1)
        
        # Generate momentum signals: +1 for positive past performance (buy winners), 
        # -1 for negative past performance (sell losers), 0 for NaN
        signals = np.sign(past_performance).fillna(0)
        
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
                           weekly_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create quick summary of signal characteristics.
        
        Args:
            signals_dict: Dictionary of strategy signals
            weekly_returns: DataFrame with weekly returns
            
        Returns:
            DataFrame with signal summary statistics
        """
        summary_stats = {}
        
        for strategy_name, signals in signals_dict.items():
            # Calculate portfolio returns for this strategy
            portfolio_returns = self.calculate_portfolio_returns(signals, weekly_returns)
            
            # Basic statistics
            total_signals = np.abs(signals).sum().sum()
            long_signals = (signals == 1).sum().sum()
            short_signals = (signals == -1).sum().sum()
            avg_active = np.abs(signals).sum(axis=1).mean()
            
            # Performance metrics (weekly frequency)
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = portfolio_returns.mean() * 52
            annual_vol = portfolio_returns.std() * np.sqrt(52)
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


class WeeklyContrarian:
    """
    Contrarian strategy implementation for weekly rebalancing.
    
    Core logic: Long if past return < 0, Short if past return > 0 (contrarian).
    Uses weekly frequency for higher granularity testing.
    """
    
    def __init__(self, lookbacks_weekly=[1, 2, 4, 8]):
        self.lookbacks = {
            'weekly': lookbacks_weekly
        }
        
    def generate_all_signals(self, weekly_returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate Contrarian signals for weekly frequency.
        
        Args:
            weekly_returns: DataFrame with weekly returns
            
        Returns:
            Dictionary with all strategy signals
        """
        all_signals = {}
        
        # Generate weekly contrarian signals
        for lookback in self.lookbacks['weekly']:
            strategy_name = f"CONTRARIAN_{lookback}W"
            signals = self._contrarian_signal(weekly_returns, lookback)
            all_signals[strategy_name] = signals
            
        return all_signals
    
    def _contrarian_signal(self, returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Simple, clean contrarian signal calculation.
        
        Args:
            returns: DataFrame with returns data
            lookback: Number of periods for contrarian calculation
            
        Returns:
            DataFrame with signals (-1, 0, +1)
        """
        # Calculate cumulative return over lookback period
        # Use shift(1) to prevent lookahead bias
        past_performance = returns.rolling(window=lookback).sum().shift(1)
        
        # Generate contrarian signals: +1 for negative past performance (buy losers), 
        # -1 for positive past performance (sell winners), 0 for NaN
        signals = -np.sign(past_performance).fillna(0)
        
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
                           weekly_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create quick summary of signal characteristics.
        
        Args:
            signals_dict: Dictionary of strategy signals
            weekly_returns: DataFrame with weekly returns
            
        Returns:
            DataFrame with signal summary statistics
        """
        summary_stats = {}
        
        for strategy_name, signals in signals_dict.items():
            # Calculate portfolio returns for this strategy
            portfolio_returns = self.calculate_portfolio_returns(signals, weekly_returns)
            
            # Basic statistics
            total_signals = np.abs(signals).sum().sum()
            long_signals = (signals == 1).sum().sum()
            short_signals = (signals == -1).sum().sum()
            avg_active = np.abs(signals).sum(axis=1).mean()
            
            # Performance metrics (weekly frequency)
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = portfolio_returns.mean() * 52
            annual_vol = portfolio_returns.std() * np.sqrt(52)
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


