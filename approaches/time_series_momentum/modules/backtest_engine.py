import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SimpleContrarianBacktest:
    """
    Simplified backtesting engine for Contrarian strategies.
    
    Clean, vectorized backtesting with proper pandas Series output and 
    unified interface for all frequencies.
    """
    
    def __init__(self):
        pass
    
    def run_backtest(self, signals_dict: Dict[str, pd.DataFrame], 
                     monthly_returns: pd.DataFrame) -> Dict:
        """
        Run backtests for all monthly strategies.
        
        Args:
            signals_dict: Dictionary of strategy signals
            monthly_returns: DataFrame with monthly returns
            
        Returns:
            Dictionary with all backtest results
        """
        
        all_results = {}
        
        for strategy_name, signals in signals_dict.items():
            # Run individual strategy backtest with monthly returns
            results = self._calculate_strategy_returns(signals, monthly_returns, strategy_name)
            all_results[strategy_name] = results
            
        return all_results
    
    def _calculate_strategy_returns(self, signals: pd.DataFrame, returns: pd.DataFrame, 
                                  strategy_name: str) -> Dict:
        """
        Calculate returns for a single strategy with proper Series output.
        
        Args:
            signals: DataFrame with position signals
            returns: DataFrame with asset returns
            strategy_name: Name of strategy
            
        Returns:
            Dictionary with strategy results (all pandas Series)
        """
        # Calculate portfolio returns
        asset_returns = signals * returns
        active_positions = np.abs(signals).sum(axis=1)
        
        # Equal weight portfolio returns
        portfolio_returns = asset_returns.sum(axis=1) / np.maximum(active_positions, 1)
        portfolio_returns = np.where(active_positions > 0, portfolio_returns, 0)
        
        # Calculate equity curves (ensure pandas Series output)
        equity_curve = (1 + pd.Series(portfolio_returns, index=returns.index)).cumprod()
        
        # Package results with proper Series format
        results = {
            'strategy_name': strategy_name,
            'returns': pd.Series(portfolio_returns, index=returns.index),
            'equity': equity_curve,
            'active_positions': pd.Series(active_positions, index=returns.index)
        }
        
        return results
    
    
    def calculate_metrics(self, results: Dict) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            results: Dictionary with backtest results
            
        Returns:
            DataFrame with performance metrics
        """
        metrics = {}
        
        for strategy_name, strategy_results in results.items():
            # Get returns (all monthly frequency)
            strategy_returns = strategy_results['returns']
            freq_mult = 12  # Monthly frequency
            
            # Calculate performance metrics
            strategy_metrics = self._single_strategy_metrics(strategy_returns, freq_mult)
            
            metrics[strategy_name] = strategy_metrics
        
        return pd.DataFrame(metrics).T
    
    def _single_strategy_metrics(self, returns: pd.Series, freq_mult: int) -> Dict:
        """Calculate metrics for a single return series."""
        if len(returns.dropna()) == 0:
            return {key: np.nan for key in ['Total_Return', 'Annual_Return', 
                                          'Annual_Vol', 'Sharpe_Ratio', 'Max_Drawdown']}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * freq_mult
        annual_vol = returns.std() * np.sqrt(freq_mult)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else np.nan
        
        # Drawdown calculation (ensure Series input)
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Annual_Vol': annual_vol,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown
        }
    
    def _quick_sharpe(self, returns: pd.Series) -> float:
        """Quick Sharpe ratio calculation for logging."""
        if len(returns) == 0 or returns.std() == 0:
            return np.nan
        freq_mult = 12  # Monthly frequency
        return returns.mean() / returns.std() * np.sqrt(freq_mult)
    
    def save_results(self, results: Dict, output_dir: str = "../results") -> None:
        """
        Save backtest results to files.
        
        Args:
            results: Dictionary with backtest results
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save monthly results (all strategies are monthly now)
        self._save_frequency_results(results, output_path, "monthly")
        
        # Save combined metrics
        metrics_df = self.calculate_metrics(results)
        metrics_df.to_parquet(output_path / "contrarian_performance_metrics.parquet")
        
    
    def _save_frequency_results(self, results: Dict, output_path: Path, freq: str) -> None:
        """Save results for a single frequency."""
        equity_curves = {}
        returns_data = {}
        
        for strategy_name, strategy_results in results.items():
            equity_curves[strategy_name] = strategy_results['equity']
            returns_data[strategy_name] = strategy_results['returns']
        
        # Save to files
        pd.DataFrame(equity_curves).to_parquet(output_path / f"contrarian_{freq}_equity_curves.parquet")
        pd.DataFrame(returns_data).to_parquet(output_path / f"contrarian_{freq}_returns.parquet")


