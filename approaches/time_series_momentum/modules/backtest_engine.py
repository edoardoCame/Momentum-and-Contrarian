import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SimpleTSMOMBacktest:
    """
    Simplified backtesting engine for TSMOM strategies.
    
    Clean, vectorized backtesting with proper pandas Series output and 
    unified interface for all frequencies.
    """
    
    def __init__(self, transaction_cost_bps: float = 5.0):
        self.tc_bps = transaction_cost_bps
    
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
        # Calculate gross returns
        asset_returns = signals * returns
        active_positions = np.abs(signals).sum(axis=1)
        
        # Equal weight portfolio returns
        gross_returns = asset_returns.sum(axis=1) / np.maximum(active_positions, 1)
        gross_returns = np.where(active_positions > 0, gross_returns, 0)
        
        # Apply transaction costs
        net_returns = self._apply_transaction_costs(gross_returns, signals)
        
        # Calculate equity curves (ensure pandas Series output)
        gross_equity = (1 + pd.Series(gross_returns, index=returns.index)).cumprod()
        net_equity = (1 + pd.Series(net_returns, index=returns.index)).cumprod()
        
        # Package results with proper Series format
        results = {
            'strategy_name': strategy_name,
            'gross_returns': pd.Series(gross_returns, index=returns.index),
            'net_returns': pd.Series(net_returns, index=returns.index),
            'gross_equity': gross_equity,
            'net_equity': net_equity,
            'active_positions': pd.Series(active_positions, index=returns.index)
        }
        
        return results
    
    def _apply_transaction_costs(self, gross_returns: np.ndarray, 
                               signals: pd.DataFrame) -> np.ndarray:
        """
        Simple transaction cost application.
        
        Args:
            gross_returns: Array of gross returns
            signals: DataFrame with position signals
            
        Returns:
            Array of net returns after transaction costs
        """
        # Calculate turnover (position changes)
        position_changes = signals.diff().fillna(signals)
        total_turnover = np.abs(position_changes).sum(axis=1)
        
        # Transaction costs as percentage of turnover
        transaction_costs = total_turnover * (self.tc_bps / 10000)
        
        # Apply costs
        net_returns = gross_returns - transaction_costs
        
        return net_returns
    
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
            gross_returns = strategy_results['gross_returns']
            net_returns = strategy_results['net_returns']
            freq_mult = 12  # Monthly frequency
            
            # Calculate metrics for both gross and net
            gross_metrics = self._single_strategy_metrics(gross_returns, freq_mult)
            net_metrics = self._single_strategy_metrics(net_returns, freq_mult)
            
            # Combine with prefixes
            combined_metrics = {}
            for key, value in gross_metrics.items():
                combined_metrics[f'Gross_{key}'] = value
            for key, value in net_metrics.items():
                combined_metrics[f'Net_{key}'] = value
                
            # Add transaction cost impact
            combined_metrics['TC_Impact_Annual'] = (gross_metrics['Annual_Return'] - 
                                                  net_metrics['Annual_Return'])
            
            metrics[strategy_name] = combined_metrics
        
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
        metrics_df.to_parquet(output_path / "tsmom_performance_metrics.parquet")
        
    
    def _save_frequency_results(self, results: Dict, output_path: Path, freq: str) -> None:
        """Save results for a single frequency."""
        equity_curves = {}
        returns_data = {}
        
        for strategy_name, strategy_results in results.items():
            equity_curves[f"{strategy_name}_Gross"] = strategy_results['gross_equity']
            equity_curves[f"{strategy_name}_Net"] = strategy_results['net_equity']
            returns_data[f"{strategy_name}_Gross"] = strategy_results['gross_returns']
            returns_data[f"{strategy_name}_Net"] = strategy_results['net_returns']
        
        # Save to files
        pd.DataFrame(equity_curves).to_parquet(output_path / f"tsmom_{freq}_equity_curves.parquet")
        pd.DataFrame(returns_data).to_parquet(output_path / f"tsmom_{freq}_returns.parquet")


