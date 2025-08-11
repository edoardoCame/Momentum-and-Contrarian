"""
Portfolio construction and optimization utilities.
Advanced portfolio construction techniques for momentum strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Advanced portfolio construction and optimization.
    """
    
    def __init__(self, returns_data: pd.DataFrame, lookback_window: int = 252):
        """
        Initialize portfolio optimizer.
        
        Args:
            returns_data: DataFrame of asset returns
            lookback_window: Window for covariance estimation
        """
        self.returns_data = returns_data
        self.lookback_window = lookback_window
    
    def calculate_risk_parity_weights(self, 
                                    signals: pd.DataFrame,
                                    vol_window: int = 20) -> pd.DataFrame:
        """
        Calculate risk parity weights for active positions.
        
        Args:
            signals: DataFrame of signals
            vol_window: Window for volatility estimation
        
        Returns:
            DataFrame of risk parity weights
        """
        print("⚖️ Calculating risk parity weights...")
        
        # Calculate rolling volatilities
        rolling_vol = self.returns_data.rolling(vol_window).std()
        
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        
        for date in signals.index[vol_window:]:
            # Get active positions for this date
            active_signals = signals.loc[date]
            active_positions = active_signals != 0
            
            if active_positions.sum() > 0:
                # Get volatilities for active positions
                active_vols = rolling_vol.loc[date, active_positions]
                
                # Skip if any volatility is NaN or zero
                if active_vols.isna().any() or (active_vols <= 0).any():
                    continue
                
                # Calculate inverse volatility weights
                inv_vols = 1 / active_vols
                risk_weights = inv_vols / inv_vols.sum()
                
                # Apply signal direction
                for ticker in active_positions.index[active_positions]:
                    signal_direction = active_signals[ticker]
                    weights.loc[date, ticker] = signal_direction * risk_weights[ticker]
        
        return weights
    
    def calculate_mean_variance_weights(self,
                                     signals: pd.DataFrame,
                                     target_vol: float = 0.15,
                                     risk_aversion: float = 5.0) -> pd.DataFrame:
        """
        Calculate mean-variance optimal weights.
        
        Args:
            signals: DataFrame of signals
            target_vol: Target portfolio volatility
            risk_aversion: Risk aversion parameter
        
        Returns:
            DataFrame of optimal weights
        """
        print("📊 Calculating mean-variance weights...")
        
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        
        for date in signals.index[self.lookback_window:]:
            # Get historical data window
            start_date = date - pd.Timedelta(days=self.lookback_window * 2)  # Buffer for weekends
            hist_returns = self.returns_data.loc[start_date:date].iloc[:-1]  # Exclude current day
            
            if len(hist_returns) < self.lookback_window // 2:
                continue
                
            # Get active positions
            active_signals = signals.loc[date]
            active_positions = active_signals != 0
            
            if active_positions.sum() < 2:  # Need at least 2 assets for diversification
                continue
            
            active_assets = active_positions.index[active_positions]
            active_returns = hist_returns[active_assets].dropna()
            
            if len(active_returns) < 20:  # Need sufficient history
                continue
            
            try:
                # Calculate expected returns and covariance
                mu = active_returns.mean() * 252  # Annualized
                cov_matrix = active_returns.cov() * 252  # Annualized
                
                # Apply signal constraints (long/short based on signals)
                signal_constraints = active_signals[active_assets].values
                
                # Optimize portfolio
                n_assets = len(active_assets)
                
                def objective(w):
                    portfolio_return = np.dot(w, mu)
                    portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                    return -(portfolio_return - 0.5 * risk_aversion * portfolio_vol**2)
                
                # Constraints
                constraints = []
                
                # Signal direction constraints
                for i, signal in enumerate(signal_constraints):
                    if signal > 0:  # Long signal
                        constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i]})  # w[i] >= 0
                    elif signal < 0:  # Short signal  
                        constraints.append({'type': 'ineq', 'fun': lambda w, i=i: -w[i]})  # w[i] <= 0
                
                # Volatility constraint
                if target_vol:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w: target_vol**2 - np.dot(w.T, np.dot(cov_matrix, w))
                    })
                
                # Initial guess
                x0 = signal_constraints / len(active_assets)
                
                # Bounds
                bounds = [(-1, 1) for _ in range(n_assets)]
                
                # Optimize
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    optimal_weights = result.x
                    # Assign weights
                    for i, asset in enumerate(active_assets):
                        weights.loc[date, asset] = optimal_weights[i]
                        
            except Exception as e:
                # Fall back to equal weights if optimization fails
                n_active = active_positions.sum()
                for asset in active_assets:
                    signal = active_signals[asset]
                    weights.loc[date, asset] = signal / n_active
        
        return weights
    
    def calculate_volatility_target_weights(self,
                                          signals: pd.DataFrame,
                                          target_vol: float = 0.12,
                                          vol_window: int = 30) -> pd.DataFrame:
        """
        Scale portfolio to target volatility.
        
        Args:
            signals: DataFrame of signals
            target_vol: Target annualized volatility
            vol_window: Window for volatility estimation
        
        Returns:
            DataFrame of volatility-scaled weights
        """
        print(f"🎯 Scaling to target volatility: {target_vol:.1%}")
        
        # Start with equal weight positions
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        
        for date in signals.index:
            active_signals = signals.loc[date]
            active_positions = active_signals != 0
            
            if active_positions.sum() > 0:
                # Equal weight among active positions
                n_active = active_positions.sum()
                base_weight = 1.0 / n_active
                
                for ticker in active_positions.index[active_positions]:
                    signal_direction = active_signals[ticker]
                    weights.loc[date, ticker] = signal_direction * base_weight
        
        # Apply volatility scaling
        scaled_weights = weights.copy()
        portfolio_returns = (weights.shift(1) * self.returns_data).sum(axis=1)
        
        # Calculate rolling portfolio volatility
        rolling_vol = portfolio_returns.rolling(vol_window).std() * np.sqrt(252)
        
        for date in rolling_vol.index[vol_window:]:
            if rolling_vol.loc[date] > 0 and not pd.isna(rolling_vol.loc[date]):
                vol_scalar = target_vol / rolling_vol.loc[date]
                # Cap the scaling to avoid extreme leverage
                vol_scalar = np.clip(vol_scalar, 0.1, 3.0)
                scaled_weights.loc[date] *= vol_scalar
        
        return scaled_weights
    
    def blend_strategies(self,
                        strategy_weights: Dict[str, pd.DataFrame],
                        blend_method: str = "equal_weight",
                        optimization_window: int = 252) -> pd.DataFrame:
        """
        Blend multiple strategies into a single portfolio.
        
        Args:
            strategy_weights: Dict of strategy_name -> weights DataFrame
            blend_method: "equal_weight", "inverse_vol", "optimize", "adaptive"
            optimization_window: Window for optimization
        
        Returns:
            DataFrame of blended weights
        """
        print(f"🌟 Blending {len(strategy_weights)} strategies using {blend_method}")
        
        # Align all strategies to common index/columns
        common_index = None
        common_columns = None
        
        for name, weights in strategy_weights.items():
            if common_index is None:
                common_index = weights.index
                common_columns = weights.columns
            else:
                common_index = common_index.intersection(weights.index)
                common_columns = common_columns.intersection(weights.columns)
        
        # Align all strategy weights
        aligned_weights = {}
        for name, weights in strategy_weights.items():
            aligned_weights[name] = weights.loc[common_index, common_columns]
        
        blended_weights = pd.DataFrame(0.0, index=common_index, columns=common_columns)
        
        if blend_method == "equal_weight":
            # Simple equal weighting
            n_strategies = len(aligned_weights)
            for weights in aligned_weights.values():
                blended_weights += weights / n_strategies
                
        elif blend_method == "inverse_vol":
            # Weight by inverse volatility
            strategy_returns = {}
            
            # Calculate returns for each strategy
            for name, weights in aligned_weights.items():
                returns = (weights.shift(1) * self.returns_data.loc[common_index, common_columns]).sum(axis=1)
                strategy_returns[name] = returns
            
            # Calculate rolling inverse volatility weights
            vol_window = 60
            
            for date in common_index[vol_window:]:
                strategy_vols = {}
                for name, returns in strategy_returns.items():
                    vol = returns.loc[:date].iloc[-vol_window:].std()
                    if vol > 0:
                        strategy_vols[name] = vol
                
                if len(strategy_vols) > 0:
                    # Calculate inverse volatility weights
                    inv_vols = {name: 1/vol for name, vol in strategy_vols.items()}
                    total_inv_vol = sum(inv_vols.values())
                    vol_weights = {name: inv_vol/total_inv_vol for name, inv_vol in inv_vols.items()}
                    
                    # Blend weights
                    for name, vol_weight in vol_weights.items():
                        blended_weights.loc[date] += vol_weight * aligned_weights[name].loc[date]
                        
        elif blend_method == "adaptive":
            # Adaptive weighting based on recent performance
            strategy_returns = {}
            lookback_window = 60
            
            # Calculate returns for each strategy
            for name, weights in aligned_weights.items():
                returns = (weights.shift(1) * self.returns_data.loc[common_index, common_columns]).sum(axis=1)
                strategy_returns[name] = returns
            
            for date in common_index[lookback_window:]:
                strategy_scores = {}
                
                for name, returns in strategy_returns.items():
                    recent_returns = returns.loc[:date].iloc[-lookback_window:]
                    
                    if len(recent_returns) > 0:
                        # Score based on Sharpe ratio
                        mean_ret = recent_returns.mean()
                        vol_ret = recent_returns.std()
                        
                        if vol_ret > 0:
                            sharpe = mean_ret / vol_ret
                            # Use softmax to convert to weights
                            strategy_scores[name] = np.exp(sharpe * 10)  # Scale for sensitivity
                        else:
                            strategy_scores[name] = 1.0
                
                if len(strategy_scores) > 0:
                    # Normalize scores to weights
                    total_score = sum(strategy_scores.values())
                    adaptive_weights = {name: score/total_score for name, score in strategy_scores.items()}
                    
                    # Blend weights
                    for name, adaptive_weight in adaptive_weights.items():
                        blended_weights.loc[date] += adaptive_weight * aligned_weights[name].loc[date]
        
        return blended_weights

class RiskManager:
    """
    Risk management utilities for portfolio construction.
    """
    
    @staticmethod
    def apply_position_limits(weights: pd.DataFrame, 
                            max_position: float = 0.1) -> pd.DataFrame:
        """
        Apply maximum position size limits.
        
        Args:
            weights: Portfolio weights
            max_position: Maximum weight per position
        
        Returns:
            Weights with position limits applied
        """
        limited_weights = weights.copy()
        
        # Clip positions to maximum
        limited_weights = limited_weights.clip(-max_position, max_position)
        
        return limited_weights
    
    @staticmethod
    def apply_sector_limits(weights: pd.DataFrame,
                          sectors_map: Dict[str, str],
                          max_sector_exposure: float = 0.3) -> pd.DataFrame:
        """
        Apply sector exposure limits.
        
        Args:
            weights: Portfolio weights
            sectors_map: Mapping of ticker to sector
            max_sector_exposure: Maximum exposure per sector
        
        Returns:
            Weights with sector limits applied
        """
        limited_weights = weights.copy()
        
        for date in weights.index:
            # Calculate current sector exposures
            sector_exposures = {}
            for ticker, weight in weights.loc[date].items():
                if ticker in sectors_map and abs(weight) > 0:
                    sector = sectors_map[ticker]
                    sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(weight)
            
            # Scale down sectors that exceed limits
            for sector, exposure in sector_exposures.items():
                if exposure > max_sector_exposure:
                    scale_factor = max_sector_exposure / exposure
                    
                    # Apply scaling to all positions in this sector
                    sector_tickers = [t for t, s in sectors_map.items() if s == sector and t in weights.columns]
                    for ticker in sector_tickers:
                        limited_weights.loc[date, ticker] *= scale_factor
        
        return limited_weights
    
    @staticmethod
    def apply_turnover_control(weights: pd.DataFrame,
                             max_turnover: float = 2.0) -> pd.DataFrame:
        """
        Control portfolio turnover.
        
        Args:
            weights: Portfolio weights
            max_turnover: Maximum annual turnover
        
        Returns:
            Weights with turnover control
        """
        controlled_weights = weights.copy()
        
        # Calculate daily turnover
        daily_turnover = abs(weights.diff()).sum(axis=1)
        
        # Apply turnover constraint
        max_daily_turnover = max_turnover / 252
        
        for i, date in enumerate(weights.index[1:], 1):
            if daily_turnover.iloc[i] > max_daily_turnover:
                # Scale down the change
                prev_date = weights.index[i-1]
                weight_change = weights.loc[date] - weights.loc[prev_date]
                
                scale_factor = max_daily_turnover / daily_turnover.iloc[i]
                controlled_weights.loc[date] = weights.loc[prev_date] + scale_factor * weight_change
        
        return controlled_weights

if __name__ == "__main__":
    # Test portfolio optimization
    from data_loader import load_all_data
    from signals import basic_contrarian_signals
    
    print("🧪 Testing portfolio optimization...")
    
    # Load data
    _, returns = load_all_data()
    
    # Generate signals
    signals = basic_contrarian_signals(returns)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns)
    
    # Test risk parity
    print("\n⚖️ Testing risk parity weights...")
    rp_weights = optimizer.calculate_risk_parity_weights(signals)
    print(f"Risk parity weights shape: {rp_weights.shape}")
    print(f"Non-zero weights: {(rp_weights != 0).sum().sum()}")
    
    # Test volatility targeting
    print("\n🎯 Testing volatility targeting...")
    vol_weights = optimizer.calculate_volatility_target_weights(signals)
    print(f"Vol target weights shape: {vol_weights.shape}")
    
    print("\n✅ Portfolio optimization tests completed!")