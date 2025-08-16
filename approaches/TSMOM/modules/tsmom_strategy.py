#!/usr/bin/env python3
"""
TSMOM Strategy Module

Implements Time Series Momentum strategy following Moskowitz-Ooi-Pedersen (2012).
Includes signal generation, position sizing, and portfolio construction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

class TSMOMStrategy:
    """
    Time Series Momentum Strategy Implementation
    """
    
    def __init__(self, k: int = 12, h: int = 1, target_vol: float = 0.40):
        """
        Initialize TSMOM strategy
        
        Args:
            k: Lookback period for momentum signals (months)
            h: Holding period (months) 
            target_vol: Target volatility for position sizing (40% as per paper)
        """
        self.k = k
        self.h = h
        self.target_vol = target_vol
        
        print(f"TSMOM Strategy initialized:")
        print(f"Lookback period (k): {self.k} months")
        print(f"Holding period (h): {self.h} months")
        print(f"Target volatility: {self.target_vol:.0%}")
    
    def calculate_momentum_signals(self, monthly_returns: pd.DataFrame, 
                                 min_history: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate TSMOM signals using k-month lookback (vectorized)
        
        CRITICAL: Implements proper temporal separation to eliminate lookahead bias
        - For November positions: uses returns from Nov-k to October only
        - Signals are shifted by 1 period after calculation
        - This ensures November trades use only information available at end of October
        
        Args:
            monthly_returns: Monthly returns DataFrame
            min_history: Minimum history required (defaults to k)
            
        Returns:
            DataFrame with momentum signals (+1, -1, 0) properly lagged
        """
        min_history = min_history or self.k
        print(f"\nCalculating {self.k}-month momentum signals (vectorized)...")
        
        # VECTORIZED: Calculate k-month cumulative returns using optimized rolling
        cumulative_returns = (1 + monthly_returns).rolling(
            window=self.k, min_periods=min_history
        ).apply(lambda x: np.prod(x), raw=True) - 1
        
        # Generate signals in single vectorized operation
        signals = np.sign(cumulative_returns).fillna(0)
        
        # CRITICAL: Shift signals by 1 period to eliminate lookahead bias
        # For November positions, use signals calculated with data only through October
        signals = signals.shift(1).fillna(0)
        
        # VECTORIZED: Quality stats in single pass
        signal_stats = (signals == 1).sum(), (signals == -1).sum(), (signals == 0).sum()
        print(f"Signals - Long: {signal_stats[0].sum()}, Short: {signal_stats[1].sum()}, None: {signal_stats[2].sum()}")
        
        return signals
    
    def calculate_position_weights(self, signals: pd.DataFrame, 
                                 vol_estimates: pd.DataFrame,
                                 max_leverage: float = 10.0) -> pd.DataFrame:
        """
        Calculate position weights using volatility targeting (vectorized)
        
        Args:
            signals: Momentum signals DataFrame
            vol_estimates: Volatility estimates DataFrame  
            max_leverage: Maximum leverage constraint
            
        Returns:
            Position weights DataFrame
        """
        print(f"\nCalculating position weights (target vol: {self.target_vol:.0%})...")
        
        # VECTORIZED: Direct broadcast operations without explicit alignment
        raw_weights = signals * (self.target_vol / vol_estimates)
        
        # VECTORIZED: Clean and cap in single operation chain
        leverage_cap = max_leverage * self.target_vol
        weights = raw_weights.replace([np.inf, -np.inf], 0).fillna(0).clip(-leverage_cap, leverage_cap)
        
        # Single-pass statistics
        weights_abs = weights.abs()
        print(f"Weights: mean={weights_abs.mean().mean():.2f}, max={weights_abs.max().max():.2f}")
        
        return weights
    
    def implement_holding_period(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Implement holding period strategy (vectorized)
        
        Args:
            weights: Position weights DataFrame
            
        Returns:
            Final weights with holding period implementation
        """
        if self.h == 1:
            return weights
        
        print(f"Implementing {self.h}-month holding period...")
        
        # VECTORIZED: Create all shifted weights in single operation
        shifted_weights = pd.concat([weights.shift(i) for i in range(self.h)], axis=1, keys=range(self.h))
        final_weights = shifted_weights.groupby(axis=1, level=1).mean().reindex(columns=weights.columns).fillna(0)
        
        print(f"Holding period applied (reduction: {final_weights.abs().mean().mean() / weights.abs().mean().mean():.2f}x)")
        return final_weights
    
    def calculate_portfolio_returns(self, weights: pd.DataFrame, 
                                  monthly_returns: pd.DataFrame,
                                  equal_weight_assets: bool = True) -> Dict[str, pd.Series]:
        """
        Calculate portfolio returns with equal weighting across assets
        
        Args:
            weights: Position weights DataFrame
            monthly_returns: Monthly returns DataFrame
            equal_weight_assets: Whether to equal weight across assets
            
        Returns:
            Dictionary with portfolio return series
        """
        print(f"Calculating portfolio returns...")
        
        # Align data
        common_index = weights.index.intersection(monthly_returns.index)
        common_cols = weights.columns.intersection(monthly_returns.columns)
        
        weights_aligned = weights.loc[common_index, common_cols]
        returns_aligned = monthly_returns.loc[common_index, common_cols]
        
        # Calculate individual asset contributions
        asset_contributions = weights_aligned * returns_aligned
        
        # Calculate sub-portfolios
        # Identify commodity and FX assets
        commodity_assets = [col for col in common_cols if '=F' in col]
        fx_assets = [col for col in common_cols if '=X' in col]
        
        results = {}
        
        if commodity_assets:
            commodity_contributions = asset_contributions[commodity_assets]
            results['commodities'] = commodity_contributions.mean(axis=1) if equal_weight_assets else commodity_contributions.sum(axis=1)
        
        if fx_assets:
            fx_contributions = asset_contributions[fx_assets]
            results['forex'] = fx_contributions.mean(axis=1) if equal_weight_assets else fx_contributions.sum(axis=1)
        
        # VECTORIZED: Calculate balanced 50/50 portfolio between asset classes
        # Monthly rebalancing on the same day as signal calculation (no lookahead bias)
        if 'commodities' in results and 'forex' in results:
            results['total'] = 0.5 * results['commodities'] + 0.5 * results['forex']
        
        # Calculate number of active positions
        active_positions = (weights_aligned != 0).sum(axis=1)
        results['n_active_assets'] = active_positions
        
        # Report main portfolio performance (50/50 balanced)
        if 'total' in results:
            total_ret = results['total'].mean() * 12
            total_vol = results['total'].std() * np.sqrt(12)
            sharpe = total_ret / total_vol if total_vol > 0 else 0
            print(f"Portfolio (50/50 Balanced): {total_ret:.1%} return, {total_vol:.1%} vol, {sharpe:.2f} Sharpe, {active_positions.mean():.0f} avg assets")
        
        return results
    
    def run_strategy(self, monthly_returns: pd.DataFrame, 
                    vol_estimates: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Run complete TSMOM strategy
        
        Args:
            monthly_returns: Monthly returns DataFrame
            vol_estimates: Volatility estimates DataFrame
            
        Returns:
            Tuple of (final_weights, portfolio_returns_dict)
        """
        print(f"\n{'='*60}")
        print(f"RUNNING TSMOM STRATEGY")
        print(f"{'='*60}")
        
        # Step 1: Generate momentum signals
        signals = self.calculate_momentum_signals(monthly_returns)
        
        # Step 2: Calculate position weights with vol targeting
        weights = self.calculate_position_weights(signals, vol_estimates)
        
        # Step 3: Implement holding period
        final_weights = self.implement_holding_period(weights)
        
        # Step 4: Calculate portfolio returns
        portfolio_returns = self.calculate_portfolio_returns(final_weights, monthly_returns)
        
        print(f"\n{'='*60}")
        print(f"STRATEGY EXECUTION COMPLETE")
        print(f"{'='*60}")
        
        return final_weights, portfolio_returns
    
    def parameter_grid_analysis(self, monthly_returns: pd.DataFrame,
                              vol_estimates: pd.DataFrame,
                              k_values: List[int] = [3, 6, 9, 12],
                              h_values: List[int] = [1, 3, 6]) -> pd.DataFrame:
        """
        Run parameter grid analysis for robustness testing
        
        Args:
            monthly_returns: Monthly returns DataFrame
            vol_estimates: Volatility estimates DataFrame
            k_values: List of lookback periods to test
            h_values: List of holding periods to test
            
        Returns:
            DataFrame with Sharpe ratios for each parameter combination
        """
        print(f"\n{'='*60}")
        print(f"PARAMETER GRID ANALYSIS")
        print(f"{'='*60}")
        print(f"Testing k values: {k_values}")
        print(f"Testing h values: {h_values}")
        
        results = []
        
        for k in k_values:
            for h in h_values:
                print(f"\nTesting k={k}, h={h}...")
                
                # Temporarily update parameters
                original_k, original_h = self.k, self.h
                self.k, self.h = k, h
                
                try:
                    # Run strategy
                    _, portfolio_returns = self.run_strategy(monthly_returns, vol_estimates)
                    
                    # Calculate Sharpe ratio
                    returns = portfolio_returns['total']
                    if len(returns) > 12:  # Need at least 1 year
                        sharpe = returns.mean() / returns.std() * np.sqrt(12)
                        results.append({
                            'k': k,
                            'h': h,
                            'sharpe': sharpe,
                            'mean_return': returns.mean() * 12,
                            'volatility': returns.std() * np.sqrt(12),
                            'n_obs': len(returns)
                        })
                    else:
                        results.append({
                            'k': k,
                            'h': h,
                            'sharpe': np.nan,
                            'mean_return': np.nan,
                            'volatility': np.nan,
                            'n_obs': len(returns)
                        })
                
                except Exception as e:
                    print(f"Error with k={k}, h={h}: {e}")
                    results.append({
                        'k': k,
                        'h': h,
                        'sharpe': np.nan,
                        'mean_return': np.nan,
                        'volatility': np.nan,
                        'n_obs': 0
                    })
                
                # Restore original parameters
                self.k, self.h = original_k, original_h
        
        results_df = pd.DataFrame(results)
        
        # Create Sharpe ratio pivot table
        sharpe_table = results_df.pivot(index='k', columns='h', values='sharpe')
        
        print(f"\nSharpe Ratio Grid:")
        print(sharpe_table.round(3))
        
        return results_df