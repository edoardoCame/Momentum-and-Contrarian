#!/usr/bin/env python3
"""
Volatility Estimator Module for TSMOM Strategy

Implements EWMA volatility estimation with center-of-mass ≈ 60 days
following Moskowitz-Ooi-Pedersen (2012) specifications.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict

class TSMOMVolatilityEstimator:
    """
    EWMA volatility estimator for TSMOM strategy
    """
    
    def __init__(self, center_of_mass: float = 60.0, annualization_factor: float = 261.0):
        """
        Initialize volatility estimator
        
        Args:
            center_of_mass: EWMA center of mass parameter (≈ 60 days as per paper)
            annualization_factor: Days per year for annualization (261 trading days)
        """
        self.center_of_mass = center_of_mass
        self.annualization_factor = annualization_factor
        
        # Calculate decay parameter: center_of_mass = δ / (1 - δ)
        # Solving: δ = center_of_mass / (1 + center_of_mass)
        self.decay = center_of_mass / (1 + center_of_mass)
        
        print(f"EWMA Volatility Estimator initialized:")
        print(f"Center of mass: {self.center_of_mass:.1f} days")
        print(f"Decay parameter (δ): {self.decay:.4f}")
        print(f"Annualization factor: {self.annualization_factor} days")
    
    def calculate_ewma_variance(self, returns: pd.DataFrame, min_periods: int = 60) -> pd.DataFrame:
        """
        Calculate EWMA variance using pandas exponential weighted functions
        
        Args:
            returns: Daily returns DataFrame
            min_periods: Minimum periods required for calculation
            
        Returns:
            DataFrame with EWMA variance estimates
        """
        print(f"Calculating EWMA variance (shape: {returns.shape})...")
        
        # Calculate EWMA variance using pandas
        # Note: pandas uses alpha = 2 / (span + 1), where span = 2 * center_of_mass + 1
        span = 2 * self.center_of_mass + 1
        
        ewma_var = returns.ewm(
            span=span,
            min_periods=min_periods,
            adjust=False  # Use recursive calculation as per academic literature
        ).var()
        
        print(f"EWMA variance calculated (span: {span:.0f}, avg obs: {ewma_var.notna().sum().mean():.0f})")
        
        return ewma_var
    
    def annualize_volatility(self, variance: pd.DataFrame) -> pd.DataFrame:
        """
        Convert daily EWMA variance to annualized volatility
        
        Args:
            variance: Daily EWMA variance DataFrame
            
        Returns:
            Annualized volatility DataFrame
        """
        # σ_annual = sqrt(variance_daily * annualization_factor)
        annual_vol = np.sqrt(variance * self.annualization_factor)
        
        print(f"Annualized volatility: {annual_vol.mean().mean():.1%} avg, {annual_vol.min().min():.1%}-{annual_vol.max().max():.1%} range")
        
        return annual_vol
    
    def calculate_monthly_volatility(self, daily_returns: pd.DataFrame, 
                                   lag_periods: int = 1) -> pd.DataFrame:
        """
        Calculate monthly volatility estimates (vectorized pipeline)
        
        Args:
            daily_returns: Daily returns DataFrame
            lag_periods: Number of periods to lag
            
        Returns:
            Monthly volatility DataFrame
        """
        print(f"\nCalculating monthly volatility (vectorized pipeline)...")
        
        # VECTORIZED: Single pipeline - variance -> volatility -> lag -> resample
        daily_variance = self.calculate_ewma_variance(daily_returns)
        monthly_vol = (np.sqrt(daily_variance * self.annualization_factor)
                      .shift(lag_periods)
                      .resample('M').last())
        
        # Single-pass statistics
        if not monthly_vol.empty:
            valid_counts = monthly_vol.notna().sum()
            print(f"Monthly vol: {monthly_vol.shape}, valid obs: {valid_counts.mean():.0f} avg")
        
        return monthly_vol
    
    def get_vol_target_weights(self, monthly_vol: pd.DataFrame, 
                             target_vol: float = 0.40) -> pd.DataFrame:
        """
        Calculate volatility target weights (vectorized)
        
        Args:
            monthly_vol: Monthly volatility estimates
            target_vol: Target volatility
            
        Returns:
            Vol target weights DataFrame
        """
        print(f"Calculating vol target weights (target: {target_vol:.0%})...")
        
        # VECTORIZED: Single operation chain
        max_weight = 16.0  # 10x leverage cap (40% / 2.5% min vol)
        vol_weights = (target_vol / monthly_vol).replace([np.inf, -np.inf], 0).fillna(0).clip(upper=max_weight)
        
        # Single-pass statistics  
        print(f"Vol weights: {vol_weights.mean().mean():.2f} avg, {vol_weights.max().max():.2f} max")
        return vol_weights
    
    def validate_volatility_estimates(self, daily_returns: pd.DataFrame, 
                                    monthly_vol: pd.DataFrame) -> Dict[str, float]:
        """
        Validate volatility estimates (vectorized correlations)
        
        Args:
            daily_returns: Daily returns for validation
            monthly_vol: Monthly volatility estimates
            
        Returns:
            Dictionary with validation statistics
        """
        print(f"\nValidating volatility estimates...")
        
        # VECTORIZED: Calculate realized volatility
        realized_vol_monthly = (daily_returns.rolling(21).std() * np.sqrt(self.annualization_factor)).resample('M').last()
        
        # VECTORIZED: Align and calculate correlations in batch
        aligned_est, aligned_real = monthly_vol.align(realized_vol_monthly, join='inner')
        
        if len(aligned_est) > 12:
            # VECTORIZED: Calculate all correlations at once
            correlations = [aligned_est[col].corr(aligned_real[col]) 
                          for col in aligned_est.columns 
                          if aligned_est[col].notna().sum() > 12 and aligned_real[col].notna().sum() > 12]
            
            correlations = [c for c in correlations if not np.isnan(c)]
            
            if correlations:
                stats = {'mean_correlation': np.mean(correlations), 
                        'assets_validated': len(correlations)}
                print(f"Validation: {stats['mean_correlation']:.3f} avg correlation, {stats['assets_validated']} assets")
                return stats
        
        print("Insufficient data for validation")
        return {'error': 'insufficient_data'}