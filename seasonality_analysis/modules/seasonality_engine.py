"""
Seasonality Analysis Engine for Commodities

This module provides comprehensive seasonality analysis tools for commodity futures data.
It handles the multi-level column structure from Yahoo Finance parquet files and calculates
various seasonality metrics with statistical significance testing.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CommoditySeasonalityAnalyzer:
    """
    Comprehensive seasonality analysis for commodity futures data.
    
    Handles Yahoo Finance multi-level column structure and provides
    various seasonality calculations with statistical testing.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the seasonality analyzer.
        
        Args:
            data_path: Path to commodities data directory
        """
        if data_path is None:
            # Default to the project's commodity data directory
            # Get the absolute path to the module file, then navigate to data directory
            module_path = Path(__file__).resolve()  # Get absolute path
            # From seasonality_analysis/modules/seasonality_engine.py 
            # Go up to seasonality_analysis/modules/ -> seasonality_analysis/ -> main_project/ -> data/commodities/
            self.data_path = module_path.parent.parent.parent / "data" / "commodities"
        else:
            self.data_path = Path(data_path)
        
        self.commodities_data = {}
        self.commodity_symbols = {}
        self.sector_mapping = self._create_sector_mapping()
    
    def _create_sector_mapping(self) -> Dict[str, str]:
        """Create mapping of commodity symbols to sectors."""
        return {
            'BZ_F': 'Energy',    # Brent Crude Oil
            'CL_F': 'Energy',    # Crude Oil WTI
            'HO_F': 'Energy',    # Heating Oil
            'NG_F': 'Energy',    # Natural Gas
            'RB_F': 'Energy',    # RBOB Gasoline
            'GC_F': 'Metals',    # Gold
            'SI_F': 'Metals',    # Silver
            'HG_F': 'Metals',    # Copper
            'PA_F': 'Metals',    # Palladium
            'CC_F': 'Agriculture',  # Cocoa
            'CT_F': 'Agriculture',  # Cotton
            'SB_F': 'Agriculture',  # Sugar
            'ZC_F': 'Agriculture',  # Corn
            'ZS_F': 'Agriculture',  # Soybeans
            'ZW_F': 'Agriculture'   # Wheat
        }
    
    def load_commodity_data(self, symbol: str = None) -> pd.DataFrame:
        """
        Load commodity data from parquet files.
        
        Args:
            symbol: Specific commodity symbol to load (e.g., 'GC_F')
                   If None, loads all available commodities
        
        Returns:
            DataFrame with commodity price data
        """
        if symbol:
            file_path = self.data_path / f"{symbol}.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"Commodity data file not found: {file_path}")
            
            data = pd.read_parquet(file_path)
            # Extract the actual symbol from multi-level columns
            actual_symbol = data.columns.get_level_values(1)[0]
            self.commodity_symbols[symbol] = actual_symbol
            self.commodities_data[symbol] = data
            return data
        else:
            # Load all commodities
            parquet_files = list(self.data_path.glob("*.parquet"))
            all_data = {}
            
            for file_path in parquet_files:
                symbol = file_path.stem
                data = pd.read_parquet(file_path)
                actual_symbol = data.columns.get_level_values(1)[0]
                self.commodity_symbols[symbol] = actual_symbol
                self.commodities_data[symbol] = data
                all_data[symbol] = data
            
            return all_data
    
    def extract_returns_data(self, use_adjusted: bool = True) -> pd.DataFrame:
        """
        Extract returns data for all commodities.
        
        Args:
            use_adjusted: Whether to use adjusted close prices
        
        Returns:
            DataFrame with daily returns for all commodities
        """
        if not self.commodities_data:
            self.load_commodity_data()
        
        returns_data = pd.DataFrame()
        price_type = 'Adj Close' if use_adjusted else 'Close'
        
        for symbol, data in self.commodities_data.items():
            actual_symbol = self.commodity_symbols[symbol]
            prices = data[(price_type, actual_symbol)]
            returns = prices.pct_change().dropna()
            returns_data[symbol] = returns
        
        return returns_data.dropna()
    
    def calculate_monthly_seasonality(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly seasonality patterns.
        
        Args:
            returns_data: DataFrame with daily returns
        
        Returns:
            DataFrame with average monthly returns
        """
        # Add month column
        monthly_data = returns_data.copy()
        monthly_data['Month'] = monthly_data.index.month
        
        # Calculate average monthly returns
        monthly_returns = monthly_data.groupby('Month').mean()
        
        # Add month names for better visualization
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns.index = month_names
        
        return monthly_returns
    
    def calculate_seasonal_significance(self, returns_data: pd.DataFrame, 
                                     confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Test statistical significance of seasonal patterns.
        
        Args:
            returns_data: DataFrame with daily returns
            confidence_level: Confidence level for statistical tests
        
        Returns:
            DataFrame with p-values and significance indicators
        """
        monthly_data = returns_data.copy()
        monthly_data['Month'] = monthly_data.index.month
        
        significance_results = pd.DataFrame(index=range(1, 13))
        
        for commodity in returns_data.columns:
            p_values = []
            for month in range(1, 13):
                month_returns = monthly_data[monthly_data['Month'] == month][commodity]
                other_returns = monthly_data[monthly_data['Month'] != month][commodity]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(month_returns, other_returns)
                p_values.append(p_value)
            
            significance_results[f'{commodity}_pvalue'] = p_values
            significance_results[f'{commodity}_significant'] = [
                p < (1 - confidence_level) for p in p_values
            ]
        
        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        significance_results.index = month_names
        
        return significance_results
    
    def calculate_day_of_week_effects(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate day-of-week seasonality effects.
        
        Args:
            returns_data: DataFrame with daily returns
        
        Returns:
            DataFrame with average day-of-week returns
        """
        dow_data = returns_data.copy()
        dow_data['DayOfWeek'] = dow_data.index.day_name()
        
        # Calculate average returns by day of week
        dow_returns = dow_data.groupby('DayOfWeek').mean()
        
        # Reorder to standard week order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        dow_returns = dow_returns.reindex(day_order)
        
        return dow_returns
    
    def calculate_seasonal_volatility(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate seasonal volatility patterns.
        
        Args:
            returns_data: DataFrame with daily returns
        
        Returns:
            DataFrame with monthly volatility averages
        """
        monthly_data = returns_data.copy()
        monthly_data['Month'] = monthly_data.index.month
        
        # Calculate monthly volatility (standard deviation)
        monthly_volatility = monthly_data.groupby('Month').std()
        
        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_volatility.index = month_names
        
        return monthly_volatility
    
    def calculate_sector_seasonality(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate seasonality patterns by sector.
        
        Args:
            returns_data: DataFrame with daily returns
        
        Returns:
            DataFrame with sector-based seasonal patterns
        """
        # Group commodities by sector
        sector_data = {}
        for symbol, sector in self.sector_mapping.items():
            if symbol in returns_data.columns:
                if sector not in sector_data:
                    sector_data[sector] = []
                sector_data[sector].append(returns_data[symbol])
        
        # Calculate equal-weighted sector returns
        sector_returns = pd.DataFrame()
        for sector, commodity_returns in sector_data.items():
            sector_returns[sector] = pd.concat(commodity_returns, axis=1).mean(axis=1)
        
        # Calculate monthly seasonality for sectors
        return self.calculate_monthly_seasonality(sector_returns)
    
    def get_seasonal_summary_stats(self, returns_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive seasonal summary statistics.
        
        Args:
            returns_data: DataFrame with daily returns
        
        Returns:
            Dictionary with various seasonal statistics
        """
        monthly_seasonality = self.calculate_monthly_seasonality(returns_data)
        significance = self.calculate_seasonal_significance(returns_data)
        dow_effects = self.calculate_day_of_week_effects(returns_data)
        seasonal_vol = self.calculate_seasonal_volatility(returns_data)
        sector_seasonality = self.calculate_sector_seasonality(returns_data)
        
        # Calculate best and worst months for each commodity
        best_months = monthly_seasonality.idxmax()
        worst_months = monthly_seasonality.idxmin()
        
        # Calculate seasonal strength (max month - min month)
        seasonal_strength = monthly_seasonality.max() - monthly_seasonality.min()
        
        return {
            'monthly_returns': monthly_seasonality,
            'statistical_significance': significance,
            'day_of_week_effects': dow_effects,
            'seasonal_volatility': seasonal_vol,
            'sector_seasonality': sector_seasonality,
            'best_months': best_months,
            'worst_months': worst_months,
            'seasonal_strength': seasonal_strength,
            'data_period': (returns_data.index.min(), returns_data.index.max()),
            'total_observations': len(returns_data),
            'commodities_analyzed': list(returns_data.columns)
        }
    
    def save_results(self, results: Dict, output_path: str = None):
        """
        Save analysis results to files.
        
        Args:
            results: Dictionary with analysis results
            output_path: Path to save results
        """
        if output_path is None:
            output_path = Path(__file__).parent.parent / "results"
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(exist_ok=True)
        
        # Save each result as CSV
        for key, data in results.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path / f"{key}.csv")
            elif isinstance(data, pd.Series):
                data.to_csv(output_path / f"{key}.csv")