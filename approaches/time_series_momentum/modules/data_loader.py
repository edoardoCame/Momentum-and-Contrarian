import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleForexLoader:
    """
    Simplified forex data loader for TSMOM strategies.
    Single class that handles all data loading and resampling in one place.
    """
    
    def __init__(self):
        self.data_dir = self._find_data_dir()
    
    def _find_data_dir(self) -> Path:
        """Simple path resolution for data directory."""
        possible_paths = [
            "../../../data/forex",  # From modules
            "../../data/forex",     # From notebooks  
            "../data/forex",        # From approaches
            "data/forex"            # From root
        ]
        
        for path in possible_paths:
            test_path = Path(path)
            if test_path.exists() and list(test_path.glob("*_X.parquet")):
                return test_path
                
        raise FileNotFoundError(f"Forex data directory not found. Tried: {possible_paths}")
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and process all forex data in one method.
        
        Returns:
            Tuple of (daily_returns, weekly_returns, monthly_returns)
        """
        
        # Load raw price data
        forex_data = self._load_forex_pairs()
        
        # Calculate daily returns
        daily_returns = self._calculate_returns(forex_data)
        
        # Resample to different frequencies
        weekly_returns = self._resample_returns(daily_returns, 'W-FRI', 'weekly')
        monthly_returns = self._resample_returns(daily_returns, 'BM', 'monthly')
        
        return daily_returns, weekly_returns, monthly_returns
    
    def _load_forex_pairs(self) -> dict:
        """Load all forex pairs efficiently."""
        forex_files = list(self.data_dir.glob("*_X.parquet"))
        forex_data = {}
        
        for file_path in forex_files:
            symbol = file_path.stem
            try:
                df = pd.read_parquet(file_path)
                
                # Clean index and columns
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                df.index = pd.to_datetime(df.index)
                df.columns = [col.title() for col in df.columns]
                
                forex_data[symbol] = df
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
        
        return forex_data
    
    def _calculate_returns(self, forex_data: dict, price_col: str = 'Close') -> pd.DataFrame:
        """Vectorized returns calculation."""
        returns_data = {}
        
        for symbol, df in forex_data.items():
            if price_col in df.columns:
                returns_data[symbol] = df[price_col].pct_change()
        
        # Combine and clean
        returns_df = pd.DataFrame(returns_data).dropna(how='all')
        return returns_df
    
    def _resample_returns(self, daily_returns: pd.DataFrame, freq: str, name: str) -> pd.DataFrame:
        """Simple resampling for any frequency."""
        resampled = daily_returns.resample(freq).apply(lambda x: (1 + x).prod() - 1)
        resampled = resampled.dropna(how='all')
        
        return resampled
    
    def get_available_pairs(self) -> list:
        """Get list of available forex pairs."""
        forex_files = list(self.data_dir.glob("*_X.parquet"))
        return [f.stem for f in forex_files]
    
    def analyze_data_availability(self) -> pd.DataFrame:
        """
        Analyze data availability for all forex pairs.
        
        Returns:
            DataFrame with start_date, end_date, duration_years, and data_points for each pair
        """
        availability_data = []
        forex_files = list(self.data_dir.glob("*_X.parquet"))
        
        for file_path in forex_files:
            symbol = file_path.stem
            try:
                df = pd.read_parquet(file_path)
                
                # Clean index
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                df.index = pd.to_datetime(df.index)
                
                # Calculate availability metrics
                start_date = df.index.min()
                end_date = df.index.max()
                duration_years = (end_date - start_date).days / 365.25
                data_points = len(df)
                
                # Categorize pair type
                if 'USD' in symbol:
                    if symbol.startswith('USD'):
                        pair_type = 'USD_Base'  # USD/XXX
                    else:
                        pair_type = 'USD_Quote'  # XXX/USD
                else:
                    pair_type = 'Cross'  # XXX/YYY
                
                availability_data.append({
                    'pair': symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_years': duration_years,
                    'data_points': data_points,
                    'pair_type': pair_type
                })
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        
        # Create DataFrame and sort by start date
        availability_df = pd.DataFrame(availability_data)
        if not availability_df.empty:
            availability_df = availability_df.sort_values('start_date')
        
        return availability_df


