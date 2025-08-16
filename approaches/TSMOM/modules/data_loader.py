#!/usr/bin/env python3
"""
Data Loader Module for TSMOM Strategy

Downloads and preprocesses Yahoo Finance data for commodities and FX pairs
following Moskowitz-Ooi-Pedersen (2012) specifications.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TSMOMDataLoader:
    """
    Data loader for TSMOM strategy implementation
    """
    
    def __init__(self, start_date: str = "2000-01-01", cache_dir: str = "data/cached"):
        self.start_date = start_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.daily_prices_cache = self.cache_dir / "daily_prices.parquet"
        self.monthly_prices_cache = self.cache_dir / "monthly_prices.parquet"
        self.daily_returns_cache = self.cache_dir / "daily_returns.parquet"
        
        # UPDATED: Load universe from actual data files in main data directory
        self.main_data_dir = Path(__file__).parent.parent.parent.parent / "data"
        self.COM_FUT, self.FX_SPOT = self._discover_universe()
        
        # USD base pairs that need sign adjustment (expanded list)
        self.USD_BASE_PAIRS = [symbol for symbol in self.FX_SPOT if symbol.startswith("USD")]
        
        print(f"TSMOM Data Loader initialized (cache: {self.cache_dir})")
        print(f"Universe: {len(self.COM_FUT)} commodities + {len(self.FX_SPOT)} forex = {len(self.COM_FUT) + len(self.FX_SPOT)} total assets")
        if self.cache_dir.exists():
            cache_info = self.get_cache_info()
            for name, status in cache_info.items():
                print(f"  {name}: {status}")
    
    def _discover_universe(self) -> Tuple[List[str], List[str]]:
        """
        Discover trading universe from actual data files
        
        Returns:
            Tuple of (commodities_list, forex_list) with Yahoo Finance symbols
        """
        commodities_dir = self.main_data_dir / "commodities"
        forex_dir = self.main_data_dir / "forex"
        
        # Discover commodities
        com_fut = []
        if commodities_dir.exists():
            for file in commodities_dir.glob("*.parquet"):
                # Convert filename back to Yahoo Finance symbol (e.g., CL_F.parquet -> CL=F)
                symbol = file.stem.replace("_", "=")
                com_fut.append(symbol)
        
        # Discover forex pairs
        fx_spot = []
        if forex_dir.exists():
            for file in forex_dir.glob("*.parquet"):
                # Skip aggregated files
                if file.name in ["all_pairs_data.parquet", "all_pairs_returns.parquet"]:
                    continue
                # Convert filename back to Yahoo Finance symbol (e.g., EURUSD_X.parquet -> EURUSD=X)
                symbol = file.stem.replace("_", "=")
                fx_spot.append(symbol)
        
        com_fut.sort()
        fx_spot.sort()
        
        print(f"Discovered universe from {self.main_data_dir}:")
        print(f"  Commodities ({len(com_fut)}): {com_fut[:5]}..." if len(com_fut) > 5 else f"  Commodities ({len(com_fut)}): {com_fut}")
        print(f"  Forex ({len(fx_spot)}): {fx_spot[:5]}..." if len(fx_spot) > 5 else f"  Forex ({len(fx_spot)}): {fx_spot}")
        
        return com_fut, fx_spot
    
    def download_data(self, symbols: List[str], max_retries: int = 3) -> pd.DataFrame:
        """
        Load price data from local parquet files or download from Yahoo Finance
        
        Args:
            symbols: List of Yahoo Finance symbols
            max_retries: Maximum retry attempts for failed downloads
            
        Returns:
            DataFrame with Close prices, columns = symbols
        """
        print(f"Loading {len(symbols)} symbols from {self.start_date}...")
        
        successful_data = {}
        failed_symbols = []
        
        for symbol in symbols:
            # Try to load from local parquet files first
            if self._load_from_parquet(symbol, successful_data):
                continue
                
            # Fall back to Yahoo Finance download
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=self.start_date, auto_adjust=True, back_adjust=True)
                    
                    if not data.empty and len(data) > 100:  # Minimum data requirement
                        successful_data[symbol] = data['Close']
                        print(f"✓ {symbol}: {len(data)} observations (Yahoo Finance)")
                        break
                    else:
                        print(f"⚠ {symbol}: Insufficient data (attempt {attempt + 1})")
                        
                except Exception as e:
                    print(f"⚠ {symbol}: Error {e} (attempt {attempt + 1})")
                    
            else:
                failed_symbols.append(symbol)
                print(f"✗ {symbol}: Failed after {max_retries} attempts")
        
        if not successful_data:
            raise ValueError("No data could be loaded")
            
        # Combine into single DataFrame
        prices_df = pd.DataFrame(successful_data)
        
        # VECTORIZED: Clean data in single operation
        prices_df = (prices_df.fillna(method='ffill', limit=5)
                    .loc[:, prices_df.isnull().sum() / len(prices_df) < 0.1])
        
        print(f"\nSuccessfully loaded {len(prices_df.columns)} symbols")
        print(f"Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
        
        if failed_symbols:
            print(f"Failed symbols: {failed_symbols}")
            
        return prices_df
    
    def _load_from_parquet(self, symbol: str, successful_data: dict) -> bool:
        """
        Try to load symbol data from local parquet files
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'CL=F', 'EURUSD=X')
            successful_data: Dictionary to store successful data
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            # Convert Yahoo symbol to filename (e.g., CL=F -> CL_F.parquet)
            filename = symbol.replace("=", "_") + ".parquet"
            
            # Determine which directory to check
            if symbol.endswith("=F"):
                file_path = self.main_data_dir / "commodities" / filename
            elif symbol.endswith("=X"):
                file_path = self.main_data_dir / "forex" / filename
            else:
                return False
            
            if file_path.exists():
                # Load parquet file
                df = pd.read_parquet(file_path)
                
                # Filter by start date
                df = df[df.index >= self.start_date]
                
                if len(df) > 100:  # Minimum data requirement
                    # Use Close column if available, otherwise use the first numeric column
                    if 'Close' in df.columns:
                        successful_data[symbol] = df['Close']
                    else:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            successful_data[symbol] = df[numeric_cols[0]]
                        else:
                            return False
                    
                    print(f"✓ {symbol}: {len(df)} observations (parquet)")
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠ {symbol}: Error loading parquet: {e}")
            return False
    
    def load_all_data(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all price data with caching support
        
        Args:
            force_refresh: If True, ignore cache and download fresh data
            
        Returns:
            Tuple of (daily_prices, monthly_prices, daily_returns)
        """
        # Check if cached data exists and is recent
        if not force_refresh and self._cache_is_valid():
            print("Loading data from cache...")
            return self._load_from_cache()
        
        print("Cache not found or invalid, downloading fresh data...")
        
        # Download all symbols
        all_symbols = self.COM_FUT + self.FX_SPOT
        daily_prices = self.download_data(all_symbols)
        
        # Create monthly prices (last business day of month)
        monthly_prices = daily_prices.resample('M').last()
        
        # Calculate daily returns for volatility estimation
        daily_returns = daily_prices.pct_change().dropna()
        
        # Normalize FX signs for USD base pairs
        monthly_prices = self._normalize_fx_signs(monthly_prices)
        daily_prices = self._normalize_fx_signs(daily_prices)
        
        # Save to cache
        self._save_to_cache(daily_prices, monthly_prices, daily_returns)
        
        print(f"\nData preparation complete:")
        print(f"Daily prices: {daily_prices.shape}")
        print(f"Monthly prices: {monthly_prices.shape}")
        print(f"Daily returns: {daily_returns.shape}")
        
        return daily_prices, monthly_prices, daily_returns
    
    def _normalize_fx_signs(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize FX pair signs (vectorized)
        
        Args:
            prices: Price DataFrame
            
        Returns:
            DataFrame with normalized FX signs
        """
        # VECTORIZED: Find USD base pairs in columns and invert in single operation
        usd_pairs = [col for col in prices.columns if col in self.USD_BASE_PAIRS]
        if usd_pairs:
            prices[usd_pairs] = 1.0 / prices[usd_pairs]
            print(f"Inverted {len(usd_pairs)} USD base pairs")
        return prices
    
    def calculate_monthly_returns(self, monthly_prices: pd.DataFrame, 
                                excess_returns: bool = False) -> pd.DataFrame:
        """
        Calculate monthly returns (vectorized)
        
        Args:
            monthly_prices: Monthly price DataFrame
            excess_returns: Whether to calculate excess returns over T-bills
            
        Returns:
            Monthly returns DataFrame
        """
        monthly_returns = monthly_prices.pct_change().dropna()
        
        if excess_returns:
            try:
                tbill = yf.Ticker("^IRX")
                tbill_data = tbill.history(start=self.start_date, auto_adjust=True)
                
                if not tbill_data.empty:
                    # VECTORIZED: Apply T-bill adjustment using broadcast
                    tbill_monthly = (tbill_data['Close'].resample('M').last() / 1200
                                   ).reindex(monthly_returns.index, method='ffill')
                    monthly_returns = monthly_returns.sub(tbill_monthly, axis=0)
                    print("Applied T-bill adjustment (vectorized)")
                else:
                    print("T-bill data unavailable")
            except Exception as e:
                print(f"T-bill failed: {e}")
        
        return monthly_returns
    
    def get_universe_info(self) -> Dict[str, List[str]]:
        """
        Get information about the trading universe
        
        Returns:
            Dictionary with asset class breakdown
        """
        return {
            'commodities': self.COM_FUT,
            'forex': self.FX_SPOT,
            'usd_base_pairs': self.USD_BASE_PAIRS,
            'total_symbols': len(self.COM_FUT) + len(self.FX_SPOT)
        }
    
    def _cache_is_valid(self) -> bool:
        """
        Check if cached data exists and is reasonably recent
        
        Returns:
            True if cache is valid, False otherwise
        """
        cache_files = [self.daily_prices_cache, self.monthly_prices_cache, self.daily_returns_cache]
        
        # Check if all cache files exist
        if not all(f.exists() for f in cache_files):
            return False
        
        # Check if cache is not too old (more than 1 day)
        import time
        cache_age = time.time() - self.daily_prices_cache.stat().st_mtime
        max_age = 24 * 60 * 60  # 1 day in seconds
        
        if cache_age > max_age:
            print(f"Cache is {cache_age/3600:.1f} hours old, refreshing...")
            return False
        
        print(f"Using cache (age: {cache_age/3600:.1f} hours)")
        return True
    
    def _load_from_cache(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data from cache files
        
        Returns:
            Tuple of (daily_prices, monthly_prices, daily_returns)
        """
        try:
            daily_prices = pd.read_parquet(self.daily_prices_cache)
            monthly_prices = pd.read_parquet(self.monthly_prices_cache)
            daily_returns = pd.read_parquet(self.daily_returns_cache)
            
            print(f"Loaded from cache:")
            print(f"Daily prices: {daily_prices.shape}")
            print(f"Monthly prices: {monthly_prices.shape}")
            print(f"Daily returns: {daily_returns.shape}")
            
            return daily_prices, monthly_prices, daily_returns
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            raise
    
    def _save_to_cache(self, daily_prices: pd.DataFrame, 
                      monthly_prices: pd.DataFrame, 
                      daily_returns: pd.DataFrame) -> None:
        """
        Save data to cache files
        
        Args:
            daily_prices: Daily prices DataFrame
            monthly_prices: Monthly prices DataFrame
            daily_returns: Daily returns DataFrame
        """
        try:
            daily_prices.to_parquet(self.daily_prices_cache)
            monthly_prices.to_parquet(self.monthly_prices_cache)
            daily_returns.to_parquet(self.daily_returns_cache)
            
            print(f"Data saved to cache: {self.cache_dir}")
            
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
    
    def clear_cache(self) -> None:
        """
        Clear all cached data files
        """
        cache_files = [self.daily_prices_cache, self.monthly_prices_cache, self.daily_returns_cache]
        
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                print(f"Deleted: {cache_file}")
        
        print("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, str]:
        """
        Get information about cached data
        
        Returns:
            Dictionary with cache information
        """
        import time
        
        info = {}
        cache_files = {
            'daily_prices': self.daily_prices_cache,
            'monthly_prices': self.monthly_prices_cache,
            'daily_returns': self.daily_returns_cache
        }
        
        for name, path in cache_files.items():
            if path.exists():
                mtime = path.stat().st_mtime
                age_hours = (time.time() - mtime) / 3600
                info[name] = f"Exists (age: {age_hours:.1f}h)"
            else:
                info[name] = "Not cached"
        
        return info