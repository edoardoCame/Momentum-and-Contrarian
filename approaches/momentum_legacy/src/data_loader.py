"""
Data loading utilities for commodity momentum strategies.
Centralizes all data loading and preprocessing functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CommodityDataLoader:
    """
    Centralizes commodity data loading and preprocessing.
    """
    
    # Define commodity sectors
    SECTORS = {
        'Energy': ['CL=F', 'NG=F', 'BZ=F', 'RB=F', 'HO=F'],
        'Precious_Metals': ['GC=F', 'SI=F'],
        'Industrial_Metals': ['HG=F', 'PA=F'],
        'Agriculture': ['ZC=F', 'ZW=F', 'ZS=F', 'SB=F', 'CT=F', 'CC=F']
    }
    
    def __init__(self, data_dir: str = 'raw'):
        self.data_dir = Path(data_dir)
        self.price_data = None
        self.returns_data = None
        self.sectors_map = {}
        
        # Create reverse mapping: ticker -> sector
        for sector, tickers in self.SECTORS.items():
            for ticker in tickers:
                self.sectors_map[ticker] = sector
    
    def load_commodity_data(self) -> pd.DataFrame:
        """
        Load all commodity price data from parquet files.
        Returns DataFrame with adjusted close prices.
        """
        print("📊 Caricamento dati commodity...")
        
        parquet_files = list(self.data_dir.glob('*.parquet'))
        if not parquet_files:
            raise ValueError(f"Nessun file parquet trovato in {self.data_dir}")
        
        price_data = pd.DataFrame()
        
        for file_path in parquet_files:
            # Extract ticker from filename (CL_F.parquet -> CL=F)
            ticker = file_path.stem.replace('_', '=')
            
            try:
                df = pd.read_parquet(file_path)
                
                # Extract adjusted close price
                if ('Adj Close', ticker) in df.columns:
                    close_prices = df[('Adj Close', ticker)]
                elif ('Close', ticker) in df.columns:
                    close_prices = df[('Close', ticker)]
                else:
                    print(f"⚠️ Close price not found for {ticker}")
                    continue
                
                price_data[ticker] = close_prices
                print(f"✅ Loaded {ticker}")
                
            except Exception as e:
                print(f"❌ Error loading {ticker}: {e}")
                continue
        
        # Remove rows with all NaN
        price_data = price_data.dropna(how='all')
        
        print(f"📈 Loaded {len(price_data.columns)} commodities")
        print(f"📅 Period: {price_data.index.min().date()} to {price_data.index.max().date()}")
        print(f"📊 Total observations: {len(price_data):,}")
        
        self.price_data = price_data
        return price_data
    
    def calculate_returns(self, price_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        """
        if price_data is None:
            if self.price_data is None:
                self.load_commodity_data()
            price_data = self.price_data
        
        print("🔢 Calculating daily returns...")
        returns = price_data.pct_change().dropna(how='all')
        
        print(f"📊 Returns calculated for {len(returns.columns)} commodities")
        print(f"📅 Returns period: {returns.index.min().date()} to {returns.index.max().date()}")
        
        self.returns_data = returns
        return returns
    
    def get_sector_data(self, sector: str, data_type: str = 'returns') -> pd.DataFrame:
        """
        Get data for specific sector.
        
        Args:
            sector: Sector name (Energy, Precious_Metals, etc.)
            data_type: 'prices' or 'returns'
        """
        if sector not in self.SECTORS:
            raise ValueError(f"Unknown sector: {sector}")
        
        tickers = self.SECTORS[sector]
        
        if data_type == 'prices':
            if self.price_data is None:
                self.load_commodity_data()
            data = self.price_data
        else:  # returns
            if self.returns_data is None:
                self.calculate_returns()
            data = self.returns_data
        
        # Filter for tickers that exist in our data
        available_tickers = [t for t in tickers if t in data.columns]
        
        if not available_tickers:
            raise ValueError(f"No data available for sector {sector}")
        
        return data[available_tickers]
    
    def get_sector_mapping(self) -> Dict[str, str]:
        """Return ticker -> sector mapping."""
        return self.sectors_map.copy()
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers."""
        if self.price_data is None:
            self.load_commodity_data()
        return list(self.price_data.columns)
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the dataset."""
        if self.price_data is None:
            self.load_commodity_data()
        if self.returns_data is None:
            self.calculate_returns()
        
        summary = {
            'n_commodities': len(self.price_data.columns),
            'date_range': (self.price_data.index.min(), self.price_data.index.max()),
            'trading_days': len(self.price_data),
            'sectors': {sector: len(tickers) for sector, tickers in self.SECTORS.items()},
            'missing_data_pct': (self.price_data.isnull().sum() / len(self.price_data) * 100).to_dict(),
            'returns_stats': {
                'mean_daily_return': self.returns_data.mean().to_dict(),
                'daily_volatility': self.returns_data.std().to_dict()
            }
        }
        
        return summary

def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all data.
    Returns (prices, returns) DataFrames.
    """
    loader = CommodityDataLoader()
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns(prices)
    return prices, returns

if __name__ == "__main__":
    # Test the data loader
    loader = CommodityDataLoader()
    prices = loader.load_commodity_data()
    returns = loader.calculate_returns()
    
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    summary = loader.get_data_summary()
    print(f"📊 Commodities: {summary['n_commodities']}")
    print(f"📅 Period: {summary['date_range'][0].date()} to {summary['date_range'][1].date()}")
    print(f"📈 Trading days: {summary['trading_days']:,}")
    
    print("\n📋 Sectors:")
    for sector, count in summary['sectors'].items():
        print(f"  {sector}: {count} commodities")
    
    print("\n✅ Data loading test completed!")