import pandas as pd
import yfinance as yf
import os
from pathlib import Path

def get_commodity_tickers():
    """Return the 25 most liquid commodity futures tickers for yfinance"""
    return [
        # Energy (5)
        'CL=F', 'NG=F', 'BZ=F', 'RB=F', 'HO=F',
        
        # Precious Metals (3) 
        'GC=F', 'SI=F', 'PA=F',
        
        # Industrial Metals (2)
        'HG=F', 'PL=F',
        
        # Grains (6)
        'ZC=F', 'ZW=F', 'ZS=F', 'ZM=F', 'ZL=F', 'ZO=F',
        
        # Livestock (3)
        'LE=F', 'HE=F', 'GF=F',
        
        # Softs (5)
        'SB=F', 'CT=F', 'CC=F', 'KC=F', 'OJ=F'
    ]

def download_commodity_data(tickers=None, start_date='2000-01-01', end_date='2025-08-08', data_dir='../data/raw'):
    """Download commodity futures data and save as parquet files"""
    if tickers is None:
        tickers = get_commodity_tickers()
    
    # Ensure directory exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    successful_downloads = []
    failed_downloads = []
    
    print(f"Downloading {len(tickers)} commodity futures from {start_date} to {end_date}...")
    
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker.replace('=', '_')}.parquet")
        
        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"✓ {ticker} - cached")
            successful_downloads.append(ticker)
            continue
            
        try:
            print(f"Downloading {ticker}...", end=' ')
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            
            if data.empty:
                print("✗ No data")
                failed_downloads.append(ticker)
                continue
                
            # Save as parquet
            data.to_parquet(file_path)
            print(f"✓ {len(data)} records")
            successful_downloads.append(ticker)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_downloads.append(ticker)
    
    print(f"\nDownload Summary:")
    print(f"✓ Successful: {len(successful_downloads)}")
    print(f"✗ Failed: {len(failed_downloads)}")
    if failed_downloads:
        print(f"Failed tickers: {failed_downloads}")
    
    return successful_downloads, failed_downloads

def load_commodity_data(tickers=None, data_dir='../data/raw'):
    """Load cached commodity data from parquet files"""
    if tickers is None:
        tickers = get_commodity_tickers()
    
    commodity_data = {}
    
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker.replace('=', '_')}.parquet")
        
        if os.path.exists(file_path):
            try:
                data = pd.read_parquet(file_path)
                if not data.empty:
                    commodity_data[ticker] = data
                    print(f"✓ Loaded {ticker}: {len(data)} records")
            except Exception as e:
                print(f"✗ Error loading {ticker}: {e}")
        else:
            print(f"✗ File not found: {ticker}")
    
    print(f"\nLoaded {len(commodity_data)} commodity datasets")
    return commodity_data

if __name__ == "__main__":
    # Download all commodity data
    successful, failed = download_commodity_data()
    
    # Load and verify data
    data = load_commodity_data(successful)