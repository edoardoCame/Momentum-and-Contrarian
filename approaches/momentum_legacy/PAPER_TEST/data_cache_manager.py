#!/usr/bin/env python3
"""
Data Cache Manager for Commodity Cross-Sectional Backtest System

Handles caching and loading of price data, returns, and strategy results
to avoid re-downloading and re-calculating everything from scratch.

Features:
- Intelligent data caching with metadata tracking
- Automatic freshness checks
- Resume functionality for interrupted calculations
- Data integrity validation
- Cache invalidation and cleanup utilities

🚀 Generated with Claude Code
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import warnings
warnings.filterwarnings('ignore')

class DataCacheManager:
    """
    Manages caching and loading of data for the commodity backtest system.
    """
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize the cache manager.
        
        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.cache_dir = self.base_dir / "cache"
        self.results_dir = self.base_dir / "results"
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.cache_dir, self.cache_dir / "intermediate_results"]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.daily_prices_file = self.data_dir / "daily_prices.parquet"
        self.monthly_prices_file = self.data_dir / "monthly_prices.parquet"  
        self.monthly_returns_file = self.data_dir / "monthly_returns.parquet"
        self.download_metadata_file = self.data_dir / "download_metadata.json"
        self.strategy_cache_file = self.cache_dir / "strategy_cache.json"
        
        print(f"📁 Data Cache Manager initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Cache directory: {self.cache_dir}")
    
    # ==========================================================================
    # DATA CACHING FUNCTIONS
    # ==========================================================================
    
    def save_price_data(self, daily_prices: pd.DataFrame, monthly_prices: pd.DataFrame, 
                       monthly_returns: pd.DataFrame, tickers: List[str], 
                       start_date: str, end_date: str) -> None:
        """
        Save price and return data to cache.
        
        Args:
            daily_prices: Daily price DataFrame
            monthly_prices: Monthly price DataFrame  
            monthly_returns: Monthly returns DataFrame
            tickers: List of tickers downloaded
            start_date: Start date for download
            end_date: End date for download
        """
        print("💾 Saving price data to cache...")
        
        # Save data files
        daily_prices.to_parquet(self.daily_prices_file)
        monthly_prices.to_parquet(self.monthly_prices_file)
        monthly_returns.to_parquet(self.monthly_returns_file)
        
        # Create metadata
        metadata = {
            'download_timestamp': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'tickers_requested': tickers,
            'tickers_successful': list(daily_prices.columns),
            'tickers_failed': [t for t in tickers if t not in daily_prices.columns],
            'data_summary': {
                'daily_observations': len(daily_prices),
                'monthly_observations': len(monthly_prices), 
                'return_observations': len(monthly_returns),
                'daily_date_range': [daily_prices.index.min().isoformat(), 
                                    daily_prices.index.max().isoformat()],
                'monthly_date_range': [monthly_prices.index.min().isoformat(),
                                     monthly_prices.index.max().isoformat()],
                'data_completeness': {
                    ticker: float(1 - monthly_returns[ticker].isna().mean()) 
                    for ticker in monthly_returns.columns
                }
            },
            'file_hashes': {
                'daily_prices': self._calculate_file_hash(self.daily_prices_file),
                'monthly_prices': self._calculate_file_hash(self.monthly_prices_file),
                'monthly_returns': self._calculate_file_hash(self.monthly_returns_file)
            }
        }
        
        # Save metadata
        with open(self.download_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Cached {len(daily_prices.columns)} commodities")
        print(f"   Daily: {len(daily_prices)} observations")
        print(f"   Monthly: {len(monthly_prices)} observations")
        print(f"   Returns: {len(monthly_returns)} observations")
        
    def load_price_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], 
                                     Optional[pd.DataFrame], Optional[Dict]]:
        """
        Load cached price data if available.
        
        Returns:
            Tuple of (daily_prices, monthly_prices, monthly_returns, metadata)
            Returns None for missing data
        """
        print("📥 Checking for cached price data...")
        
        # Check if all required files exist
        required_files = [
            self.daily_prices_file,
            self.monthly_prices_file, 
            self.monthly_returns_file,
            self.download_metadata_file
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            print(f"❌ Missing cache files: {[f.name for f in missing_files]}")
            return None, None, None, None
        
        try:
            # Load data
            daily_prices = pd.read_parquet(self.daily_prices_file)
            monthly_prices = pd.read_parquet(self.monthly_prices_file)
            monthly_returns = pd.read_parquet(self.monthly_returns_file)
            
            # Load metadata
            with open(self.download_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Validate data integrity
            if not self._validate_data_integrity(metadata):
                print("⚠️ Data integrity check failed - cache may be corrupted")
                return None, None, None, None
            
            print(f"✅ Loaded cached data:")
            print(f"   Downloaded: {metadata['download_timestamp']}")
            print(f"   Period: {metadata['start_date']} to {metadata['end_date']}")
            print(f"   Commodities: {len(metadata['tickers_successful'])}")
            
            return daily_prices, monthly_prices, monthly_returns, metadata
            
        except Exception as e:
            print(f"❌ Error loading cached data: {str(e)}")
            return None, None, None, None
    
    def is_data_fresh(self, max_age_hours: int = 24) -> bool:
        """
        Check if cached data is fresh enough.
        
        Args:
            max_age_hours: Maximum age in hours before data is considered stale
            
        Returns:
            True if data is fresh, False otherwise
        """
        if not self.download_metadata_file.exists():
            return False
        
        try:
            with open(self.download_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            download_time = datetime.fromisoformat(metadata['download_timestamp'])
            age_hours = (datetime.now() - download_time).total_seconds() / 3600
            
            is_fresh = age_hours < max_age_hours
            print(f"📅 Data age: {age_hours:.1f} hours ({'fresh' if is_fresh else 'stale'})")
            
            return is_fresh
            
        except Exception as e:
            print(f"❌ Error checking data freshness: {str(e)}")
            return False
    
    # ==========================================================================
    # STRATEGY CACHE FUNCTIONS
    # ==========================================================================
    
    def save_strategy_cache(self, cache_data: Dict[str, Any]) -> None:
        """
        Save strategy cache data.
        
        Args:
            cache_data: Dictionary containing strategy results and metadata
        """
        print("💾 Saving strategy cache...")
        
        cache_data['cache_timestamp'] = datetime.now().isoformat()
        
        with open(self.strategy_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        print(f"✅ Strategy cache saved with {len(cache_data.get('completed_strategies', []))} strategies")
    
    def load_strategy_cache(self) -> Optional[Dict[str, Any]]:
        """
        Load strategy cache data.
        
        Returns:
            Cache data dictionary or None if not available
        """
        if not self.strategy_cache_file.exists():
            return None
        
        try:
            with open(self.strategy_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            print(f"📥 Loaded strategy cache from: {cache_data.get('cache_timestamp', 'unknown')}")
            return cache_data
            
        except Exception as e:
            print(f"❌ Error loading strategy cache: {str(e)}")
            return None
    
    def get_completed_strategies(self) -> List[str]:
        """
        Get list of completed strategies from cache.
        
        Returns:
            List of completed strategy keys
        """
        cache_data = self.load_strategy_cache()
        if cache_data is None:
            return []
        
        return cache_data.get('completed_strategies', [])
    
    def is_strategy_completed(self, strategy_key: str) -> bool:
        """
        Check if a specific strategy has been completed.
        
        Args:
            strategy_key: Strategy identifier (e.g., "momentum_R3_H6_long_short")
            
        Returns:
            True if strategy is completed, False otherwise
        """
        completed = self.get_completed_strategies()
        return strategy_key in completed
    
    def mark_strategy_completed(self, strategy_key: str) -> None:
        """
        Mark a strategy as completed in the cache.
        
        Args:
            strategy_key: Strategy identifier
        """
        cache_data = self.load_strategy_cache() or {'completed_strategies': []}
        
        if strategy_key not in cache_data['completed_strategies']:
            cache_data['completed_strategies'].append(strategy_key)
            self.save_strategy_cache(cache_data)
    
    # ==========================================================================
    # DATA VALIDATION AND UTILITIES
    # ==========================================================================
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        if not file_path.exists():
            return ""
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _validate_data_integrity(self, metadata: Dict) -> bool:
        """
        Validate that cached data files haven't been corrupted.
        
        Args:
            metadata: Metadata dictionary containing file hashes
            
        Returns:
            True if data is intact, False otherwise
        """
        file_hash_mapping = {
            'daily_prices': self.daily_prices_file,
            'monthly_prices': self.monthly_prices_file,
            'monthly_returns': self.monthly_returns_file
        }
        
        stored_hashes = metadata.get('file_hashes', {})
        
        for file_key, file_path in file_hash_mapping.items():
            if file_key in stored_hashes:
                current_hash = self._calculate_file_hash(file_path)
                stored_hash = stored_hashes[file_key]
                
                if current_hash != stored_hash:
                    print(f"⚠️ Hash mismatch for {file_key}: {current_hash} != {stored_hash}")
                    return False
        
        return True
    
    def get_cache_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of cached data and strategies.
        
        Returns:
            Dictionary with cache summary information
        """
        summary = {
            'data_cache': {
                'exists': self.download_metadata_file.exists(),
                'fresh': self.is_data_fresh() if self.download_metadata_file.exists() else False,
                'metadata': None
            },
            'strategy_cache': {
                'exists': self.strategy_cache_file.exists(),
                'completed_count': len(self.get_completed_strategies()),
                'completed_strategies': self.get_completed_strategies()
            },
            'files': {
                'daily_prices': {
                    'exists': self.daily_prices_file.exists(),
                    'size_mb': self.daily_prices_file.stat().st_size / 1024 / 1024 if self.daily_prices_file.exists() else 0
                },
                'monthly_prices': {
                    'exists': self.monthly_prices_file.exists(),
                    'size_mb': self.monthly_prices_file.stat().st_size / 1024 / 1024 if self.monthly_prices_file.exists() else 0
                },
                'monthly_returns': {
                    'exists': self.monthly_returns_file.exists(),
                    'size_mb': self.monthly_returns_file.stat().st_size / 1024 / 1024 if self.monthly_returns_file.exists() else 0
                }
            }
        }
        
        # Load metadata if available
        if self.download_metadata_file.exists():
            try:
                with open(self.download_metadata_file, 'r') as f:
                    summary['data_cache']['metadata'] = json.load(f)
            except Exception as e:
                summary['data_cache']['metadata'] = f"Error loading: {str(e)}"
        
        return summary
    
    def print_cache_status(self) -> None:
        """Print detailed cache status report."""
        print("\n" + "=" * 60)
        print("📊 CACHE STATUS REPORT")
        print("=" * 60)
        
        summary = self.get_cache_summary()
        
        # Data cache status
        data_cache = summary['data_cache']
        print(f"📥 Data Cache:")
        print(f"   Exists: {'✅' if data_cache['exists'] else '❌'}")
        print(f"   Fresh: {'✅' if data_cache['fresh'] else '❌' if data_cache['exists'] else 'N/A'}")
        
        if data_cache['exists'] and data_cache['metadata']:
            metadata = data_cache['metadata']
            if isinstance(metadata, dict):
                print(f"   Downloaded: {metadata.get('download_timestamp', 'Unknown')}")
                print(f"   Period: {metadata.get('start_date')} to {metadata.get('end_date')}")
                print(f"   Commodities: {len(metadata.get('tickers_successful', []))}")
                
                if metadata.get('tickers_failed'):
                    print(f"   Failed tickers: {metadata['tickers_failed']}")
        
        # Strategy cache status
        strategy_cache = summary['strategy_cache']
        print(f"\n🔄 Strategy Cache:")
        print(f"   Exists: {'✅' if strategy_cache['exists'] else '❌'}")
        print(f"   Completed strategies: {strategy_cache['completed_count']}")
        
        # File sizes
        files = summary['files']
        print(f"\n💾 Cache Files:")
        for file_name, file_info in files.items():
            status = '✅' if file_info['exists'] else '❌'
            size = f"{file_info['size_mb']:.1f} MB" if file_info['exists'] else "0 MB"
            print(f"   {file_name}: {status} ({size})")
        
        total_size = sum(f['size_mb'] for f in files.values() if f['exists'])
        print(f"\n📊 Total cache size: {total_size:.1f} MB")
    
    def clear_cache(self, data_only: bool = False, strategy_only: bool = False) -> None:
        """
        Clear cache files.
        
        Args:
            data_only: Only clear data cache
            strategy_only: Only clear strategy cache  
        """
        if not data_only and not strategy_only:
            # Clear everything
            files_to_remove = [
                self.daily_prices_file,
                self.monthly_prices_file,
                self.monthly_returns_file,
                self.download_metadata_file,
                self.strategy_cache_file
            ]
            print("🗑️ Clearing all cache files...")
        elif data_only:
            files_to_remove = [
                self.daily_prices_file,
                self.monthly_prices_file,
                self.monthly_returns_file,
                self.download_metadata_file
            ]
            print("🗑️ Clearing data cache files...")
        else:  # strategy_only
            files_to_remove = [self.strategy_cache_file]
            print("🗑️ Clearing strategy cache files...")
        
        removed_count = 0
        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
                removed_count += 1
        
        print(f"✅ Removed {removed_count} cache files")
    
    def recommend_actions(self) -> List[str]:
        """
        Recommend actions based on current cache status.
        
        Returns:
            List of recommended actions
        """
        recommendations = []
        summary = self.get_cache_summary()
        
        # Data cache recommendations
        if not summary['data_cache']['exists']:
            recommendations.append("📥 Download fresh data from Yahoo Finance")
        elif not summary['data_cache']['fresh']:
            recommendations.append("🔄 Update data cache (data is stale)")
        
        # Strategy cache recommendations  
        if not summary['strategy_cache']['exists']:
            recommendations.append("🚀 Run initial strategy backtest")
        elif summary['strategy_cache']['completed_count'] < 360:  # Expected total strategies
            recommendations.append(f"⚡ Resume backtest ({summary['strategy_cache']['completed_count']}/360 completed)")
        
        # File size recommendations
        total_size = sum(f['size_mb'] for f in summary['files'].values() if f['exists'])
        if total_size > 100:  # MB
            recommendations.append("🧹 Consider cache cleanup if disk space is limited")
        
        if not recommendations:
            recommendations.append("✅ Cache is optimal - ready to use!")
        
        return recommendations

# =============================================================================
# CONVENIENCE FUNCTIONS  
# =============================================================================

def get_cache_manager(base_dir: str = ".") -> DataCacheManager:
    """Get a DataCacheManager instance."""
    return DataCacheManager(base_dir)

def quick_cache_check(base_dir: str = ".") -> None:
    """Perform a quick cache status check."""
    manager = get_cache_manager(base_dir)
    manager.print_cache_status()
    
    print("\n💡 RECOMMENDATIONS:")
    recommendations = manager.recommend_actions()
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    # Demo of cache manager functionality
    print("🧪 TESTING DATA CACHE MANAGER")
    print("=" * 50)
    
    manager = DataCacheManager()
    manager.print_cache_status()
    
    print("\n💡 Recommendations:")
    recommendations = manager.recommend_actions()
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")