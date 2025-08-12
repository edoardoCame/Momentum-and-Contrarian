import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway

# Commodity sector definitions
COMMODITY_SECTORS = {
    'Energy': ['CL=F', 'NG=F', 'BZ=F', 'RB=F', 'HO=F'],
    'Grains': ['ZC=F', 'ZW=F', 'ZS=F', 'ZM=F', 'ZL=F', 'ZO=F'], 
    'Metals': ['GC=F', 'SI=F', 'PA=F', 'HG=F', 'PL=F'],
    'Livestock': ['LE=F', 'HE=F', 'GF=F'],
    'Softs': ['SB=F', 'CT=F', 'CC=F', 'KC=F', 'OJ=F']
}

# Season definitions (months)
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5], 
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

def extract_monthly_positions(monthly_prices: pd.DataFrame, 
                            strategy_params: Dict[str, int]) -> Dict[str, pd.DataFrame]:
    """
    Extract monthly positions for all strategies
    
    Args:
        monthly_prices: DataFrame with monthly commodity prices
        strategy_params: Dictionary with strategy names and their lookback periods
        
    Returns:
        Dictionary with strategy names and their position DataFrames
    """
    from monthly_strategy import monthly_contrarian_strategy
    
    positions_dict = {}
    
    for strategy_name, lookback in strategy_params.items():
        print(f"Extracting positions for {strategy_name} (lookback={lookback})...")
        
        # Run strategy to get positions
        _, positions = monthly_contrarian_strategy(monthly_prices, lookback)
        
        # Store positions
        positions_dict[strategy_name] = positions
        
        print(f"  ✓ Extracted {len(positions)} monthly position records")
    
    return positions_dict

def calculate_position_weights(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate normalized position weights for each commodity
    
    Args:
        positions: DataFrame with positions for each commodity
        
    Returns:
        DataFrame with normalized weights
    """
    # Calculate absolute weights
    abs_positions = positions.abs()
    
    # Normalize to get percentage weights
    weights = positions.div(abs_positions.sum(axis=1), axis=0)
    
    # Fill NaN with 0 (when no positions)
    weights = weights.fillna(0)
    
    return weights

def analyze_position_evolution(positions_dict: Dict[str, pd.DataFrame], 
                             commodity_names: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Analyze evolution of positions over time for key commodities
    
    Args:
        positions_dict: Dictionary with strategy positions
        commodity_names: List of commodities to focus on (None = all)
        
    Returns:
        Dictionary with evolution analysis for each strategy
    """
    evolution_dict = {}
    
    for strategy_name, positions in positions_dict.items():
        print(f"Analyzing position evolution for {strategy_name}...")
        
        if commodity_names is None:
            commodity_names = positions.columns.tolist()
        
        # Calculate weights
        weights = calculate_position_weights(positions)
        
        # Focus on specified commodities
        focused_weights = weights[commodity_names] if commodity_names else weights
        
        # Calculate additional metrics
        evolution_stats = pd.DataFrame(index=positions.index)
        
        # Number of long positions
        evolution_stats['n_long'] = (positions > 0).sum(axis=1)
        
        # Number of short positions  
        evolution_stats['n_short'] = (positions < 0).sum(axis=1)
        
        # Total absolute exposure
        evolution_stats['total_exposure'] = positions.abs().sum(axis=1)
        
        # Combine weights and stats
        evolution_analysis = pd.concat([focused_weights, evolution_stats], axis=1)
        
        evolution_dict[strategy_name] = evolution_analysis
        
        print(f"  ✓ Analysis complete: {len(evolution_analysis)} periods")
    
    return evolution_dict

def get_commodity_contributions(monthly_prices: pd.DataFrame, 
                              positions_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate each commodity's contribution to strategy returns
    
    Args:
        monthly_prices: Monthly price data
        positions_dict: Dictionary with strategy positions
        
    Returns:
        Dictionary with contribution analysis for each strategy
    """
    contributions_dict = {}
    
    # Calculate monthly returns
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    
    for strategy_name, positions in positions_dict.items():
        print(f"Calculating contributions for {strategy_name}...")
        
        # Align positions with returns (avoid lookahead bias)
        aligned_positions = positions.shift(1)
        
        # Calculate individual commodity contributions
        contributions = aligned_positions * monthly_returns
        
        # Fill NaN with 0
        contributions = contributions.fillna(0)
        
        # Add total strategy return
        contributions['total_strategy'] = contributions.sum(axis=1)
        
        contributions_dict[strategy_name] = contributions
        
        print(f"  ✓ Contributions calculated for {len(contributions.columns)-1} commodities")
    
    return contributions_dict

def analyze_2020_positions(positions_dict: Dict[str, pd.DataFrame],
                         contributions_dict: Dict[str, pd.DataFrame] = None,
                         start_date: str = '2019-01-01',
                         end_date: str = '2021-12-31') -> Dict[str, Dict]:
    """
    Detailed analysis of positions and performance during 2020 crisis
    
    Args:
        positions_dict: Dictionary with strategy positions
        contributions_dict: Dictionary with contribution data (optional)
        start_date: Start date for analysis period
        end_date: End date for analysis period
        
    Returns:
        Dictionary with detailed 2020 analysis for each strategy
    """
    analysis_2020 = {}
    
    for strategy_name, positions in positions_dict.items():
        print(f"Analyzing 2020 period for {strategy_name}...")
        
        # Filter to analysis period
        period_positions = positions.loc[start_date:end_date]
        
        strategy_analysis = {}
        
        # Key dates analysis
        key_dates = {
            'pre_crash': '2020-01-31',
            'crash_start': '2020-03-31', 
            'crash_bottom': '2020-04-30',
            'recovery': '2020-08-31',
            'year_end': '2020-12-31'
        }
        
        positions_by_date = {}
        for date_name, date_str in key_dates.items():
            if date_str in period_positions.index.strftime('%Y-%m-%d'):
                date_positions = period_positions.loc[date_str]
                positions_by_date[date_name] = {
                    'long_commodities': date_positions[date_positions > 0].to_dict(),
                    'short_commodities': date_positions[date_positions < 0].to_dict(),
                    'n_long': (date_positions > 0).sum(),
                    'n_short': (date_positions < 0).sum()
                }
        
        strategy_analysis['key_positions'] = positions_by_date
        
        # Oil-specific analysis (CL=F)
        if 'CL=F' in period_positions.columns:
            oil_positions = period_positions['CL=F']
            strategy_analysis['oil_analysis'] = {
                'was_long_2020': (oil_positions > 0).any(),
                'was_short_2020': (oil_positions < 0).any(),
                'position_changes': oil_positions.diff().abs().sum(),
                'avg_position': oil_positions.mean(),
                'max_position': oil_positions.max(),
                'min_position': oil_positions.min()
            }
        
        # Performance analysis if contributions available
        if contributions_dict and strategy_name in contributions_dict:
            period_contrib = contributions_dict[strategy_name].loc[start_date:end_date]
            
            # Top/bottom contributors for 2020
            contrib_2020 = period_contrib['2020-01-01':'2020-12-31']
            annual_contrib = contrib_2020.sum()
            
            strategy_analysis['performance_2020'] = {
                'total_return': annual_contrib['total_strategy'],
                'top_contributors': annual_contrib.nlargest(5).to_dict(),
                'bottom_contributors': annual_contrib.nsmallest(5).to_dict(),
                'oil_contribution': annual_contrib.get('CL=F', 0)
            }
        
        analysis_2020[strategy_name] = strategy_analysis
        print(f"  ✓ 2020 analysis complete")
    
    return analysis_2020

def create_position_heatmap_data(positions_dict: Dict[str, pd.DataFrame], 
                               strategy_name: str,
                               start_date: str = None,
                               end_date: str = None,
                               top_n_commodities: int = 15) -> pd.DataFrame:
    """
    Prepare data for position heatmap visualization
    
    Args:
        positions_dict: Dictionary with strategy positions
        strategy_name: Name of strategy to analyze
        start_date: Start date filter (None = from beginning)
        end_date: End date filter (None = to end)
        top_n_commodities: Number of most active commodities to include
        
    Returns:
        DataFrame suitable for heatmap plotting
    """
    if strategy_name not in positions_dict:
        raise ValueError(f"Strategy {strategy_name} not found in positions_dict")
    
    positions = positions_dict[strategy_name].copy()
    
    # Apply date filters
    if start_date:
        positions = positions.loc[start_date:]
    if end_date:
        positions = positions.loc[:end_date]
    
    # Find most active commodities (by total absolute position)
    commodity_activity = positions.abs().sum().sort_values(ascending=False)
    top_commodities = commodity_activity.head(top_n_commodities).index.tolist()
    
    # Filter to top commodities
    heatmap_data = positions[top_commodities].copy()
    
    # Resample to quarterly for better visualization (too many months for heatmap)
    quarterly_data = heatmap_data.resample('QE').mean()
    
    return quarterly_data

def get_contrarian_timing_analysis(monthly_prices: pd.DataFrame,
                                 positions_dict: Dict[str, pd.DataFrame],
                                 commodity: str = 'CL=F') -> Dict[str, pd.DataFrame]:
    """
    Analyze contrarian timing for specific commodity (especially oil)
    
    Args:
        monthly_prices: Monthly price data
        positions_dict: Dictionary with strategy positions  
        commodity: Commodity to analyze (default: CL=F for oil)
        
    Returns:
        Dictionary with timing analysis for each strategy
    """
    timing_analysis = {}
    
    if commodity not in monthly_prices.columns:
        print(f"Warning: {commodity} not found in price data")
        return timing_analysis
    
    # Get commodity price and returns
    commodity_prices = monthly_prices[commodity]
    commodity_returns = commodity_prices.pct_change(fill_method=None)
    
    for strategy_name, positions in positions_dict.items():
        if commodity not in positions.columns:
            continue
            
        print(f"Analyzing contrarian timing for {commodity} in {strategy_name}...")
        
        # Get positions for this commodity
        commodity_positions = positions[commodity]
        
        # Analyze timing
        analysis_df = pd.DataFrame(index=commodity_positions.index)
        analysis_df['price'] = commodity_prices
        analysis_df['returns'] = commodity_returns  
        analysis_df['position'] = commodity_positions
        analysis_df['lagged_position'] = commodity_positions.shift(1)  # Position taken previous period
        
        # Calculate if strategy was contrarian
        # Positive return followed by negative position = contrarian short after gains
        # Negative return followed by positive position = contrarian long after losses
        analysis_df['next_return'] = commodity_returns.shift(-1)  # Next period's return
        analysis_df['contrarian_long'] = (analysis_df['position'] > 0) & (analysis_df['returns'] < 0)
        analysis_df['contrarian_short'] = (analysis_df['position'] < 0) & (analysis_df['returns'] > 0)
        
        # Calculate success of contrarian bets
        analysis_df['contrarian_success'] = (
            (analysis_df['contrarian_long'] & (analysis_df['next_return'] > 0)) |
            (analysis_df['contrarian_short'] & (analysis_df['next_return'] < 0))
        )
        
        timing_analysis[strategy_name] = analysis_df
        
        print(f"  ✓ Timing analysis complete")
    
    return timing_analysis

def add_seasonal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonal columns to a DataFrame with datetime index
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added season and month columns
    """
    result = df.copy()
    result['month'] = result.index.month
    result['quarter'] = result.index.quarter
    
    # Add season column
    season_map = {}
    for season, months in SEASONS.items():
        for month in months:
            season_map[month] = season
    
    result['season'] = result['month'].map(season_map)
    return result

def analyze_seasonal_patterns(positions_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Analyze seasonal patterns in position allocation by commodity sector
    
    Args:
        positions_dict: Dictionary with strategy positions
        
    Returns:
        Dictionary with seasonal analysis for each strategy
    """
    seasonal_analysis = {}
    
    for strategy_name, positions in positions_dict.items():
        print(f"Analyzing seasonal patterns for {strategy_name}...")
        
        # Add seasonal columns
        positions_with_seasons = add_seasonal_columns(positions)
        
        # Initialize results storage
        sector_seasonal_stats = {}
        
        for sector_name, commodities in COMMODITY_SECTORS.items():
            # Filter to commodities that exist in this dataset
            available_commodities = [c for c in commodities if c in positions.columns]
            
            if not available_commodities:
                continue
                
            print(f"  Analyzing {sector_name} sector ({len(available_commodities)} commodities)...")
            
            # Calculate sector aggregate position (mean across commodities)
            sector_positions = positions[available_commodities].mean(axis=1)
            sector_positions_seasonal = add_seasonal_columns(sector_positions.to_frame('position'))
            
            # Calculate seasonal statistics
            seasonal_stats = {}
            
            for season in SEASONS.keys():
                season_data = sector_positions_seasonal[sector_positions_seasonal['season'] == season]['position']
                
                if len(season_data) > 0:
                    seasonal_stats[season] = {
                        'mean_position': season_data.mean(),
                        'median_position': season_data.median(),
                        'std_position': season_data.std(),
                        'pct_long': (season_data > 0).mean() * 100,
                        'pct_short': (season_data < 0).mean() * 100,
                        'pct_neutral': (season_data == 0).mean() * 100,
                        'n_observations': len(season_data),
                        'mean_abs_position': season_data.abs().mean()
                    }
                
            sector_seasonal_stats[sector_name] = seasonal_stats
        
        # Convert to DataFrame for easier analysis
        seasonal_df_list = []
        for sector, seasons_data in sector_seasonal_stats.items():
            for season, stats in seasons_data.items():
                row = {'sector': sector, 'season': season, **stats}
                seasonal_df_list.append(row)
        
        if seasonal_df_list:
            seasonal_df = pd.DataFrame(seasonal_df_list)
            seasonal_analysis[strategy_name] = seasonal_df
        
        print(f"  ✓ Seasonal analysis complete for {strategy_name}")
    
    return seasonal_analysis

def calculate_seasonal_statistics(seasonal_analysis: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Calculate statistical significance of seasonal patterns
    
    Args:
        seasonal_analysis: Output from analyze_seasonal_patterns
        
    Returns:
        Dictionary with statistical test results for each strategy
    """
    statistical_results = {}
    
    for strategy_name, seasonal_df in seasonal_analysis.items():
        print(f"Calculating seasonal statistics for {strategy_name}...")
        
        strategy_stats = {}
        
        # Test for each sector
        for sector in seasonal_df['sector'].unique():
            sector_data = seasonal_df[seasonal_df['sector'] == sector]
            
            if len(sector_data) < 4:  # Need at least 4 seasons
                continue
                
            sector_tests = {}
            
            # Test 1: ANOVA for mean position differences across seasons
            try:
                season_groups = [sector_data[sector_data['season'] == season]['mean_position'].values 
                               for season in SEASONS.keys() 
                               if season in sector_data['season'].values]
                
                # Filter out empty groups
                season_groups = [group for group in season_groups if len(group) > 0]
                
                if len(season_groups) >= 2:
                    f_stat, p_value_anova = f_oneway(*season_groups)
                    sector_tests['anova_mean_position'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value_anova,
                        'significant': p_value_anova < 0.05
                    }
            except Exception as e:
                print(f"    Warning: ANOVA failed for {sector}: {e}")
            
            # Test 2: Chi-square for long/short preferences by season
            try:
                # Create contingency table: seasons vs position direction
                contingency_data = []
                for season in SEASONS.keys():
                    season_row = sector_data[sector_data['season'] == season]
                    if len(season_row) > 0:
                        row_data = season_row.iloc[0]
                        # Convert percentages to approximate counts (assume 100 total observations)
                        long_count = int(row_data['pct_long'] * row_data['n_observations'] / 100)
                        short_count = int(row_data['pct_short'] * row_data['n_observations'] / 100)
                        neutral_count = row_data['n_observations'] - long_count - short_count
                        contingency_data.append([long_count, short_count, max(0, neutral_count)])
                
                if len(contingency_data) >= 2:
                    contingency_table = np.array(contingency_data)
                    if contingency_table.sum() > 0 and contingency_table.min() >= 0:
                        chi2_stat, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
                        sector_tests['chi2_direction'] = {
                            'chi2_statistic': chi2_stat,
                            'p_value': p_value_chi2,
                            'degrees_of_freedom': dof,
                            'significant': p_value_chi2 < 0.05
                        }
            except Exception as e:
                print(f"    Warning: Chi-square test failed for {sector}: {e}")
            
            # Test 3: Coefficient of variation across seasons (measure of seasonal variability)
            try:
                seasonal_means = sector_data.groupby('season')['mean_position'].mean()
                if len(seasonal_means) > 1:
                    cv = seasonal_means.std() / abs(seasonal_means.mean()) if seasonal_means.mean() != 0 else np.inf
                    sector_tests['coefficient_variation'] = {
                        'cv': cv,
                        'seasonal_means': seasonal_means.to_dict(),
                        'high_variability': cv > 0.5  # Threshold for high seasonal variability
                    }
            except Exception as e:
                print(f"    Warning: CV calculation failed for {sector}: {e}")
                
            if sector_tests:
                strategy_stats[sector] = sector_tests
        
        statistical_results[strategy_name] = strategy_stats
        print(f"  ✓ Statistical analysis complete for {strategy_name}")
    
    return statistical_results

def seasonal_sector_analysis(positions_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Detailed analysis of seasonal preferences by sector
    
    Args:
        positions_dict: Dictionary with strategy positions
        
    Returns:
        Dictionary with detailed seasonal sector analysis
    """
    sector_analysis = {}
    
    for strategy_name, positions in positions_dict.items():
        print(f"Detailed seasonal sector analysis for {strategy_name}...")
        
        # Add seasonal information
        positions_seasonal = add_seasonal_columns(positions)
        
        # Create analysis matrix: Sectors x Seasons
        analysis_matrix = pd.DataFrame(index=COMMODITY_SECTORS.keys(), 
                                     columns=SEASONS.keys())
        
        # Calculate statistics for each sector-season combination
        detailed_stats = {}
        
        for sector_name, commodities in COMMODITY_SECTORS.items():
            available_commodities = [c for c in commodities if c in positions.columns]
            
            if not available_commodities:
                continue
                
            sector_detailed = {}
            
            for season_name in SEASONS.keys():
                # Filter data for this season
                season_mask = positions_seasonal['season'] == season_name
                season_positions = positions_seasonal[season_mask]
                
                if len(season_positions) > 0:
                    # Calculate sector aggregate for this season
                    sector_positions_season = season_positions[available_commodities]
                    
                    # Statistics
                    mean_sector_position = sector_positions_season.mean(axis=1).mean()
                    median_sector_position = sector_positions_season.mean(axis=1).median()
                    
                    # Direction preferences
                    sector_means = sector_positions_season.mean(axis=1)
                    pct_long_months = (sector_means > 0).mean() * 100
                    pct_short_months = (sector_means < 0).mean() * 100
                    
                    # Store in analysis matrix
                    analysis_matrix.loc[sector_name, season_name] = mean_sector_position
                    
                    # Detailed statistics
                    sector_detailed[season_name] = {
                        'mean_position': mean_sector_position,
                        'median_position': median_sector_position,
                        'pct_long_months': pct_long_months,
                        'pct_short_months': pct_short_months,
                        'n_months': len(season_positions),
                        'commodities_count': len(available_commodities),
                        'position_intensity': sector_positions_season.abs().mean(axis=1).mean()
                    }
                    
            detailed_stats[sector_name] = sector_detailed
        
        # Convert analysis matrix to numeric
        analysis_matrix = analysis_matrix.astype(float)
        
        # Store results
        sector_analysis[strategy_name] = {
            'matrix': analysis_matrix,
            'detailed_stats': detailed_stats
        }
        
        print(f"  ✓ Detailed sector analysis complete for {strategy_name}")
    
    return sector_analysis

def create_seasonal_heatmap_data(seasonal_analysis: Dict[str, pd.DataFrame], 
                               metric: str = 'mean_position') -> Dict[str, pd.DataFrame]:
    """
    Prepare heatmap data for seasonal analysis visualization
    
    Args:
        seasonal_analysis: Output from analyze_seasonal_patterns
        metric: Metric to use for heatmap ('mean_position', 'pct_long', etc.)
        
    Returns:
        Dictionary with heatmap data for each strategy
    """
    heatmap_data = {}
    
    for strategy_name, seasonal_df in seasonal_analysis.items():
        # Pivot to create heatmap format: sectors as rows, seasons as columns
        heatmap = seasonal_df.pivot(index='sector', columns='season', values=metric)
        
        # Reorder columns to follow natural seasonal order
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        heatmap = heatmap.reindex(columns=[s for s in season_order if s in heatmap.columns])
        
        heatmap_data[strategy_name] = heatmap
    
    return heatmap_data

if __name__ == "__main__":
    # Example usage - would need to import other modules
    from data_loader import load_commodity_data
    from monthly_strategy import prepare_monthly_data
    
    print("Loading data for position analysis...")
    commodity_data = load_commodity_data(data_dir='../data/raw')
    monthly_prices = prepare_monthly_data(commodity_data)
    
    # Test strategy parameters
    strategy_params = {
        '6M Lookback': 6,
        '9M Lookback': 9
    }
    
    print("Extracting positions...")
    positions_dict = extract_monthly_positions(monthly_prices, strategy_params)
    
    print("Analyzing seasonal patterns...")
    seasonal_analysis = analyze_seasonal_patterns(positions_dict)
    
    print("Position analysis module test complete!")