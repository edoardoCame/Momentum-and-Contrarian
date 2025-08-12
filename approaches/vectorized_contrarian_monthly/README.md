# Monthly Contrarian Commodity Strategy

A vectorized implementation of a monthly contrarian trading strategy on commodity futures.

## Strategy Overview

- **Universe**: 24 most liquid commodity futures from yfinance
- **Strategy**: Quintile-based contrarian approach
  - Long bottom quintile (worst performing commodities over lookback period)
  - Short top quintile (best performing commodities over lookback period)  
- **Rebalancing**: Monthly (month-end)
- **Weighting**: Equal weight within each quintile
- **Lookback**: Optimizable parameter (3-24 months tested)
- **Data Period**: 2000-01-01 to 2025-08-08

## Commodity Universe

### Energy (5)
- CL=F (Crude Oil WTI)
- NG=F (Natural Gas)
- BZ=F (Brent Crude Oil)
- RB=F (RBOB Gasoline)
- HO=F (Heating Oil)

### Metals (5)
- GC=F (Gold)
- SI=F (Silver) 
- PA=F (Palladium)
- HG=F (Copper)
- PL=F (Platinum)

### Grains (6)
- ZC=F (Corn)
- ZW=F (Wheat)
- ZS=F (Soybeans)
- ZM=F (Soybean Meal)
- ZL=F (Soybean Oil)
- ZO=F (Oats)

### Livestock (3)
- LE=F (Live Cattle)
- HE=F (Lean Hogs)
- GF=F (Feeder Cattle)

### Softs (5)
- SB=F (Sugar)
- CT=F (Cotton)
- CC=F (Cocoa)
- KC=F (Coffee)
- OJ=F (Orange Juice)

## Quick Start

### 1. Download Data
```bash
cd modules
python3 data_loader.py
```

### 2. Run Strategy Test
```bash
python3 monthly_strategy.py
```

### 3. Full Optimization Analysis
```bash
jupyter notebook notebooks/optimization_analysis.ipynb
```

## Key Results (Preview)

Based on initial parameter sweep:
- **Best Performance**: 6-month lookback period
- **Total Return**: ~38.7% (over full period)
- **Max Drawdown**: ~-37.7%
- **Data Coverage**: 308 months (~25.7 years)

## Files Structure

```
vectorized_contrarian_monthly/
├── data/raw/                           # Cached commodity data (.parquet)
├── modules/
│   ├── data_loader.py                 # yfinance download & caching
│   └── monthly_strategy.py            # Core strategy implementation
├── notebooks/
│   └── optimization_analysis.ipynb    # Parameter optimization & analysis
└── README.md                          # This file
```

## Implementation Features

- **Vectorized Operations**: Pure pandas/numpy for performance
- **No Lookahead Bias**: Uses `.shift(1)` for signal generation  
- **Robust Data Handling**: Handles missing data gracefully
- **Caching System**: Avoids re-downloading data
- **Minimal Code**: Each module under 100 lines
- **Parameter Optimization**: Systematic testing of lookback periods

## Strategy Logic

1. **Monthly Resampling**: Convert daily data to month-end close prices
2. **Performance Calculation**: Rolling N-month returns for each commodity
3. **Quintile Ranking**: Rank commodities by performance percentiles (0-1)
4. **Position Generation**: 
   - Long positions: Bottom 20% (quintile ≤ 0.2)
   - Short positions: Top 20% (quintile ≥ 0.8)
5. **Equal Weighting**: Within each quintile
6. **Monthly Rebalancing**: Apply new positions at month-end

## Usage Examples

```python
from data_loader import load_commodity_data
from monthly_strategy import prepare_monthly_data, monthly_contrarian_strategy

# Load data
commodity_data = load_commodity_data()
monthly_prices = prepare_monthly_data(commodity_data)

# Run strategy
results, positions = monthly_contrarian_strategy(monthly_prices, lookback_months=6)

# View results
print(f"Total Return: {results['cumulative_returns'].iloc[-1] - 1:.1%}")
```

This implementation prioritizes simplicity, vectorization, and educational clarity while maintaining professional-grade bias prevention and robust data handling.