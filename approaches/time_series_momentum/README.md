# Time Series Momentum (TSMOM) Strategy

A simple, vectorized implementation of Time Series Momentum strategies for Forex markets with both weekly and monthly rebalancing frequencies.

## Overview

This implementation focuses on **simplicity** and **vectorization** while maintaining strict adherence to preventing lookahead bias. The core momentum logic is straightforward: go long when past returns are positive, short when negative, using equal weighting across positions.

## Strategy Logic

**Core TSMOM Signal:**
```python
# Go long if past N-period return > 0, short if < 0
cum_returns = returns_df.rolling(window=lookback).apply(lambda x: (1 + x).prod() - 1).shift(1)
signals = np.where(cum_returns > 0, 1, np.where(cum_returns < 0, -1, 0))
```

**Key Features:**
- ✅ Pure return-based signals (no complex indicators)
- ✅ Strict lookahead bias prevention with `.shift(1)`
- ✅ Equal weighting across all active positions
- ✅ Vectorized implementation for performance
- ✅ Transaction cost modeling (5bp per trade)

## Strategy Configurations

### Weekly Rebalancing
- **TSMOM_4W**: 4-week momentum lookback
- **TSMOM_8W**: 8-week momentum lookback
- **TSMOM_12W**: 12-week momentum lookback

### Monthly Rebalancing
- **TSMOM_1M**: 1-month momentum lookback
- **TSMOM_3M**: 3-month momentum lookback
- **TSMOM_6M**: 6-month momentum lookback

## Directory Structure

```
time_series_momentum/
├── modules/
│   ├── data_loader.py          # Forex data loading and resampling
│   ├── tsmom_strategy.py       # TSMOM signal generation
│   ├── backtest_engine.py      # Vectorized backtesting
│   └── performance_utils.py    # Performance analysis and plotting
├── notebooks/
│   └── tsmom_analysis.ipynb    # Main analysis notebook
├── results/                    # Generated results and plots
└── README.md                   # This file
```

## Usage

### Quick Start
```python
from modules.data_loader import ForexDataLoader
from modules.tsmom_strategy import TSMOMStrategy
from modules.backtest_engine import TSMOMBacktestEngine

# Load data
loader = ForexDataLoader()
weekly_returns, monthly_returns = loader.prepare_data_for_backtest()

# Generate signals
tsmom = TSMOMStrategy()
weekly_signals = tsmom.calculate_tsmom_signals(weekly_returns, [4, 8, 12], 'weekly')

# Run backtest
engine = TSMOMBacktestEngine(transaction_cost_bps=5.0)
results = engine.run_all_backtests(weekly_signals, weekly_returns)
```

### Jupyter Notebook Analysis
Run the complete analysis:
```bash
cd notebooks/
jupyter notebook tsmom_analysis.ipynb
```

## Module Details

### data_loader.py
- Loads forex data from `../../data/forex/`
- Provides weekly (Friday) and monthly (month-end) resampling
- Handles missing data and data quality validation

### tsmom_strategy.py
- Generates TSMOM signals for different lookback periods
- Implements equal-weight portfolio construction
- Strict lookahead bias prevention with `.shift(1)`

### backtest_engine.py
- Vectorized backtesting with transaction costs
- Comprehensive performance metrics calculation
- Handles both gross and net return calculations

### performance_utils.py
- Performance visualization tools
- Equity curves, drawdown analysis, risk-return plots
- Summary statistics and comparison tables

## Data Requirements

The strategy uses forex data from the unified `data/forex/` directory:
- 20 major forex pairs in parquet format
- Daily OHLCV data from Yahoo Finance
- Automatic resampling to weekly/monthly frequencies

## Output Files

The analysis generates several output files in the `results/` directory:
- `tsmom_equity_curves.parquet` - All strategy equity curves
- `tsmom_returns.parquet` - Strategy return series
- `tsmom_performance_metrics.parquet` - Comprehensive metrics
- `tsmom_performance_summary.csv` - Formatted summary table
- `equity_curves.png` - Equity curve plots
- `performance_comparison.png` - Performance comparison charts
- `drawdown_analysis.png` - Drawdown analysis plots

## Performance Metrics

The framework calculates comprehensive performance metrics:
- **Return Metrics**: Total return, annualized return
- **Risk Metrics**: Volatility, Sharpe ratio, Sortino ratio
- **Drawdown Metrics**: Maximum drawdown, Calmar ratio
- **Cost Analysis**: Transaction cost impact on performance

## Key Implementation Notes

### Lookahead Bias Prevention
All signals use `.shift(1)` to ensure decisions are made on t-1 data:
```python
# Correct implementation
signals = calculate_momentum(returns).shift(1)  # Use t-1 for t decisions

# NEVER do this (lookahead bias)
signals = calculate_momentum(returns)  # Uses t data for t decisions
```

### Equal Weighting Implementation
```python
# Equal weight across active positions only
active_positions = np.abs(signals_df).sum(axis=1)
portfolio_returns = (signals_df * returns_df).sum(axis=1) / np.maximum(active_positions, 1)
```

### Transaction Costs
- 5 basis points per trade (entry/exit combined)
- Applied at portfolio level based on position turnover
- Realistic modeling of forex trading costs

## Assumptions and Limitations

**Assumptions:**
- Daily forex data availability and quality
- 5bp transaction costs (conservative estimate)
- Equal weighting feasibility across all pairs
- No funding costs or margin requirements

**Limitations:**
- Simple equal weighting (no risk-based sizing)
- No volatility targeting or risk management
- Basic transaction cost model
- No consideration of market microstructure

## Future Enhancements

Potential extensions while maintaining simplicity:
- Volatility-adjusted position sizing
- Simple risk parity weighting
- Additional lookback period combinations
- Extension to other asset classes (commodities)

---

**Note**: This implementation prioritizes simplicity, clarity, and vectorization over complex optimizations. It serves as a solid foundation for more advanced momentum strategies while maintaining educational value and research flexibility.