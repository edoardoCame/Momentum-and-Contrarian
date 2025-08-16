# Contrarian Strategy - Monthly Framework

A simple, vectorized implementation of Contrarian strategies for Forex markets with **monthly rebalancing only** for temporal consistency and coherent holding periods.

## Overview

This implementation focuses on **monthly rebalancing consistency**, **simplicity** and **vectorization** while maintaining strict adherence to preventing lookahead bias. The strategy uses a **1-month holding period** aligned with monthly rebalancing for temporal coherence and implements contrarian (mean reversion) logic.

## Strategy Logic

### Core Contrarian Signal Generation
```python
# Monthly contrarian with strict lookahead prevention
past_performance = returns.rolling(window=lookback).sum().shift(1)  # T-1 data for T decisions
signals = -np.sign(past_performance).fillna(0)  # +1 (Long losers), -1 (Short winners), 0 (No position)
```

### Entry and Exit Rules

**Entry Rules (End of Month):**
1. **Signal Date**: Last trading day of each month
2. **Data Used**: Past N months of returns (T-N to T-1)
3. **Position Decision**: Long if cumulative return < 0, Short if > 0 (contrarian)
4. **Portfolio Weight**: Equal weight across all active positions
5. **Execution**: Enter at month-end closing prices

**Exit Rules (Next Month End):**
1. **Holding Period**: Exactly 1 month
2. **Exit Timing**: Last trading day of following month
3. **No Intra-Month Changes**: Positions held for complete month
4. **Automatic Rebalancing**: All positions re-evaluated monthly

### Temporal Consistency
- **Rebalancing Frequency**: Monthly only (no weekly mixing)
- **Holding Period**: 1 month (aligned with rebalancing)
- **Lookahead Prevention**: `.shift(1)` ensures T-1 data for T decisions
- **Signal-to-Execution**: Same month-end timing for all assets

**Key Features:**
- ✅ **Monthly-Only Focus**: Coherent 1-month holding periods
- ✅ **Strict Lookahead Prevention**: `.shift(1)` on all signals
- ✅ **Equal Weight Rebalancing**: Monthly equal weight across active positions
- ✅ **Vectorized Implementation**: Efficient pandas operations
- ✅ **No Transaction Costs**: Pure strategy performance without friction

## Strategy Configurations

### Monthly Contrarian Strategies (All with 1-Month Holding Period)
- **CONTRARIAN_1M**: 1-month lookback contrarian
- **CONTRARIAN_3M**: 3-month lookback contrarian  
- **CONTRARIAN_6M**: 6-month lookback contrarian
- **CONTRARIAN_12M**: 12-month lookback contrarian

### Detailed Timeline Example

**January 31st (Signal Generation & Entry):**
```
For CONTRARIAN_3M:
- Lookback Data: October + November + December returns
- Signal Calculation: -sign(sum(Oct_ret + Nov_ret + Dec_ret)).shift(1)
- Action: Enter contrarian positions for February holding period
- Portfolio: Equal weight across all Long/Short signals
```

**February 28th (Exit & New Signal Generation):**
```
- Exit: Close all January positions at month-end
- New Lookback: November + December + January returns  
- New Signals: Generate signals for March holding period
- Rebalancing: Equal weight across new active positions
```

**March 31st (Continue Monthly Cycle):**
```
- Exit: Close February positions
- New Lookback: December + January + February returns
- Action: Enter new positions for April holding period
```

## Directory Structure

```
time_series_momentum/
├── modules/
│   ├── data_loader.py          # SimpleForexLoader - monthly data loading
│   ├── tsmom_strategy.py       # SimpleTSMOM - monthly signal generation
│   ├── backtest_engine.py      # SimpleTSMOMBacktest - monthly backtesting
│   └── performance_utils.py    # SimplePerformanceAnalyzer - analysis and plotting
├── notebooks/
│   └── tsmom_analysis.ipynb    # Complete monthly TSMOM analysis
├── results/                    # Generated monthly strategy results
└── README.md                   # This file
```

## Usage

### Complete Analysis in 4 Lines
```python
# 1. Load monthly forex data
loader = SimpleForexLoader()
monthly = loader.load_all_data()[2]  # Get monthly returns

# 2. Generate monthly TSMOM signals (1M, 3M, 6M, 12M lookbacks)
tsmom = SimpleTSMOM(lookbacks_monthly=[1, 3, 6, 12])
signals = tsmom.generate_all_signals(monthly)

# 3. Run monthly backtests with transaction costs
backtest = SimpleTSMOMBacktest(transaction_cost_bps=5.0)
results = backtest.run_backtest(signals, monthly)

# 4. Analyze and visualize performance
metrics = backtest.calculate_metrics(results)
analyzer = SimplePerformanceAnalyzer()
analyzer.plot_equity_curves(results)
```

### Detailed Usage
```python
# Load and inspect data
loader = SimpleForexLoader()
daily, weekly, monthly = loader.load_all_data()
print(f"Available forex pairs: {monthly.columns.tolist()}")
print(f"Monthly data shape: {monthly.shape}")

# Generate signals with custom lookbacks
tsmom = SimpleTSMOM(lookbacks_monthly=[1, 3, 6, 12])
signals = tsmom.generate_all_signals(monthly)
print(f"Generated strategies: {list(signals.keys())}")

# Run comprehensive backtest
backtest = SimpleTSMOMBacktest(transaction_cost_bps=5.0)
results = backtest.run_backtest(signals, monthly)

# Calculate detailed metrics
metrics = backtest.calculate_metrics(results)
print("Performance Summary:")
print(metrics[['Net_Sharpe_Ratio', 'Net_Annual_Return', 'Net_Max_Drawdown']])

# Create visualizations
analyzer = SimplePerformanceAnalyzer()
analyzer.plot_equity_curves(results)
analyzer.plot_performance_summary(metrics)
analyzer.plot_drawdown_analysis(results)

# Save results
backtest.save_results(results)
summary_table = analyzer.create_summary_table(metrics)
summary_table.to_csv('monthly_performance_summary.csv')
```

### Jupyter Notebook Analysis
Run the complete monthly TSMOM analysis:
```bash
cd notebooks/
jupyter notebook tsmom_analysis.ipynb
```

## Module Details

### data_loader.py - SimpleForexLoader
- **Purpose**: Load and resample forex data for monthly strategies
- **Key Method**: `load_all_data()` returns daily, weekly, monthly returns
- **Data Source**: `../../data/forex/` directory with .parquet files
- **Smart Path Resolution**: Works from any execution directory
- **Output**: Clean monthly returns DataFrame for all forex pairs

### tsmom_strategy.py - SimpleTSMOM  
- **Purpose**: Generate monthly momentum signals with different lookback periods
- **Key Method**: `generate_all_signals(monthly_returns)` creates all strategies
- **Signal Logic**: `sign(rolling_sum(returns, lookback).shift(1))`
- **Lookback Periods**: Configurable (default: 1M, 3M, 6M, 12M)
- **Bias Prevention**: Strict `.shift(1)` ensures T-1 data for T decisions

### backtest_engine.py - SimpleTSMOMBacktest
- **Purpose**: Vectorized monthly backtesting with transaction costs
- **Key Method**: `run_backtest(signals, monthly_returns)` processes all strategies
- **Portfolio Construction**: Equal weight across active monthly positions
- **Transaction Costs**: 5bp per trade applied on position changes
- **Output**: Comprehensive results dict with equity curves and metrics

### performance_utils.py - SimplePerformanceAnalyzer
- **Purpose**: Analysis and visualization of monthly strategy performance
- **Key Methods**: 
  - `plot_equity_curves()` - Monthly strategy performance plots
  - `plot_performance_summary()` - Risk-return analysis and comparisons
  - `plot_drawdown_analysis()` - Drawdown timing and recovery analysis
- **Features**: Fixed date formatting, proper Series handling, professional plots

## Data Requirements

The monthly strategies use forex data from the unified `data/forex/` directory:
- **20+ major forex pairs** in parquet format (`*_X.parquet`)
- **Daily OHLC data** from Yahoo Finance, resampled to monthly
- **Automatic Processing**: Data loading, cleaning, and monthly resampling
- **Quality Validation**: Timezone normalization and missing data handling

## Output Files

The monthly analysis generates focused output files in the `results/` directory:
- `tsmom_monthly_equity_curves.parquet` - Monthly strategy equity curves
- `tsmom_monthly_returns.parquet` - Monthly strategy return series  
- `tsmom_performance_metrics.parquet` - Comprehensive performance metrics
- `monthly_performance_summary.csv` - Formatted summary table
- `monthly_signal_analysis.csv` - Signal characteristic analysis
- `monthly_summary_report.txt` - Text summary report
- `monthly_equity_curves.png` - Monthly equity curve plots
- `monthly_performance_summary.png` - Performance comparison charts
- `monthly_drawdown_analysis.png` - Monthly drawdown analysis plots

## Performance Metrics

The framework calculates comprehensive monthly performance metrics:
- **Return Metrics**: Total return, annualized return (monthly → yearly)
- **Risk Metrics**: Volatility, Sharpe ratio (12 monthly periods annualization)
- **Drawdown Metrics**: Maximum drawdown, underwater periods analysis
- **Cost Analysis**: Transaction cost impact on monthly rebalancing performance
- **Signal Analysis**: Long/short signal distribution, active position counts

## Key Implementation Details

### Lookahead Bias Prevention
All monthly signals use `.shift(1)` to ensure decisions use T-1 data:
```python
# CORRECT: Monthly momentum with lookahead prevention
momentum = returns.rolling(window=lookback).sum().shift(1)  # T-1 data for T decisions
signals = np.sign(momentum).fillna(0)

# WRONG: Would use T data for T decisions (lookahead bias)
momentum = returns.rolling(window=lookback).sum()  # No shift - LOOKAHEAD BIAS!
```

### Monthly Equal Weighting Implementation
```python
# Equal weight across active monthly positions only
asset_returns = signals * returns  # Element-wise signal * return
active_positions = np.abs(signals).sum(axis=1)  # Count active positions per month
portfolio_returns = asset_returns.sum(axis=1) / np.maximum(active_positions, 1)
portfolio_returns = np.where(active_positions > 0, portfolio_returns, 0)  # 0 when no positions
```

### Monthly Rebalancing & Transaction Costs
```python
# Monthly position changes and transaction costs
position_changes = signals.diff().fillna(signals)  # Monthly position changes
total_turnover = np.abs(position_changes).sum(axis=1)  # Total monthly turnover
transaction_costs = total_turnover * (5 / 10000)  # 5bp per trade
net_returns = gross_returns - transaction_costs  # Net monthly returns
```

### Temporal Consistency Verification
- **Signal Generation**: End-of-month using past N months data
- **Position Holding**: Exactly 1 month (aligned with rebalancing frequency)  
- **Portfolio Rebalancing**: Monthly equal weight across active positions
- **Transaction Timing**: All changes occur at month-end simultaneously

## Assumptions and Limitations

**Assumptions:**
- **Monthly Rebalancing Feasibility**: Can trade all forex pairs at month-end
- **Transaction Costs**: 5bp per trade (conservative for major forex pairs)
- **Equal Weight Viability**: All active positions can be equally weighted
- **Data Quality**: Clean, gap-free monthly return series
- **No Funding Costs**: Simplified model without carry/funding considerations

**Limitations:**
- **Simple Equal Weighting**: No risk-based or volatility-adjusted position sizing
- **Binary Signals**: Only Long/Short/No position (no signal strength gradation)
- **No Risk Management**: No stop-losses, volatility targeting, or drawdown controls
- **Basic Cost Model**: Simplified transaction cost structure
- **Monthly-Only**: No intra-month trading or dynamic adjustments

## Future Enhancements

Potential extensions while maintaining monthly consistency:
- **Risk Parity Weighting**: Monthly volatility-adjusted position sizing
- **Signal Ranking**: Momentum score-based position sizing instead of binary
- **Risk Management**: Monthly volatility targeting and drawdown controls
- **Multi-Asset Extension**: Apply framework to commodities and equity indices
- **Alternative Momentum**: Risk-adjusted momentum or momentum combinations

## Research Applications

### Academic Research
- **Clean Baseline**: Simple, replicable momentum implementation
- **Bias-Free Framework**: Verified lookahead bias prevention
- **Monthly Consistency**: Coherent holding periods for academic studies

### Institutional Use
- **Production-Ready Code**: Professional-grade implementation
- **Scalable Design**: Easy extension to more assets or strategies
- **Comprehensive Analysis**: Full performance and risk metrics

### Strategy Development
- **Foundation Framework**: Base for more complex momentum strategies
- **Modular Design**: Easy to modify individual components
- **Educational Value**: Clear implementation for learning momentum concepts

---

**Framework Status**: ✅ Production-ready with comprehensive monthly momentum strategies

**Key Strengths**: Temporal consistency, bias prevention, vectorized implementation, comprehensive analysis

**Usage**: Ideal for momentum strategy research, academic studies, and as foundation for institutional strategy development

**Last Updated**: January 2025