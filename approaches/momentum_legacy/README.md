# 🚀 Advanced Momentum Strategies Framework

## 📊 Complete Framework for Commodity Momentum & Contrarian Strategies

This comprehensive framework implements and evaluates multiple advanced momentum strategies on commodity futures, revealing key insights about what works (and what doesn't) in commodity trading.

## 🎯 Key Findings

### 🏆 **Winners:**
- **Best Strategy**: Basic Contrarian 1D (+68.84%, Sharpe 0.103)
- **Runner-up**: Basic Momentum 5D (+59.58%, Sharpe 0.102)

### 💡 **Key Insights:**
1. **Simple beats complex** - Basic strategies outperform advanced ones
2. **Contrarian dominates** - Mean reversion works better than momentum on commodities  
3. **1-day lookbacks** are surprisingly effective
4. **Advanced techniques** often add noise, not signal

## 📁 Project Structure

```
MOMENTUM/
├── src/                           # Core framework modules
│   ├── data_loader.py            # Centralized data loading
│   ├── signals.py                # All signal generation functions
│   ├── backtest_engine.py        # Unified backtesting framework
│   └── portfolio.py              # Advanced portfolio construction
├── strategies/                    # Strategy implementations
│   ├── multi_timeframe.py        # 5D, 10D, 20D momentum combinations
│   ├── volatility_adjusted.py    # Z-score based signals
│   ├── percentile_based.py       # Cross-sectional ranking
│   ├── cross_sectoral.py         # Sector-relative momentum
│   └── regime_adaptive.py        # Market regime switching
├── notebooks/                    # Analysis notebooks
│   ├── momentum_analysis.ipynb   # Original momentum vs contrarian
│   ├── contrarian_analysis.ipynb # Detailed contrarian analysis
│   └── comprehensive_strategy_analysis.ipynb # Complete framework analysis
├── results/                      # All backtest results and comparisons
├── raw/                         # Original commodity data (15 futures)
└── main.py                      # Run all strategies
```

## 🚀 Quick Start

### 1. Run Complete Analysis
```bash
python main.py
```

### 2. Individual Strategy Categories
```bash
# Multi-timeframe strategies
python strategies/multi_timeframe.py

# Volatility-adjusted strategies  
python strategies/volatility_adjusted.py

# Percentile-based strategies
python strategies/percentile_based.py
```

### 3. Explore Results
```bash
# Open the comprehensive analysis notebook
jupyter notebook notebooks/comprehensive_strategy_analysis.ipynb
```

## 📊 Strategies Implemented

### 🔵 **Basic Strategies**
- Basic Momentum (1D, 5D)
- Basic Contrarian (1D, 5D)

### 🔴 **Advanced Strategies**
- **Multi-Timeframe**: Combines 1D, 5D, 10D, 20D signals
- **Volatility-Adjusted**: Z-score based mean reversion
- **Percentile-Based**: Cross-sectional ranking (top/bottom %)
- **Cross-Sectoral**: Momentum relative to commodity sectors
- **Regime-Adaptive**: Switches between momentum/contrarian

### 🌟 **Portfolio Techniques**
- Risk Parity weighting
- Volatility targeting
- Strategy blending
- Dollar-neutral construction

## 📈 Performance Summary

| Strategy | Total Return | Sharpe Ratio | Max Drawdown |
|----------|--------------|--------------|--------------|
| **Basic Contrarian 1D** | **+68.84%** | **0.103** | **-53.02%** |
| Basic Momentum 5D | +59.58% | 0.102 | -40.57% |
| Multi-TF [1,5,10] | -41.00% | -0.297 | -65.53% |
| Vol-Adj Contrarian | -79.86% | -0.912 | -87.74% |
| Percentile Contrarian | -13.79% | -0.132 | -78.86% |

## 🛠️ Technical Features

### 📊 **Data Processing**
- Automated commodity data loading from parquet files
- Sector classification (Energy, Metals, Agriculture)
- Comprehensive data quality checks

### 🧮 **Signal Generation**
- Lookahead bias prevention (proper shifting)
- Multiple timeframe support
- Volatility normalization
- Cross-sectional ranking

### 🎯 **Backtesting Engine**
- Unified framework for all strategies
- Multiple portfolio construction methods
- Comprehensive performance metrics
- Risk-adjusted analysis

### 📊 **Portfolio Construction**
- Equal weight, Risk parity, Volatility targeting
- Long/short, Dollar-neutral options
- Strategy blending and optimization
- Risk management overlays

## 📚 Key Modules

### `src/data_loader.py`
- Centralized commodity data loading
- Sector mapping and classification
- Data quality validation

### `src/signals.py`
- All signal generation functions
- Consistent interface across strategies
- Proper lookahead bias handling

### `src/backtest_engine.py`
- Unified backtesting framework
- Comprehensive performance metrics
- Multiple portfolio construction options

### `src/portfolio.py`
- Advanced portfolio optimization
- Risk parity, volatility targeting
- Strategy blending techniques

## 🎯 Usage Examples

### Load Data and Generate Signals
```python
from src.data_loader import CommodityDataLoader
from src.signals import basic_contrarian_signals

# Load data
loader = CommodityDataLoader('raw')
returns = loader.calculate_returns()

# Generate signals
signals = basic_contrarian_signals(returns, lookback=1)
```

### Run Backtest
```python
from src.backtest_engine import BacktestEngine

# Initialize engine
engine = BacktestEngine(returns)

# Run backtest
result = engine.run_backtest(
    signals=signals,
    strategy_name="My Strategy",
    portfolio_type="long_short"
)

print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
```

## 📊 Data Requirements

### Commodity Universe (15 Futures)
- **Energy**: CL=F, NG=F, BZ=F, RB=F, HO=F
- **Precious Metals**: GC=F, SI=F  
- **Industrial Metals**: HG=F, PA=F
- **Agriculture**: ZC=F, ZW=F, ZS=F, SB=F, CT=F, CC=F

### Data Format
- Parquet files with OHLCV data
- Daily frequency, 2010-2025
- Yahoo Finance format with MultiIndex columns

## 🏆 Performance Insights

### What Works
1. **Simple contrarian strategies** (1-day mean reversion)
2. **Basic momentum** on 5-day timeframe
3. **Equal weight allocation** among signals
4. **Daily rebalancing** for best results

### What Doesn't Work
1. **Complex multi-timeframe combinations**
2. **Volatility-adjusted signals** (too noisy)
3. **Percentile-based ranking** (unstable)
4. **Advanced portfolio optimization** (overfitting)

### Key Lessons
- **Simplicity wins** in commodity trading
- **Mean reversion** dominates momentum
- **Avoid over-engineering** signals
- **Transaction costs matter** (not included in this analysis)

## ⚙️ Configuration

### Strategy Parameters
- Lookback periods: 1, 5, 10, 20, 60 days
- Z-score thresholds: 1.0, 1.5, 2.0, 2.5, 3.0
- Percentile cutoffs: 10%, 20%, 30%
- Volatility windows: 10, 20, 30, 60 days

### Portfolio Settings
- Rebalancing: Daily, Weekly, Monthly
- Position limits: 5%, 10%, 15% max per commodity
- Portfolio types: Long-only, Long-short, Dollar-neutral

## 📈 Future Enhancements

- [ ] Transaction cost integration
- [ ] Options-based signals  
- [ ] Macro-economic overlays
- [ ] Machine learning features
- [ ] Real-time data feeds
- [ ] Live trading integration

## 🚨 Important Notes

1. **No Transaction Costs**: Results are gross of all trading costs
2. **Survivorship Bias**: Only includes currently active contracts
3. **Backtesting Limitations**: Past performance ≠ future results
4. **Risk Management**: Always use proper position sizing and risk controls

## 📧 Contact & Support

This framework was developed as a comprehensive study of momentum strategies in commodity markets. The code is designed to be:
- **Modular** and easily extensible
- **Well-documented** with clear examples
- **Performance-optimized** for large datasets
- **Research-focused** with detailed analysis

---

## ✅ **Final Recommendation**: 
Use **Basic Contrarian 1D** as your core strategy, with **Basic Momentum 5D** as a diversifier. Keep it simple - complexity is the enemy of profitability in commodity momentum strategies!