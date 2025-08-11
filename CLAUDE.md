# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this unified repository.

## Project Overview

This is a comprehensive quantitative finance research framework that unifies three distinct approaches to momentum and contrarian trading strategies across Commodity Futures and Forex markets. The repository contains:

1. **Momentum Legacy** (`approaches/momentum_legacy/`) - Commodity momentum & contrarian strategies
2. **Vectorized Contrarian** (`approaches/vectorized_contrarian/`) - Educational contrarian framework  
3. **Advanced Contrarian Engine** (`approaches/advanced_contrarian/`) - Production-grade backtesting

## Repository Structure

### Main Directory Structure
```
momentum_contrarian_unified/
├── data/                           # Unified data storage
│   ├── commodities/               # 15 commodity futures (.parquet)
│   └── forex/                     # 20+ forex pairs + analysis results
├── src/                           # Future unified core modules
├── approaches/                    # Three preserved approaches
├── notebooks/                     # Future unified analysis
├── results/                       # Consolidated results
└── scripts/                       # Utility scripts
```

### Approaches Directory Details
Each approach is self-contained and maintains its original structure:

**`approaches/momentum_legacy/`** (from MOMENTUM project)
- `src/` - Core modules (data_loader.py, signals.py, backtest_engine.py, portfolio.py)
- `strategies/` - Strategy implementations
- `notebooks/` - Jupyter analysis notebooks
- `raw/` - Original commodity data (preserved but duplicated in `data/commodities/`)
- `results/` - Backtest results and performance metrics

**`approaches/vectorized_contrarian/`** (from contrarian_fx/vectorized_approach)
- `modules/` - Core contrarian modules
- `forex/` - Forex-specific analysis
- `commodities/` - Commodity-specific analysis  
- `shared/` - Shared utilities and multi-asset analysis

**`approaches/advanced_contrarian/`** (from contrarian_fx/advanced_engine)
- `modules/` - Professional-grade modules with Numba optimization
- `notebooks/` - Institutional analysis notebooks
- `data/` - High-performance data storage (duplicated in `data/forex/`)
- `results/` - Comprehensive backtesting results

## Development Commands

### Working with Individual Approaches

**Momentum Legacy (Commodities)**
```bash
cd approaches/momentum_legacy
python main.py                                    # Run all momentum strategies
python strategies/volatility_adjusted.py          # Run specific strategy category
jupyter notebook notebooks/comprehensive_strategy_analysis.ipynb
```

**Vectorized Contrarian (Educational)**
```bash
cd approaches/vectorized_contrarian
python -m modules.forex_backtest                  # Run forex backtest
python -m modules.commodities_backtest            # Run commodities backtest
jupyter notebook forex/notebooks/fx_main_educational.ipynb
```

**Advanced Contrarian (Production)**
```bash
cd approaches/advanced_contrarian
cd modules && python backtesting_engine.py        # High-performance backtesting
cd modules && python parameter_optimizer.py       # Parameter optimization
jupyter notebook notebooks/portfolio_analysis.ipynb
```

### Unified Data Access
```bash
# Commodities data (available in both approaches/momentum_legacy/raw/ and data/commodities/)
ls data/commodities/                              # 15 commodity futures (.parquet)

# Forex data (available in approaches/*/data/ and data/forex/)  
ls data/forex/                                    # 20+ currency pairs (.parquet)
```

## Core Technologies and Patterns

### Programming Languages & Libraries
- **Python 3.8+** - Primary language
- **Pandas/NumPy** - Data manipulation and numerical computing
- **Numba** - JIT compilation for performance-critical code (advanced engine)
- **Jupyter** - Interactive analysis and visualization
- **yfinance** - Financial data acquisition
- **cvxpy** - Portfolio optimization (advanced features)

### Data Patterns
- **Parquet format** - Efficient data storage for time series
- **Daily frequency** - All strategies use daily OHLC data
- **Yahoo Finance format** - Standardized data structure across approaches
- **Lookahead bias prevention** - Consistent `.shift(1)` patterns for signals

### Strategy Implementation Patterns

**Signal Generation (All Approaches)**
```python
# Always prevent lookahead bias
def generate_signals(returns_data, lookback=20):
    # Use t-1 data for t decisions
    lagged_returns = returns_data.shift(1)
    rolling_performance = lagged_returns.rolling(window=lookback).sum()
    
    # Generate signals
    signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
    # ... signal logic ...
    return signals.fillna(0)
```

**Portfolio Construction Patterns**
```python
# Risk parity weighting (contrarian approaches)
def calculate_risk_parity_weights(returns_covariance):
    # Equal Risk Contribution implementation
    # Returns normalized weights
    
# Equal weighting (momentum legacy)
def equal_weight_portfolio(signals):
    # Simple equal weighting across active signals
```

## Approach-Specific Development Guidance

### Momentum Legacy Development
- **Module Structure**: Well-organized src/ directory with clear separation
- **Strategy Categories**: Multi-timeframe, volatility-adjusted, percentile-based, etc.
- **Performance Focus**: Comprehensive strategy comparison and analysis
- **Key Files**: 
  - `src/signals.py` - All signal generation functions
  - `src/backtest_engine.py` - Unified backtesting framework
  - `main.py` - Orchestrates all strategies

### Vectorized Contrarian Development  
- **Educational Focus**: Clear, readable implementations over optimization
- **Modular Design**: Separate modules for each asset class
- **Documentation**: Extensive inline documentation and educational notebooks
- **Key Files**:
  - `modules/strategy_contrarian.py` - Core contrarian logic
  - `modules/forex_backtest.py` - Forex backtesting
  - Notebooks in `forex/notebooks/` and `commodities/notebooks/`

### Advanced Contrarian Development
- **Production Quality**: Professional-grade code with comprehensive error handling
- **Performance Optimization**: Numba JIT compilation for critical paths
- **Institutional Features**: Transaction costs, risk management, compliance reporting
- **Key Files**:
  - `modules/backtesting_engine.py` - High-performance backtesting
  - `modules/portfolio_manager.py` - Advanced portfolio construction
  - `modules/parameter_optimizer.py` - Multi-objective optimization

## Data Management

### Data Location Strategy
- **Unified Data**: `data/commodities/` and `data/forex/` for cross-approach access
- **Approach-Specific**: Original data preserved in each approach directory
- **Results**: Each approach maintains its own results structure

### Data Quality and Validation
- **Automatic Validation**: Built into data loaders across all approaches
- **Missing Data Handling**: Graceful degradation when instruments fail
- **Quality Checks**: Validate price data, check for gaps, ensure consistency

## Key Development Principles

### Bias Prevention (Critical)
- **Temporal Separation**: All signals use T-1 data for T execution
- **Lookahead Validation**: Mathematical guarantees against future information
- **Consistent Patterns**: Always use `.shift(1)` for signal generation

### Performance Optimization
- **Vectorized Operations**: Prefer pandas/numpy vectorization
- **Memory Efficiency**: Use appropriate data types and lazy loading
- **JIT Compilation**: Use Numba for performance-critical loops (advanced engine)
- **Parallel Processing**: Multi-core optimization where appropriate

### Code Organization
- **Modular Design**: Clear separation of data, signals, backtesting, and analysis
- **Single Responsibility**: Each module/function has a clear, focused purpose
- **Documentation**: Comprehensive docstrings and inline comments
- **Error Handling**: Robust error handling and logging

## Testing and Validation

### Approach Testing
Each approach should be tested independently in its directory:

```bash
# Test momentum legacy
cd approaches/momentum_legacy && python main.py

# Test vectorized contrarian  
cd approaches/vectorized_contrarian && python test_structure.py

# Test advanced contrarian
cd approaches/advanced_contrarian/modules && python backtesting_engine.py
```

### Data Integrity Validation
```bash
# Verify data migration integrity
python scripts/validate_data_migration.py        # Future utility script

# Check for data consistency across approaches
python scripts/data_consistency_check.py         # Future utility script
```

## Future Development Guidelines

### Unified Core Development (src/)
When developing the unified core modules:
- **Backward Compatibility**: Ensure all three approaches continue to work
- **Performance**: Match or exceed the performance of individual approaches
- **Flexibility**: Support both educational and production use cases
- **Extensibility**: Easy to add new asset classes and strategies

### New Strategy Development
- **Multi-Asset Support**: Consider both commodities and forex from the start
- **Bias Prevention**: Implement strict temporal separation
- **Documentation**: Include both technical and educational documentation
- **Performance Metrics**: Comprehensive analysis and comparison

## Common Pitfalls to Avoid

1. **Lookahead Bias**: Always use `.shift(1)` for signals
2. **Data Leakage**: Never use future information in backtests
3. **Overfitting**: Be cautious with parameter optimization
4. **Transaction Costs**: Account for realistic trading costs
5. **Path Conflicts**: Be careful with relative imports across approaches

## File Dependencies and Relationships

### Inter-Approach Dependencies
- **Data**: Shared data in `data/` directory
- **Results**: Independent results in each approach
- **Utils**: Some shared utilities in vectorized approach

### Within-Approach Dependencies
Each approach maintains its own dependency graph:
- **Momentum Legacy**: src/ modules → strategies/ → main.py
- **Vectorized Contrarian**: modules/ → notebooks/ → analysis
- **Advanced Contrarian**: modules/ → notebooks/ → production analysis

## Performance Benchmarks

### Expected Performance Characteristics
- **Momentum Legacy**: ~30 seconds for complete analysis
- **Vectorized Contrarian**: ~60 seconds for full forex backtest  
- **Advanced Contrarian**: ~10 seconds with Numba optimization

### Memory Usage
- **Small Dataset**: < 500MB (daily data, 5+ years)
- **Large Dataset**: 1-2GB (comprehensive analysis with all approaches)

## Deployment Considerations

### Development Environment
- **Python 3.8+** required
- **Jupyter Lab/Notebook** for interactive analysis
- **Sufficient RAM**: 4GB minimum, 8GB recommended

### Production Deployment (Advanced Engine)
- **Numba Compilation**: First run includes compilation overhead
- **Memory Management**: Monitor memory usage with large datasets
- **Parallel Processing**: Configure based on available CPU cores

---

This unified repository provides comprehensive quantitative trading research capabilities while preserving the distinct advantages of each approach. Choose the appropriate approach based on your specific needs and development context.