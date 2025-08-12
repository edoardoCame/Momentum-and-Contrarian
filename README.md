# 🚀 Momentum & Contrarian Trading Strategies - Unified Framework

> Advanced quantitative finance research framework implementing momentum and contrarian trading strategies across Commodity Futures and Forex markets

## 🎯 Project Overview

This comprehensive framework unifies three distinct approaches to quantitative trading research:

### 📊 **Three Approaches, One Repository**

1. **🔵 Momentum Legacy** (`approaches/momentum_legacy/`) 
   - 15 Commodity Futures (Energy, Metals, Agriculture)
   - Advanced momentum & contrarian strategies
   - Multi-timeframe analysis
   - **Best Strategy**: Basic Contrarian 1D (+68.84%, Sharpe 0.103)

2. **🟡 Vectorized Contrarian** (`approaches/vectorized_contrarian/`)
   - Educational framework for learning contrarian strategies
   - 20+ Forex currency pairs + Commodities
   - Risk parity portfolio construction
   - Clear, comprehensible implementation

3. **🔴 Advanced Contrarian Engine** (`approaches/advanced_contrarian/`)
   - Production-grade institutional backtesting framework
   - High-performance Numba optimization
   - Comprehensive risk management
   - Professional deployment ready

4. **🟢 Seasonality Analysis** (`seasonality_analysis/`)
   - Comprehensive seasonal pattern analysis for commodities
   - 15+ years of historical data (2010-2025)
   - Statistical significance testing
   - Sector-based seasonal insights (Energy, Metals, Agriculture)

## 🏗️ Unified Architecture

```
momentum_contrarian_unified/
├── README.md                           # This file - main documentation
├── CLAUDE.md                          # AI development instructions
├── requirements.txt                   # Unified dependencies
│
├── data/                              # Consolidated data storage
│   ├── commodities/                   # 15 commodity futures (.parquet)
│   └── forex/                         # 20+ forex pairs + results
│
├── src/                               # Unified core modules (future development)
│   ├── data_loaders/                  # Asset-specific data loaders
│   ├── strategies/                    # Unified strategy implementations
│   ├── backtest/                      # Unified backtesting engine
│   ├── portfolio/                     # Portfolio construction & optimization
│   └── utils/                         # Common utilities
│
├── seasonality_analysis/              # Commodity seasonality research
│   ├── modules/                       # Seasonality analysis engine
│   ├── notebooks/                     # Interactive seasonal analysis
│   ├── results/                       # Seasonal pattern visualizations
│   └── data/                          # Processed seasonal data
│
├── approaches/                        # Original implementations preserved
│   ├── momentum_legacy/              # Complete MOMENTUM project
│   ├── vectorized_contrarian/        # Educational contrarian approach
│   └── advanced_contrarian/          # Production-grade contrarian engine
│
├── notebooks/                         # Unified analysis notebooks
├── results/                          # Consolidated results
└── scripts/                          # Utility scripts
```

## 📈 Asset Coverage

### **Commodity Futures (15 instruments)**
- **Energy**: CL=F, NG=F, BZ=F, RB=F, HO=F
- **Precious Metals**: GC=F, SI=F, PA=F
- **Industrial Metals**: HG=F
- **Agriculture**: ZC=F, ZW=F, ZS=F, SB=F, CT=F, CC=F

### **Forex Currency Pairs (20+ instruments)**
- **Major Pairs**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- **Cross Pairs**: EURGBP, EURJPY, EURCHF, EURAUD, EURCAD, EURNZD
- **Exotic Crosses**: GBPJPY, GBPCHF, GBPAUD, GBPCAD, GBPNZD, AUDJPY, AUDCHF

## 🏆 Key Research Findings

### **Momentum Strategies (Commodities)**
- ✅ **Simple beats complex** - Basic strategies outperform advanced ones
- ✅ **Contrarian dominates** - Mean reversion works better than momentum
- ✅ **1-day lookbacks** are surprisingly effective
- ❌ **Advanced techniques** often add noise, not signal

### **Contrarian Strategies (Forex)**
- ✅ **Risk parity** portfolio construction enhances returns
- ✅ **Mean reversion** hypothesis confirmed across currency pairs
- ✅ **Weekly rebalancing** optimal for transaction cost balance
- ✅ **Inverse volatility weighting** improves risk-adjusted returns

### **Seasonality Analysis (Commodities)**
- ✅ **Strong seasonal patterns** across all commodity sectors
- ✅ **Statistical significance** in 15+ years of data
- ✅ **Sector rotation insights** - Energy, Metals, Agriculture timing
- ✅ **Calendar anomalies** - Month and day-of-week effects identified

## 🚀 Quick Start Guide

### **Option 1: Momentum Analysis (Commodities)**
```bash
cd approaches/momentum_legacy
python main.py                    # Run all momentum strategies
jupyter notebook notebooks/comprehensive_strategy_analysis.ipynb
```

### **Option 2: Educational Contrarian (Learning)**
```bash
cd approaches/vectorized_contrarian/forex/notebooks
jupyter notebook fx_main_educational.ipynb    # Start here for learning
```

### **Option 3: Advanced Contrarian (Production)**
```bash
cd approaches/advanced_contrarian/notebooks
jupyter notebook portfolio_analysis.ipynb     # Professional analysis
```

### **Option 4: Seasonality Analysis (Commodity Patterns)**
```bash
cd seasonality_analysis/notebooks
jupyter notebook commodities_seasonality.ipynb     # Seasonal pattern analysis
```

### **Option 5: Compare All Approaches**
```bash
python scripts/compare_approaches.py          # Coming soon
```

## 📊 Performance Highlights

| Strategy | Asset Class | Total Return | Sharpe Ratio | Max Drawdown |
|----------|-------------|--------------|--------------|--------------|
| **Basic Contrarian 1D** | Commodities | **+68.84%** | **0.103** | -53.02% |
| Basic Momentum 5D | Commodities | +59.58% | 0.102 | -40.57% |
| Risk Parity Contrarian | Forex | Variable | Variable | Variable |
| Advanced Contrarian | Forex | Optimized | Optimized | Controlled |
| **Seasonality Analysis** | **Commodities** | **Patterns** | **Statistical** | **Identified** |

## 🛠️ Installation & Dependencies

### **Core Dependencies**
```bash
pip install -r requirements.txt

# Manual installation if needed:
pip install yfinance pandas numpy matplotlib numba scipy scikit-learn cvxpy jupyter plotly seaborn
```

### **Optional: High Performance**
```bash
pip install numba            # For JIT compilation (advanced engine)
pip install cvxpy           # For optimization (portfolio construction)
```

## 📚 Documentation Structure

- **`README.md`** (this file) - Main overview and quick start
- **`CLAUDE.md`** - AI development instructions and architecture details
- **`seasonality_analysis/notebooks/commodities_seasonality.ipynb`** - Seasonal analysis documentation
- **`approaches/momentum_legacy/README.md`** - Detailed momentum strategies documentation
- **`approaches/advanced_contrarian/README.md`** - Professional contrarian framework docs
- **`approaches/vectorized_contrarian/CLAUDE.md`** - Educational approach documentation

## 🔬 Research Methodology

### **Bias Prevention**
- ✅ **Strict temporal separation** - Signals at T-1, execution at T
- ✅ **Lookahead validation** - Mathematical guarantees against future info leakage
- ✅ **Proper lag implementation** - Consistent `.shift()` operations

### **Performance Optimization**
- ✅ **Vectorized operations** - NumPy/Pandas bulk processing
- ✅ **JIT compilation** - Numba optimization for critical loops
- ✅ **Memory management** - Efficient data structures and lazy loading
- ✅ **Parallel processing** - Multi-core parameter optimization

### **Risk Management**
- ✅ **Value at Risk (VaR)** - Historical and parametric calculations
- ✅ **Drawdown analysis** - Maximum drawdown and recovery periods
- ✅ **Correlation monitoring** - Dynamic correlation tracking
- ✅ **Position limits** - Concentration and exposure controls

## 🎓 Educational Value

### **For Students**
- Start with `approaches/vectorized_contrarian/forex/notebooks/fx_main_educational.ipynb`
- Clear step-by-step implementation of contrarian strategies
- Interactive analysis and visualization
- Explore `seasonality_analysis/notebooks/commodities_seasonality.ipynb` for seasonal patterns

### **For Researchers**
- Comprehensive backtesting frameworks in all approaches
- Multiple asset classes and strategy types
- Extensive performance metrics and analysis

### **For Practitioners**
- Production-grade advanced engine with institutional features
- Transaction cost modeling and risk management
- Scalable architecture for real-time deployment

## 🏛️ Academic Foundation

### **Core Literature**
1. **Mean Reversion**: Fama & French (1988) - Permanent and temporary components
2. **Risk Parity**: Roncalli (2013) - Risk budgeting and portfolio construction  
3. **Momentum**: Jegadeesh & Titman (1993) - Returns to buying winners and selling losers
4. **Backtesting**: Bailey et al. (2014) - Pseudo-mathematics and overfitting

## ⚠️ Important Notes

### **Research Limitations**
- ❗ **No transaction costs** in basic implementations (gross returns)
- ❗ **Survivorship bias** - Only currently active contracts included
- ❗ **Backtesting limitations** - Past performance ≠ future results
- ❗ **Risk management** - Always implement proper position sizing

### **Data Quality**
- ✅ **High-quality data** from Yahoo Finance API
- ✅ **Daily frequency** with comprehensive coverage
- ✅ **Automated validation** and quality checks
- ✅ **Parquet format** for efficient storage and access

## 🔮 Future Development Roadmap

### **Phase 1: Unified Core (Q2 2025)**
- [ ] Unified data loader supporting both asset classes
- [ ] Common backtesting engine with switchable implementations
- [ ] Integrated performance analytics and reporting

### **Phase 2: Advanced Features (Q3 2025)**
- [ ] Real-time data feeds and live trading integration
- [ ] Machine learning enhanced signal generation
- [ ] Multi-asset portfolio optimization
- [ ] Advanced risk management overlays

### **Phase 3: Production Deployment (Q4 2025)**
- [ ] Cloud-native architecture
- [ ] API-based strategy deployment
- [ ] Institutional reporting and compliance
- [ ] Performance monitoring and alerting

## 📄 License

MIT License - See individual approach directories for specific licensing details.

## 🤝 Contributing

This is a research framework combining three distinct approaches. Each approach maintains its original structure and can be developed independently:

1. **Momentum Legacy** - Extend commodity analysis and add new strategies
2. **Vectorized Contrarian** - Improve educational content and add new asset classes  
3. **Advanced Contrarian** - Enhance production features and optimization

## 🏁 Getting Started Recommendations

1. **New to Quantitative Finance?** → Start with `approaches/vectorized_contrarian/`
2. **Interested in Momentum Strategies?** → Explore `approaches/momentum_legacy/`
3. **Want to Understand Seasonality?** → Analyze with `seasonality_analysis/`
4. **Building Production Systems?** → Study `approaches/advanced_contrarian/`
5. **Want to Compare Everything?** → Use the unified `data/` and future `src/` modules

---

**⚡ This unified framework provides everything needed for comprehensive quantitative trading research across multiple asset classes and strategy types. Choose your approach based on your experience level and intended use case!**