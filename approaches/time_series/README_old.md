# TSMOM Strategy Implementation - Moskowitz, Ooi & Pedersen (2012)

Implementazione completa e fedele della strategia **Time-Series Momentum (TSMOM)** seguendo esattamente le specifiche del paper di Moskowitz, Ooi & Pedersen (2012) "Time series momentum".

## 🎯 Caratteristiche Principali

### Fedeltà al Paper
- ✅ **Lookback 12 mesi** con skip dell'ultimo mese (t-12 a t-1)
- ✅ **EWMA volatility** con center of mass = 60 giorni
- ✅ **Target volatility 40%** per singolo contratto
- ✅ **Equal-weight aggregation** cross-sectional
- ✅ **Look-ahead bias prevention** matematicamente garantito
- ✅ **Ribilanciamento mensile** con holding period = 1 mese

### Implementazione Tecnica
- 🚀 **Completamente vettorizzata** (no loop sugli strumenti)
- 🔍 **Validazione comprehensive** con diagnostiche automatiche
- 📊 **Visualizzazioni professionali** con analisi dettagliate
- 💾 **Export multi-formato** (CSV, Parquet, JSON)
- ⚡ **Architettura modulare** per facile estensione

## 📁 Struttura Directory

```
time_series/
├── data/                            # Cache centralizzata dati (25 commodities + T-Bill)
├── modules/                         # Moduli core della strategia
│   ├── data_manager.py              # Download dati con caching (Yahoo Finance + T-Bill)
│   ├── returns_calculator.py        # Calcolo returns (daily->monthly, excess)
│   ├── volatility_estimator.py      # EWMA volatility (COM=60, lagged)
│   ├── signal_generator.py          # Segnali TSMOM (12M, skip ultimo)
│   ├── portfolio_constructor.py     # Volatility scaling + aggregazione
│   ├── performance_analyzer.py      # Metriche performance complete
│   ├── visualizer.py               # Grafici professionali
│   └── validator.py                # Validazione e diagnostiche
├── notebooks/
│   └── tsmom_demonstration.ipynb   # Demo completa + optimization results
├── results/
│   └── optimization/               # Grid search results (80 combinations)
├── optimize_tsmom.py              # Grid search optimization module
├── tsmom_strategy.py              # Classe principale (integra tutto)
├── requirements.txt               # Dipendenze Python
└── README.md                      # Questo file
```

## 🚀 Quick Start

### 1. Installazione

```bash
# Naviga nella directory
cd approaches/time_series/

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Esecuzione Base

```python
from tsmom_strategy import TSMOMStrategy

# Inizializza strategia con parametri MOP (2012) e cache centralizzata
tsmom = TSMOMStrategy(
    start_date='2000-01-01',
    target_volatility=0.40,     # 40% target vol per contratto
    lookback_months=12,         # 12 mesi lookback
    transaction_cost_bps=0,     # 0 bps costi transazione
    data_cache_dir='data/'      # Usa cache centralizzata
)

# Esecuzione completa (include validazione)
results = tsmom.execute_full_strategy(validate_results=True)

# Visualizza performance summary
print(tsmom.get_performance_summary())

# Genera grafici
tsmom.plot_equity_curves()
tsmom.plot_drawdown_analysis()
tsmom.plot_commodity_heatmap()

# Salva tutti i risultati
tsmom.save_all_results("results/my_tsmom_run")
```

### 2.1 Grid Search Optimization

```bash
# Esegui ottimizzazione parametrica completa (80 combinazioni)
python optimize_tsmom.py

# Output: results/optimization/ con CSV, JSON e top performers
```

### 3. Demo Notebook Completa

```bash
# Avvia Jupyter
jupyter notebook notebooks/tsmom_demonstration.ipynb
```

Il notebook include:
- 🎯 Esecuzione step-by-step completa con caching
- 📊 Analisi dettagliate di ogni componente  
- 🔍 Validazione look-ahead bias
- 📈 Visualizzazioni comprehensive
- 📋 Confronto con risultati del paper
- 🔧 Grid search optimization results
- 📊 Equity curves ottimizzate (Best Sharpe vs Best CAGR)

## 📊 Specifiche Implementazione

### Universo Commodities (25 Futures - 2000-2025)

```python
DEFAULT_UNIVERSE = {
    'Energy': ["CL=F", "NG=F", "HO=F", "RB=F"],  # 4 futures
    'Metals_Precious': ["GC=F", "SI=F", "PL=F", "PA=F"],  # 4 futures
    'Metals_Industrial': ["HG=F"], # 1 future
    'Agriculture_Softs': ["KC=F", "CC=F", "SB=F", "CT=F", "OJ=F"],  # 5 futures
    'Agriculture_Grains': ["ZS=F", "ZC=F", "ZW=F", "ZM=F", "ZL=F", "ZO=F", "KE=F", "ZR=F"],  # 8 futures
    'Livestock': ["HE=F", "LE=F", "GF=F"]  # 3 futures
}
# Total: 25 commodity futures with 25+ years of historical data
```

### Formula Chiave TSMOM

1. **Segnale**: `signal[t] = sign(Σ(r[t-12] to r[t-1]))`
2. **Peso**: `w[s,t] = signal[s,t] × (0.40 / σ[s,t-1])`
3. **Portfolio Return**: `R[t+1] = mean(w[s,t] × r[s,t+1])` across securities

### Prevenzione Look-Ahead Bias

- 🔒 **Strict temporal separation**: segnali al tempo `t` usano solo dati fino a `t-1`
- 📅 **Volatility lagging**: `σ[t-1]` per position sizing al tempo `t`
- 🔄 **Consistent shifting**: tutti i `.shift(1)` applicati correttamente
- ✅ **Validation matematica**: controlli automatici su campioni casuali

## 🎨 Visualizzazioni Disponibili

### 1. Equity Curves Analysis
- Cumulative returns vs benchmark
- Monthly returns distribution
- Rolling 12M performance
- Excess returns vs risk-free

### 2. Drawdown Analysis  
- Drawdown time series con peaks
- Maximum drawdown identification
- Drawdown duration statistics

### 3. Rolling Metrics (36M Windows)
- Rolling returns, volatility, Sharpe ratio
- Rolling maximum drawdown
- Risk-return profile evolution

### 4. Commodity Heatmaps
- Average weights by year/commodity
- Position frequency analysis
- Active positions over time
- Annual portfolio returns

### 5. Signal Analysis
- Signal distribution over time
- Long/short frequency by commodity
- Signal correlation matrix
- Momentum distribution

## 📈 Performance Metrics

### Return Metrics
- CAGR, Total Return, Hit Ratio
- Best/Worst month, Volatility

### Risk Metrics  
- Sharpe Ratio, Sortino Ratio
- VaR/CVaR (95%, 99%)
- Downside deviation

### Drawdown Metrics
- Maximum Drawdown, Calmar Ratio
- Average/Max DD duration
- Current drawdown status

### Distribution Metrics
- Skewness, Kurtosis
- Normality tests (Jarque-Bera)

## 🔍 Validazione Sistema

### Look-Ahead Bias Prevention
- ✅ Sample validation su punti temporali casuali
- ✅ Timing pattern consistency check
- ✅ Mathematical verification dei segnali

### Data Integrity  
- ✅ Temporal alignment validation
- ✅ Signal generation logic verification
- ✅ Portfolio construction accuracy
- ✅ Volatility calculation correctness

### Quality Assurance
- ✅ Automated error detection
- ✅ Warning system per anomalie
- ✅ Comprehensive diagnostic reports

## 🔧 Personalizzazione

### Custom Universe
```python
custom_universe = {
    'Energy': ["CL=F", "NG=F"], 
    'Metals': ["GC=F", "SI=F"]
}

tsmom = TSMOMStrategy(universe=custom_universe)
```

### Parametri Alternativi
```python
tsmom = TSMOMStrategy(
    target_volatility=0.30,        # 30% instead of 40%
    lookback_months=9,             # 9M instead of 12M  
    transaction_cost_bps=5         # 5 bps transaction costs
)
```

### Rolling Sensitivity Analysis
```python
# Test multiple configurations
configs = [
    {'target_volatility': 0.30},
    {'target_volatility': 0.50}, 
    {'lookback_months': 9},
    {'transaction_cost_bps': 5}
]

for config in configs:
    strategy = TSMOMStrategy(**config)
    results = strategy.execute_full_strategy()
    # Compare results...
```

## 📊 Benchmark vs Paper MOP (2012)

| Metric | Paper MOP | Typical Range |
|--------|-----------|---------------|
| Annual Return | ~12.4% | 10-15% |
| Annual Volatility | ~8.9% | 8-12% |
| Sharpe Ratio | ~1.39 | 1.2-1.6 |
| Max Drawdown | ~-4.6% | -5% to -8% |

**Note**: Differenze possono derivare da:
- Diversa fonte dati (Yahoo vs Datastream)
- Metodologia roll futures 
- Periodo sample esteso
- Dettagli implementativi minori

## 🚨 Limitazioni e Note

### Data Source
- **Yahoo Finance**: Continuous futures potrebbero differire dai contratti originali
- **T-Bill Rate**: Fallback al 2% fisso se download fallisce
- **Missing Data**: Gestione automatica con forward fill

### Performance Considerations
- **First Run**: Compilation overhead se usa Numba
- **Memory Usage**: ~1-2GB per dataset completo
- **Execution Time**: 2-5 minuti per run completo

### Implementation Notes
- **Timezone Naive**: Tutti i timestamp sono naive per consistency
- **Business Month-End**: Resampling all'ultimo giorno lavorativo
- **Equal Weight Aggregation**: Media semplice cross-sectional come in MOP

## 📚 References

**Primary Paper:**
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. Journal of financial economics, 104(2), 228-250.

**Related Literature:**
- AQR: "Time Series Momentum" (2011)
- Academic implementations and extensions
- Industry practitioner notes

## 🤝 Contributing

Suggerimenti per miglioramenti:

1. **Additional Asset Classes**: Estensione a FX, bonds, equity indices
2. **Alternative Specifications**: Signal smoothing, regime detection
3. **Risk Management**: Portfolio-level risk controls
4. **Performance Enhancement**: Numba optimization, parallel processing
5. **Extended Validation**: Monte Carlo simulation, bootstrap tests

## 📝 License & Disclaimer

Questa implementazione è per scopi educativi e di ricerca. 
- ⚠️ **Non costituisce consulenza finanziaria**
- ⚠️ **Performance passate non garantiscono risultati futuri**  
- ⚠️ **Testare sempre su dati out-of-sample prima di uso reale**

---

*Implementazione completa e validata della strategia TSMOM di Moskowitz, Ooi & Pedersen (2012). Ogni dettaglio è stato implementato seguendo esattamente le specifiche del paper con validazione matematica per garantire correttezza e assenza di look-ahead bias.*