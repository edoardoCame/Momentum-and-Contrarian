# Time Series Momentum (TSMOM) Strategy

Implementazione in Python della strategia Time Series Momentum basata su Moskowitz-Ooi-Pedersen (2012), ottimizzata per prevenire il lookahead bias e massimizzare le performance attraverso operazioni vettorizzate.

## Caratteristiche Implementate

### Specifiche Tecniche
- **Universo**: 21 commodities futures + 9 coppie FX da Yahoo Finance
- **Periodo**: Dal 2000-01-01 (configurabile)
- **Lookback**: k=12 mesi (parametrizzabile)
- **Holding Period**: h=1 mese (parametrizzabile)
- **Target Volatility**: 40% annualizzato
- **EWMA Volatility**: Center-of-mass ≈ 60 giorni

### Logica Temporale Corretta (Zero Lookahead)
1. **Segnali momentum**: Al primo giorno del mese t, usa ritorni fino alla fine del mese t-1 (appena concluso)
2. **Volatilità EWMA**: Lag giornaliero per evitare intraday lookahead bias
3. **Applicazione**: Segnale del mese t basato su informazioni disponibili all'inizio del mese t
4. **Validazione**: Controlli automatici della separazione temporale

### Ottimizzazioni Performance
- **Data Caching**: Dati salvati in `.parquet` per riuso istantaneo (24h cache)
- **Vettorizzazione**: `np.prod` invece di `.apply(lambda)` per calcoli momentum
- **Logging Ottimizzato**: Output conciso per esecuzione veloce
- **Memory Efficiency**: Allineamento dati e gestione memoria ottimizzata

## Struttura del Progetto

```
approaches/TSMOM/
├── modules/
│   ├── data_loader.py          # Download e preprocessing con caching
│   ├── tsmom_strategy.py       # Generazione segnali e costruzione portfolio
│   ├── volatility_estimator.py # Stima volatilità EWMA ottimizzata
│   └── performance_analyzer.py # Analisi performance e visualizzazioni
├── data/cached/                # Cache dati (.parquet files)
│   ├── daily_prices.parquet   # Prezzi giornalieri cached
│   ├── monthly_prices.parquet # Prezzi mensili cached
│   └── daily_returns.parquet  # Returns giornalieri cached
├── main.py                     # Script esecuzione principale
├── results/                    # Output generati
│   ├── tsmom_yahoo_2000.csv   # Serie temporali portfolio
│   ├── equity_tsmom.png       # Curve equity
│   ├── dd_tsmom.png           # Analisi drawdown
│   ├── performance_summary.csv # Metriche performance
│   └── parameter_grid_results.csv # Test robustezza
└── README.md                   # Documentazione
```

## Utilizzo

### Esecuzione Rapida
```bash
cd approaches/TSMOM
python3 main.py
# Scegli opzione 2 per test veloce
```

### Esecuzione Completa
```bash
cd approaches/TSMOM  
python3 main.py
# Scegli opzione 1 per implementazione completa
```

### Sistema di Caching Efficiente 
```bash
# Prima esecuzione: scarica e salva dati in cache
python3 main.py  # Opzione 1

# Esecuzioni successive: carica da cache (velocissimo!)
python3 main.py  # Opzione 1 - usa dati cached

# Forza refresh completo dati
python3 main.py  # Opzione 3 - clear cache & refresh
```

### Configurazione Parametri
Modifica `main.py` per personalizzare:
```python
START_DATE = "2000-01-01"  # Data inizio
K_LOOKBACK = 12            # Mesi lookback 
H_HOLDING = 1              # Mesi holding
TARGET_VOL = 0.40          # Volatilità target (40%)
COM_DAYS = 60              # EWMA center-of-mass
```

## Universo Trading

### Commodities Futures (21)
- **Energia**: CL=F, BZ=F, HO=F, RB=F, NG=F
- **Metalli**: GC=F, SI=F, HG=F, PL=F, PA=F  
- **Agricoli**: ZC=F, ZW=F, ZS=F, ZM=F, ZL=F
- **Soft**: KC=F, CC=F, SB=F, CT=F
- **Livestock**: LE=F, HE=F

### Forex Spot (9)
- **Major**: EURUSD=X, GBPUSD=X, AUDUSD=X, NZDUSD=X
- **USD Base**: USDJPY=X, USDCHF=X, USDCAD=X, USDSEK=X, USDNOK=X

## Metodologia

### 1. Generazione Segnali (Zero Lookahead)
```python
# Calcolo momentum k-mesi con logica temporale corretta
cumulative_returns = (1 + monthly_returns).rolling(k).apply(np.prod) - 1
# Al mese t: usa ritorni fino alla fine del mese t-1 (appena concluso)
signals = np.sign(cumulative_returns)  # Nessun lag aggiuntivo necessario
```

### 2. Position Sizing
```python
# Volatility targeting
weights = signals * (target_vol / vol_estimates)
```

### 3. Portfolio Construction
- Equal-weight cross-asset
- Sub-portfolios: commodities, forex, totale
- Tracking numero asset attivi

### 4. Gestione FX
Normalizzazione coppie USD-base (USDXXX):
```python
# Inversione per interpretazione consistente XXX vs USD
normalized_price = 1.0 / usd_base_price
```

## Vantaggi del Caching

### Performance
- **Prima esecuzione**: ~2-3 minuti (download + calcoli)
- **Esecuzioni successive**: ~30 secondi (solo calcoli)
- **Cache validity**: 24 ore (refresh automatico)

### Gestione Cache
```python
# In data_loader.py
data_loader.clear_cache()        # Cancella cache
data_loader.get_cache_info()     # Info cache
load_all_data(force_refresh=True) # Forza refresh
```

## Output Generati

### 1. Serie Temporali (`tsmom_yahoo_2000.csv`)
- Returns portfolio totale, commodities, forex
- Numero asset attivi per mese
- Sample posizioni per audit

### 2. Visualizzazioni
- **Equity Curves**: Performance cumulativa per tutti i sub-portfolio
- **Drawdown Analysis**: Analisi rischio dettagliata

### 3. Metriche Performance
- CAGR, Volatilità, Sharpe Ratio
- Max Drawdown, Calmar Ratio
- Win Rate, Skewness, Kurtosis

### 4. Test Robustezza
Grid analysis con k∈{3,6,9,12}, h∈{1,3,6}:

```
Sharpe Ratio Grid (esempio):
h       1      3      6
k                      
3   0.085  0.122  0.089
6   0.082  0.079  0.094
9  -0.123  0.040  0.169
12  0.219  0.170  0.214
```

## Esempio Risultati

Per il periodo 2000-2025 (24+ anni):

**Portfolio Totale:**
- CAGR: 2.44%
- Volatilità: 21.38%
- Sharpe: 0.22
- Max Drawdown: -62.42%

**Commodities:**
- CAGR: 2.64%
- Sharpe: 0.23

**Forex:**
- CAGR: -3.15%
- Sharpe: 0.04

## Validazioni Implementate

### 1. Lookahead Bias Check
- **Logica temporale**: Mese t usa solo info fino alla fine del mese t-1
- **Tracciamento lag**: Verifiche automatiche dei lag applicati
- **Validazione allineamento**: Controllo coerenza temporale dati

### 2. Qualità Dati
- Controlli missing data e forward-fill limitato
- Validazione serie temporali e range dates
- Correlazione volatilità stimata vs realizzata

### 3. Performance Monitoring
- Statistiche esecuzione e timing operazioni critiche
- Tracciamento memoria e utilizzo cache
- Logging ottimizzato per debug

## Dipendenze

```bash
pip install yfinance pandas numpy matplotlib
```

## Note Implementative

### Aderenza al Paper
- **Equazione (5)**: Position sizing con vol-targeting 40%
- **Setup TSMOM**: Sign-rule a 12 mesi con logica temporale corretta
- **Stima EWMA**: Center-of-mass ≈ 60 giorni per volatilità
- **Equal-weight**: Media cross-asset senza bias settoriali

### Ottimizzazioni Tecniche
- **5-10x più veloce** grazie a vettorizzazione completa
- **Zero lookahead bias** con logica temporale rigorosa
- **Cache system** per efficienza massima nelle esecuzioni
- **Memory efficient** con allineamento dati ottimizzato
- **Robusto** con extensive error handling

### Estensioni Possibili
- Transaction costs modeling (1-2 bps per turn)
- T-bill excess returns integration (^IRX)
- Multi-asset risk budgeting avanzato
- Alternative volatility estimators (GARCH, etc.)

---

**Implementazione completa della strategia TSMOM con focus su rigore temporale, zero lookahead bias, caching efficiente e performance ottimali.**