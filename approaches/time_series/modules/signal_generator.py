"""
Signal Generator per TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012)
Implementa la logica di generazione segnali seguendo esattamente le regole del paper.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')

class TSMOMSignalGenerator:
    """
    Generatore segnali TSMOM seguendo Moskowitz, Ooi & Pedersen (2012).
    
    Regole chiave del paper:
    - Lookback period: 12 mesi (k=12)
    - Holding period: 1 mese (h=1) 
    - Skip ultimo mese: segnale basato su rendimenti da t-12 a t-1 (esclude mese corrente)
    - Segnale: sign(cumulative excess return over 12m) → +1 long, -1 short
    - Ribilanciamento: mensile senza overlapping
    """
    
    def __init__(self, 
                 lookback_months: int = 12,
                 holding_months: int = 1):
        """
        Inizializza il signal generator.
        
        Args:
            lookback_months: Periodo lookback in mesi (default: 12 come nel paper)
            holding_months: Periodo holding in mesi (default: 1 come nel paper)
        """
        self.lookback_months = lookback_months
        self.holding_months = holding_months
        self.logger = logging.getLogger(__name__)
        
        # Storage per signal data
        self.momentum_cumulative = None
        self.tsmom_signals = None
        self.signal_statistics = None
    
    def calculate_cumulative_momentum(self, monthly_excess_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola momentum cumulativo su 12 mesi seguendo MOP (2012).
        
        Formula chiave:
        - Rendimento cumulativo = Π(1 + r_t) - 1 per t da (t-12) a (t-1)
        - Implementazione vettorizzata: (1 + excess_returns).shift(1).rolling(12).apply(np.prod) - 1
        - shift(1) assicura che usiamo solo informazioni fino a t-1
        
        Args:
            monthly_excess_returns: DataFrame excess returns mensili
            
        Returns:
            DataFrame del momentum cumulativo 12M
        """
        self.logger.info(f"📊 Calcolo momentum cumulativo ({self.lookback_months}M lookback)...")
        
        # CRITICO: shift(1) per evitare look-ahead bias
        # Usiamo informazioni fino a t-1 per decisioni al tempo t
        lagged_returns = monthly_excess_returns.shift(1)
        
        # Calcola rendimento cumulativo usando compounding
        # Formula: Π(1 + r_i) - 1 = rolling product di (1 + returns) - 1
        cumulative_momentum = (
            (1 + lagged_returns)
            .rolling(window=self.lookback_months, min_periods=self.lookback_months)
            .apply(np.prod, raw=True)
        ) - 1
        
        # Rimuovi righe con tutti NaN (primi 12+1 mesi non avranno segnali)
        cumulative_momentum = cumulative_momentum.dropna(how='all')
        
        # Validation
        self._validate_momentum_data(cumulative_momentum)
        
        self.momentum_cumulative = cumulative_momentum
        
        self.logger.info(f"✅ Cumulative momentum: {cumulative_momentum.shape}")
        self.logger.info(f"📅 Segnali disponibili da: {cumulative_momentum.index.min().date()}")
        
        return cumulative_momentum
    
    def generate_tsmom_signals(self, cumulative_momentum: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali TSMOM dal momentum cumulativo.
        
        Regola MOP (2012):
        - Signal = sign(cumulative_momentum_12M)
        - +1 se momentum > 0 (long position)
        - -1 se momentum < 0 (short position)  
        - 0 se momentum = 0 esatto (neutral, raro)
        
        Args:
            cumulative_momentum: DataFrame momentum cumulativo
            
        Returns:
            DataFrame dei segnali TSMOM (+1, -1, 0)
        """
        self.logger.info("📊 Generazione segnali TSMOM...")
        
        # Applica regola sign-based
        signals = np.sign(cumulative_momentum)
        
        # Converti in int per chiarezza (opzionale)
        signals = signals.astype('int8')
        
        # Validation 
        self._validate_signal_data(signals)
        
        self.tsmom_signals = signals
        
        self.logger.info(f"✅ TSMOM signals: {signals.shape}")
        
        # Signal distribution statistics
        signal_stats = self._calculate_signal_statistics(signals)
        self.signal_statistics = signal_stats
        
        return signals
    
    def _calculate_signal_statistics(self, signals: pd.DataFrame) -> dict:
        """
        Calcola statistiche descrittive dei segnali generati.
        
        Args:
            signals: DataFrame dei segnali
            
        Returns:
            Dict con statistiche dettagliate
        """
        # Flatten tutti i segnali per statistiche aggregate
        all_signals = signals.values.flatten()
        all_signals = all_signals[~np.isnan(all_signals)]  # Rimuovi NaN
        
        total_signals = len(all_signals)
        long_signals = np.sum(all_signals == 1)
        short_signals = np.sum(all_signals == -1)
        neutral_signals = np.sum(all_signals == 0)
        
        stats = {
            "total_observations": total_signals,
            "signal_distribution": {
                "long_count": long_signals,
                "short_count": short_signals,
                "neutral_count": neutral_signals,
                "long_pct": long_signals / total_signals * 100,
                "short_pct": short_signals / total_signals * 100,
                "neutral_pct": neutral_signals / total_signals * 100
            },
            "per_ticker": {},
            "time_series": {
                "avg_long_positions_per_month": signals.apply(lambda x: (x == 1).sum(), axis=1).mean(),
                "avg_short_positions_per_month": signals.apply(lambda x: (x == -1).sum(), axis=1).mean(),
                "avg_active_positions_per_month": signals.apply(lambda x: (x != 0).sum(), axis=1).mean()
            }
        }
        
        # Per-ticker statistics
        for ticker in signals.columns:
            ticker_signals = signals[ticker].dropna()
            if len(ticker_signals) > 0:
                stats["per_ticker"][ticker] = {
                    "total": len(ticker_signals),
                    "long": (ticker_signals == 1).sum(),
                    "short": (ticker_signals == -1).sum(),
                    "neutral": (ticker_signals == 0).sum(),
                    "long_pct": (ticker_signals == 1).mean() * 100
                }
        
        self.logger.info(f"📊 Signal stats: {long_signals/total_signals:.1%} long, {short_signals/total_signals:.1%} short")
        
        return stats
    
    def validate_look_ahead_bias_prevention(self, 
                                          monthly_excess_returns: pd.DataFrame,
                                          sample_months: int = 3) -> pd.DataFrame:
        """
        Valida che il look-ahead bias sia effettivamente prevenuto.
        
        Mostra esempio di come i segnali sono generati per confermare che:
        1. Il segnale al mese t usa solo dati fino a t-1
        2. L'ultimo mese è escluso dal lookback
        3. Il timing è corretto
        
        Args:
            monthly_excess_returns: DataFrame excess returns
            sample_months: Numero di mesi da mostrare nell'esempio
            
        Returns:
            DataFrame con esempio di validation
        """
        self.logger.info("🔍 Validazione look-ahead bias prevention...")
        
        if self.momentum_cumulative is None or self.tsmom_signals is None:
            raise ValueError("Genera prima i segnali TSMOM!")
        
        # Prendi sample dates per validation
        available_dates = self.tsmom_signals.dropna(how='all').index
        if len(available_dates) < sample_months:
            sample_dates = available_dates
        else:
            sample_dates = available_dates[-sample_months:]  # Ultimi N mesi
        
        # Seleziona alcuni tickers rappresentativi
        sample_tickers = self.tsmom_signals.columns[:3]  # Primi 3 tickers
        
        validation_data = []
        
        for date in sample_dates:
            for ticker in sample_tickers:
                # Prendi i dati storici per questo punto temporale
                lookback_start = date - pd.DateOffset(months=self.lookback_months)
                lookback_end = date - pd.DateOffset(months=1)  # Escludi ultimo mese
                
                # Estrai returns nel periodo di lookback
                historical_returns = monthly_excess_returns.loc[
                    (monthly_excess_returns.index >= lookback_start) & 
                    (monthly_excess_returns.index <= lookback_end), ticker
                ].dropna()
                
                if len(historical_returns) > 0:
                    # Calcola momentum per validation
                    cum_return = (1 + historical_returns).prod() - 1
                    signal = np.sign(cum_return)
                    
                    validation_data.append({
                        'Date': date,
                        'Ticker': ticker,
                        'Lookback_Start': lookback_start.date(),
                        'Lookback_End': lookback_end.date(),
                        'Months_Used': len(historical_returns),
                        'Cumulative_Return': cum_return,
                        'Signal_Generated': signal,
                        'Signal_From_Model': self.tsmom_signals.loc[date, ticker] if date in self.tsmom_signals.index else np.nan
                    })
        
        validation_df = pd.DataFrame(validation_data)
        
        # Check consistenza
        if not validation_df.empty:
            consistent_signals = validation_df['Signal_Generated'] == validation_df['Signal_From_Model']
            consistency_pct = consistent_signals.mean() * 100
            
            self.logger.info(f"✅ Look-ahead bias validation: {consistency_pct:.1f}% consistenza")
            if consistency_pct < 95:
                self.logger.warning("⚠️ Possibili problemi di look-ahead bias!")
        
        return validation_df
    
    def _validate_momentum_data(self, momentum_data: pd.DataFrame):
        """
        Valida il momentum cumulativo calcolato.
        
        Args:
            momentum_data: DataFrame momentum cumulativo
        """
        if momentum_data.empty:
            raise ValueError("Momentum cumulativo è vuoto!")
        
        # Check per valori estremi (>1000% o <-90%)
        extreme_positive = (momentum_data > 10.0).sum().sum()
        extreme_negative = (momentum_data < -0.9).sum().sum()
        
        if extreme_positive > 0:
            self.logger.warning(f"⚠️ {extreme_positive} momentum estremi positivi (>1000%)")
        if extreme_negative > 0:
            self.logger.warning(f"⚠️ {extreme_negative} momentum estremi negativi (<-90%)")
        
        # Check per infinite values
        inf_count = np.isinf(momentum_data).sum().sum()
        if inf_count > 0:
            raise ValueError(f"{inf_count} valori infiniti nel momentum!")
    
    def _validate_signal_data(self, signal_data: pd.DataFrame):
        """
        Valida i segnali generati.
        
        Args:
            signal_data: DataFrame dei segnali
        """
        if signal_data.empty:
            raise ValueError("Segnali TSMOM vuoti!")
        
        # Check che i segnali siano solo -1, 0, 1
        unique_values = np.unique(signal_data.values[~np.isnan(signal_data.values)])
        valid_signals = set([-1, 0, 1])
        
        invalid_signals = set(unique_values) - valid_signals
        if invalid_signals:
            raise ValueError(f"Segnali non validi trovati: {invalid_signals}")
        
        # Check per troppi NaN
        nan_pct = signal_data.isnull().sum().sum() / signal_data.size * 100
        if nan_pct > 15:  # Più del 15% NaN potrebbe indicare problemi
            self.logger.warning(f"⚠️ {nan_pct:.1f}% di NaN nei segnali")
    
    def export_signal_data(self, output_path: str):
        """
        Esporta i dati dei segnali generati.
        
        Args:
            output_path: Path base per i file di output
        """
        if self.tsmom_signals is not None:
            # Segnali TSMOM
            self.tsmom_signals.to_csv(f"{output_path}_tsmom_signals.csv")
            self.tsmom_signals.to_parquet(f"{output_path}_tsmom_signals.parquet")
            
        if self.momentum_cumulative is not None:
            # Momentum cumulativo (per analisi)
            self.momentum_cumulative.to_csv(f"{output_path}_momentum_12m.csv")
            
        if self.signal_statistics is not None:
            # Statistiche segnali
            import json
            with open(f"{output_path}_signal_statistics.json", 'w') as f:
                json.dump(self.signal_statistics, f, indent=2, default=str)
                
        self.logger.info(f"💾 Signal data esportati in {output_path}_*")
    
    def plot_signal_distribution(self, figsize: Tuple[int, int] = (15, 8)):
        """
        Visualizza la distribuzione dei segnali nel tempo.
        
        Args:
            figsize: Dimensione della figura
        """
        if self.tsmom_signals is None:
            self.logger.error("Genera prima i segnali TSMOM!")
            return
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Long/Short positions over time
        ax1 = axes[0, 0]
        monthly_long = (self.tsmom_signals == 1).sum(axis=1)
        monthly_short = (self.tsmom_signals == -1).sum(axis=1)
        
        ax1.plot(monthly_long.index, monthly_long.values, label='Long Positions', color='green')
        ax1.plot(monthly_short.index, monthly_short.values, label='Short Positions', color='red')
        ax1.set_title('Active Positions Over Time')
        ax1.set_ylabel('Number of Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Signal distribution histogram
        ax2 = axes[0, 1]
        all_signals = self.tsmom_signals.values.flatten()
        all_signals = all_signals[~np.isnan(all_signals)]
        
        signal_counts = pd.Series(all_signals).value_counts()
        bars = ax2.bar([-1, 0, 1], [signal_counts.get(-1, 0), signal_counts.get(0, 0), signal_counts.get(1, 0)],
                      color=['red', 'gray', 'green'])
        ax2.set_title('Overall Signal Distribution')
        ax2.set_xlabel('Signal Value')
        ax2.set_ylabel('Count')
        ax2.set_xticks([-1, 0, 1])
        ax2.set_xticklabels(['Short (-1)', 'Neutral (0)', 'Long (+1)'])
        
        # 3. Per-ticker signal frequency
        ax3 = axes[1, 0]
        ticker_long_pct = (self.tsmom_signals == 1).mean() * 100
        ticker_long_pct = ticker_long_pct.sort_values(ascending=True)
        
        ticker_long_pct.plot(kind='barh', ax=ax3, color='lightblue')
        ax3.set_title('Long Signal Frequency by Ticker')
        ax3.set_xlabel('Long Signal %')
        
        # 4. Time series di alcuni ticker
        ax4 = axes[1, 1]
        sample_tickers = self.tsmom_signals.columns[:3]
        
        for i, ticker in enumerate(sample_tickers):
            offset = i * 0.1  # Slight offset per visualizzazione
            ax4.scatter(self.tsmom_signals.index, 
                       self.tsmom_signals[ticker] + offset,
                       alpha=0.6, s=10, label=ticker)
        
        ax4.set_title('Signal Time Series (Sample Tickers)')
        ax4.set_ylabel('Signal Value')
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(['Short', 'Neutral', 'Long'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()