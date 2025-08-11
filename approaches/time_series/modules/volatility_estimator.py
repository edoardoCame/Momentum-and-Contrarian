"""
Volatility Estimator per TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012)
Implementa EWMA volatility estimation con le specifiche esatte del paper.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')

class TSMOMVolatilityEstimator:
    """
    Stimatore della volatilità ex-ante per TSMOM seguendo MOP (2012).
    
    Specifiche del paper:
    - EWMA su rendimenti giornalieri con center of mass = 60 giorni
    - Annualizzazione con √261 (trading days)
    - Lag di 1 mese per evitare look-ahead bias (σ_{t-1} per decisioni al tempo t)
    - Estrazione a fine mese (business month-end)
    """
    
    def __init__(self, 
                 center_of_mass: int = 60,
                 annualization_factor: float = np.sqrt(261)):
        """
        Inizializza il volatility estimator.
        
        Args:
            center_of_mass: Center of mass per EWMA (default: 60 come nel paper)
            annualization_factor: Fattore di annualizzazione (default: √261)
        """
        self.center_of_mass = center_of_mass
        self.annualization_factor = annualization_factor
        self.logger = logging.getLogger(__name__)
        
        # Storage per volatility data
        self.daily_volatility = None
        self.monthly_volatility = None
        self.lagged_monthly_volatility = None
    
    def calculate_daily_ewma_volatility(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola volatilità EWMA giornaliera seguendo MOP (2012).
        
        Formula EWMA:
        - Variance: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
        - λ derivato da center_of_mass: λ = COM/(COM+1)
        - Standard deviation: σ_t = √σ²_t
        - Annualized: σ_ann = σ_daily * √261
        
        Args:
            daily_returns: DataFrame dei rendimenti giornalieri
            
        Returns:
            DataFrame delle volatilità giornaliere annualizzate
        """
        self.logger.info(f"📊 Calcolo EWMA volatility (COM={self.center_of_mass})...")
        
        # Calcola EWMA variance per ogni ticker
        ewma_variance = daily_returns.ewm(
            com=self.center_of_mass, 
            adjust=False  # Importante: False per usare exponentially weighted moments
        ).var()
        
        # Convert to standard deviation
        ewma_std = np.sqrt(ewma_variance)
        
        # Annualizza usando il fattore specificato
        annualized_volatility = ewma_std * self.annualization_factor
        
        # Validation
        self._validate_volatility_data(annualized_volatility, "daily")
        
        self.daily_volatility = annualized_volatility
        
        self.logger.info(f"✅ Daily EWMA volatility: {annualized_volatility.shape}")
        self.logger.info(f"📊 Vol range: {annualized_volatility.min().min():.1%} - {annualized_volatility.max().max():.1%}")
        
        return annualized_volatility
    
    def extract_monthly_volatility(self, daily_volatility: pd.DataFrame) -> pd.DataFrame:
        """
        Estrae volatilità mensile dall'EWMA giornaliera.
        
        Metodologia MOP (2012):
        1. Prende l'ultimo valore di volatilità giornaliera del mese
        2. Usa business month-end ('BM') per allineamento
        3. Forward fill per eventuali gap
        
        Args:
            daily_volatility: DataFrame volatilità giornaliere annualizzate
            
        Returns:
            DataFrame volatilità mensili
        """
        self.logger.info("📊 Estrazione monthly volatility da daily EWMA...")
        
        # Estrai ultimo valore del mese usando business month-end
        monthly_vol = daily_volatility.resample('BM').last()
        
        # Forward fill per eventuali NaN
        monthly_vol = monthly_vol.fillna(method='ffill')
        
        # Remove righe con tutti NaN
        monthly_vol = monthly_vol.dropna(how='all')
        
        # Validation
        self._validate_volatility_data(monthly_vol, "monthly")
        
        self.monthly_volatility = monthly_vol
        
        self.logger.info(f"✅ Monthly volatility: {monthly_vol.shape}")
        self.logger.info(f"📅 Period: {monthly_vol.index.min().date()} -> {monthly_vol.index.max().date()}")
        
        return monthly_vol
    
    def apply_lag_for_position_sizing(self, monthly_volatility: pd.DataFrame) -> pd.DataFrame:
        """
        Applica lag di 1 mese per evitare look-ahead bias.
        
        CRITICO per MOP (2012): La volatilità usata per il position sizing al tempo t
        deve essere basata solo su informazioni disponibili fino al tempo t-1.
        
        Formula: σ_lagged(t) = σ(t-1)
        
        Args:
            monthly_volatility: DataFrame volatilità mensili
            
        Returns:
            DataFrame volatilità laggata di 1 mese
        """
        self.logger.info("📊 Applicazione lag per position sizing (σ_{t-1})...")
        
        # Shift di 1 periodo per lag
        lagged_volatility = monthly_volatility.shift(1)
        
        # Remove righe con tutti NaN (primo mese avrà tutti NaN dopo shift)
        lagged_volatility = lagged_volatility.dropna(how='all')
        
        # Validation
        self._validate_volatility_data(lagged_volatility, "lagged")
        
        self.lagged_monthly_volatility = lagged_volatility
        
        self.logger.info(f"✅ Lagged volatility: {lagged_volatility.shape}")
        self.logger.info(f"📅 Available from: {lagged_volatility.index.min().date()}")
        
        # Check che il lag sia applicato correttamente
        if len(lagged_volatility) == len(monthly_volatility):
            self.logger.warning("⚠️ Lag potrebbe non essere applicato correttamente!")
        else:
            self.logger.info(f"🔄 Lag applicato: {len(monthly_volatility)} -> {len(lagged_volatility)} observations")
        
        return lagged_volatility
    
    def get_volatility_statistics(self) -> dict:
        """
        Genera statistiche descrittive della volatilità calcolata.
        
        Returns:
            Dict con statistiche dettagliate
        """
        if self.lagged_monthly_volatility is None:
            return {"error": "Calcola prima la volatilità laggata"}
        
        vol_data = self.lagged_monthly_volatility
        
        stats = {
            "shape": vol_data.shape,
            "period": {
                "start": vol_data.index.min().date(),
                "end": vol_data.index.max().date(),
                "months": len(vol_data)
            },
            "cross_sectional": {
                "mean_volatility": vol_data.mean().mean(),
                "median_volatility": vol_data.median().median(),
                "min_volatility": vol_data.min().min(),
                "max_volatility": vol_data.max().max(),
                "std_of_volatility": vol_data.std().mean()
            },
            "time_series": {
                "average_vol_per_month": vol_data.mean(axis=1).describe().to_dict(),
                "vol_stability": vol_data.mean(axis=1).std(),
                "missing_data_pct": (vol_data.isnull().sum().sum() / vol_data.size) * 100
            }
        }
        
        # Per-ticker statistics
        stats["per_ticker"] = {}
        for ticker in vol_data.columns:
            ticker_vol = vol_data[ticker].dropna()
            if len(ticker_vol) > 0:
                stats["per_ticker"][ticker] = {
                    "mean": ticker_vol.mean(),
                    "std": ticker_vol.std(),
                    "min": ticker_vol.min(),
                    "max": ticker_vol.max(),
                    "observations": len(ticker_vol)
                }
        
        return stats
    
    def _validate_volatility_data(self, vol_data: pd.DataFrame, vol_type: str):
        """
        Valida la qualità dei dati di volatilità.
        
        Args:
            vol_data: DataFrame delle volatilità
            vol_type: Tipo ("daily", "monthly", "lagged")
        """
        # Check basic shape
        if vol_data.empty:
            raise ValueError(f"DataFrame {vol_type} volatility è vuoto!")
        
        # Check per valori negativi (impossibili per volatility)
        negative_count = (vol_data < 0).sum().sum()
        if negative_count > 0:
            raise ValueError(f"{negative_count} volatilità negative trovate in {vol_type}!")
        
        # Check per valori zero (problematici per position sizing)
        zero_count = (vol_data == 0).sum().sum()
        if zero_count > 0:
            self.logger.warning(f"⚠️ {zero_count} volatilità zero trovate in {vol_type}")
        
        # Check per valori estremi
        # Volatility annualized ragionevole: 5% - 200%
        min_reasonable = 0.05  # 5% annuo
        max_reasonable = 2.0   # 200% annuo
        
        too_low = (vol_data < min_reasonable).sum().sum()
        too_high = (vol_data > max_reasonable).sum().sum()
        
        if too_low > 0:
            self.logger.warning(f"⚠️ {too_low} volatilità sospettosamente basse (<{min_reasonable:.1%}) in {vol_type}")
        if too_high > 0:
            self.logger.warning(f"⚠️ {too_high} volatilità sospettosamente alte (>{max_reasonable:.1%}) in {vol_type}")
        
        # Check per NaN
        nan_count = vol_data.isnull().sum().sum()
        nan_pct = (nan_count / vol_data.size) * 100
        if nan_pct > 5:
            self.logger.warning(f"⚠️ {nan_pct:.1f}% di NaN in {vol_type} volatility")
    
    def calculate_position_sizing_weights(self, 
                                        target_volatility: float = 0.40) -> pd.DataFrame:
        """
        Calcola i pesi per position sizing usando volatility scaling.
        
        Formula MOP (2012): weight = target_vol / σ_{t-1}
        
        Args:
            target_volatility: Target volatility per singolo contratto (default: 40%)
            
        Returns:
            DataFrame dei moltiplicatori per position sizing
        """
        if self.lagged_monthly_volatility is None:
            raise ValueError("Calcola prima la lagged volatility!")
        
        self.logger.info(f"📊 Calcolo position sizing weights (target vol={target_volatility:.1%})...")
        
        # Calcola scaling weights: target_vol / realized_vol
        sizing_weights = target_volatility / self.lagged_monthly_volatility
        
        # Handle division by zero o valori estremi
        sizing_weights = sizing_weights.replace([np.inf, -np.inf], np.nan)
        
        # Cap estremi per stabilità (opcional safety measure)
        MAX_WEIGHT = 10.0  # Massimo 10x leverage per singolo contratto
        sizing_weights = sizing_weights.clip(upper=MAX_WEIGHT)
        
        self.logger.info(f"✅ Position sizing weights: {sizing_weights.shape}")
        self.logger.info(f"📊 Weight range: {sizing_weights.min().min():.2f} - {sizing_weights.max().max():.2f}")
        
        return sizing_weights
    
    def export_volatility_data(self, output_path: str):
        """
        Esporta i dati di volatilità calcolati.
        
        Args:
            output_path: Path base per i file di output
        """
        if self.lagged_monthly_volatility is not None:
            # Lagged monthly volatility (il più importante per TSMOM)
            self.lagged_monthly_volatility.to_csv(f"{output_path}_monthly_volatility_lagged.csv")
            self.lagged_monthly_volatility.to_parquet(f"{output_path}_monthly_volatility_lagged.parquet")
            
        if self.monthly_volatility is not None:
            # Unlagged monthly volatility (per reference)  
            self.monthly_volatility.to_csv(f"{output_path}_monthly_volatility.csv")
            
        if self.daily_volatility is not None:
            # Daily volatility (per detailed analysis)
            self.daily_volatility.to_parquet(f"{output_path}_daily_volatility.parquet")
            
        self.logger.info(f"💾 Volatility data esportati in {output_path}_*")
    
    def plot_volatility_time_series(self, tickers: Optional[list] = None, 
                                   figsize: Tuple[int, int] = (15, 10)):
        """
        Plotta time series delle volatilità per visualizzazione.
        
        Args:
            tickers: Lista tickers da plottare (default: primi 6)
            figsize: Dimensione figura
        """
        if self.daily_volatility is None:
            self.logger.error("Calcola prima la daily volatility!")
            return
            
        import matplotlib.pyplot as plt
        
        if tickers is None:
            tickers = self.daily_volatility.columns[:6]  # Primi 6 tickers
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, ticker in enumerate(tickers[:6]):
            if ticker in self.daily_volatility.columns:
                ax = axes[i]
                
                # Daily volatility
                self.daily_volatility[ticker].plot(ax=ax, alpha=0.7, label='Daily EWMA')
                
                # Monthly volatility (se disponibile)
                if self.monthly_volatility is not None:
                    self.monthly_volatility[ticker].plot(ax=ax, marker='o', 
                                                        linewidth=2, label='Monthly')
                
                ax.set_title(f'{ticker} - Annualized Volatility')
                ax.set_ylabel('Volatility')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format y-axis as percentage
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        plt.show()