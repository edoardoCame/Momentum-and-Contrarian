"""
Returns Calculator per TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012)
Gestisce il calcolo dei rendimenti (daily -> monthly) e degli excess returns.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
import logging

warnings.filterwarnings('ignore')

class TSMOMReturnsCalculator:
    """
    Calcolatore dei rendimenti per la strategia TSMOM seguendo le specifiche MOP (2012).
    
    Funzionalità chiave:
    - Conversione daily -> monthly usando business month-end (ultimo giorno lavorativo)
    - Calcolo excess returns: r_m - rf_m
    - Gestione timezone naive come richiesto 
    - Allineamento temporale preciso tra asset returns e risk-free rate
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.daily_returns = None
        self.monthly_returns = None
        self.monthly_excess_returns = None
    
    def calculate_daily_returns(self, price_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola i rendimenti giornalieri semplici dai prezzi.
        
        Formula: r_t = (P_t / P_{t-1}) - 1
        
        Args:
            price_matrix: DataFrame con Date x Tickers dei prezzi
            
        Returns:
            DataFrame dei rendimenti giornalieri
        """
        self.logger.info("📊 Calcolo rendimenti giornalieri...")
        
        # Calcolo rendimenti semplici
        daily_returns = price_matrix.pct_change()
        
        # Rimuovi prima riga (NaN dal pct_change)
        daily_returns = daily_returns.dropna(how='all')
        
        # Basic validation
        self._validate_returns(daily_returns, "daily")
        
        self.daily_returns = daily_returns
        
        self.logger.info(f"✅ Daily returns: {daily_returns.shape[0]} dates x {daily_returns.shape[1]} tickers")
        self.logger.info(f"📅 Period: {daily_returns.index.min().date()} -> {daily_returns.index.max().date()}")
        
        return daily_returns
    
    def convert_to_monthly_returns(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Converte rendimenti giornalieri in mensili seguendo MOP (2012).
        
        Metodologia:
        1. Raggruppa per ultimo business day del mese ('BM')
        2. Usa rendimenti semplici composti: (1+r1)*(1+r2)*...*(1+rn) - 1
        3. Timezone naive handling come specificato
        
        Args:
            daily_returns: DataFrame dei rendimenti giornalieri
            
        Returns:
            DataFrame dei rendimenti mensili
        """
        self.logger.info("📊 Conversione daily -> monthly returns...")
        
        # Assicura timezone naive index
        if daily_returns.index.tz is not None:
            daily_returns.index = daily_returns.index.tz_localize(None)
        
        # Converti in monthly usando business month-end
        # Formula compounding: prod(1 + daily_returns) - 1
        monthly_returns = daily_returns.groupby(pd.Grouper(freq='BM')).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Rimuovi righe con tutti NaN
        monthly_returns = monthly_returns.dropna(how='all')
        
        # Validation
        self._validate_returns(monthly_returns, "monthly")
        
        self.monthly_returns = monthly_returns
        
        self.logger.info(f"✅ Monthly returns: {monthly_returns.shape[0]} months x {monthly_returns.shape[1]} tickers")
        self.logger.info(f"📅 Period: {monthly_returns.index.min().date()} -> {monthly_returns.index.max().date()}")
        
        return monthly_returns
    
    def calculate_excess_returns(self, 
                               monthly_returns: pd.DataFrame,
                               risk_free_monthly: pd.Series) -> pd.DataFrame:
        """
        Calcola excess returns: r_m - rf_m seguendo MOP (2012).
        
        Key points:
        1. Allinea temporalmente monthly returns e risk-free rate
        2. Sottrae rf_m da ogni asset return nello stesso mese
        3. Gestisce missing data in modo robusto
        
        Args:
            monthly_returns: DataFrame rendimenti mensili
            risk_free_monthly: Serie risk-free rate mensile
            
        Returns:
            DataFrame excess returns mensili
        """
        self.logger.info("📊 Calcolo excess returns (r_m - rf_m)...")
        
        # Allinea indici temporali
        common_dates = monthly_returns.index.intersection(risk_free_monthly.index)
        
        if len(common_dates) == 0:
            raise ValueError("Nessuna data comune tra returns e risk-free rate!")
        
        # Align data
        aligned_returns = monthly_returns.loc[common_dates]
        aligned_rf = risk_free_monthly.loc[common_dates]
        
        # Calcola excess returns: sottrai rf da ogni colonna
        excess_returns = aligned_returns.subtract(aligned_rf, axis=0)
        
        # Remove rows con tutti NaN
        excess_returns = excess_returns.dropna(how='all')
        
        # Validation
        self._validate_returns(excess_returns, "excess")
        
        self.monthly_excess_returns = excess_returns
        
        self.logger.info(f"✅ Excess returns: {excess_returns.shape[0]} months x {excess_returns.shape[1]} tickers")
        self.logger.info(f"📊 Risk-free media: {aligned_rf.mean()*100:.3f}% mensile")
        self.logger.info(f"📅 Period: {excess_returns.index.min().date()} -> {excess_returns.index.max().date()}")
        
        return excess_returns
    
    def get_returns_summary(self) -> dict:
        """
        Genera summary statistico dei rendimenti calcolati.
        
        Returns:
            Dict con statistiche descrittive
        """
        summary = {}
        
        if self.monthly_excess_returns is not None:
            returns_data = self.monthly_excess_returns
            
            summary = {
                "shape": returns_data.shape,
                "period": {
                    "start": returns_data.index.min().date(),
                    "end": returns_data.index.max().date(),
                    "months": len(returns_data)
                },
                "statistics": {
                    "mean_monthly": returns_data.mean().mean(),
                    "std_monthly": returns_data.std().mean(),
                    "min_return": returns_data.min().min(),
                    "max_return": returns_data.max().max(),
                    "missing_data_pct": (returns_data.isnull().sum().sum() / returns_data.size) * 100
                },
                "annualized_metrics": {
                    "mean_annual": returns_data.mean().mean() * 12,
                    "volatility_annual": returns_data.std().mean() * np.sqrt(12)
                }
            }
        
        return summary
    
    def _validate_returns(self, returns_data: pd.DataFrame, return_type: str):
        """
        Valida la qualità dei rendimenti calcolati.
        
        Args:
            returns_data: DataFrame dei rendimenti
            return_type: Tipo di returns ("daily", "monthly", "excess")
        """
        # Check basic shape
        if returns_data.empty:
            raise ValueError(f"DataFrame {return_type} returns è vuoto!")
        
        # Check per valori estremi
        abs_returns = returns_data.abs()
        
        if return_type == "daily":
            # Daily returns: outlier se > 50% in un giorno
            extreme_threshold = 0.5
        elif return_type == "monthly":
            # Monthly returns: outlier se > 200% in un mese  
            extreme_threshold = 2.0
        else:
            # Excess returns: simile ai monthly
            extreme_threshold = 2.0
        
        extreme_count = (abs_returns > extreme_threshold).sum().sum()
        if extreme_count > 0:
            self.logger.warning(f"⚠️ {extreme_count} valori estremi trovati nei {return_type} returns (>{extreme_threshold*100:.0f}%)")
        
        # Check per NaN
        nan_count = returns_data.isnull().sum().sum()
        total_size = returns_data.size
        nan_pct = (nan_count / total_size) * 100
        
        if nan_pct > 10:
            self.logger.warning(f"⚠️ {nan_pct:.1f}% di NaN nei {return_type} returns")
        
        # Check per infinite values
        inf_count = np.isinf(returns_data).sum().sum()
        if inf_count > 0:
            self.logger.error(f"❌ {inf_count} valori infiniti nei {return_type} returns!")
            raise ValueError(f"Valori infiniti trovati nei {return_type} returns")
    
    def align_with_business_month_end(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Allinea i dati all'ultimo business day del mese come richiesto da MOP (2012).
        
        Args:
            data: DataFrame da allineare
            
        Returns:
            DataFrame allineato a business month-end
        """
        # Resample all'ultimo business day del mese
        aligned_data = data.resample('BM').last()
        
        # Forward fill per eventuali NaN introdotti dal resampling
        aligned_data = aligned_data.fillna(method='ffill')
        
        return aligned_data
    
    def get_correlation_matrix(self, lookback_months: Optional[int] = None) -> pd.DataFrame:
        """
        Calcola la matrice di correlazione dei rendimenti mensili.
        
        Args:
            lookback_months: Periodo per rolling correlation (default: tutto il sample)
            
        Returns:
            Matrice di correlazione
        """
        if self.monthly_excess_returns is None:
            raise ValueError("Calcola prima gli excess returns!")
        
        if lookback_months is None:
            # Correlazione su tutto il sample
            corr_matrix = self.monthly_excess_returns.corr()
        else:
            # Rolling correlation - prendi ultima finestra
            corr_matrix = self.monthly_excess_returns.rolling(lookback_months).corr().iloc[-self.monthly_excess_returns.shape[1]:]
        
        return corr_matrix
    
    def export_returns_data(self, output_path: str):
        """
        Esporta i dati dei rendimenti calcolati.
        
        Args:
            output_path: Path base per i file di output
        """
        if self.monthly_excess_returns is not None:
            # Excess returns (il più importante per TSMOM)
            self.monthly_excess_returns.to_csv(f"{output_path}_monthly_excess_returns.csv")
            self.monthly_excess_returns.to_parquet(f"{output_path}_monthly_excess_returns.parquet")
            
        if self.monthly_returns is not None:
            # Total returns
            self.monthly_returns.to_csv(f"{output_path}_monthly_total_returns.csv")
            
        if self.daily_returns is not None:
            # Daily returns (per reference)
            self.daily_returns.to_parquet(f"{output_path}_daily_returns.parquet")
            
        self.logger.info(f"💾 Returns data esportati in {output_path}_*")