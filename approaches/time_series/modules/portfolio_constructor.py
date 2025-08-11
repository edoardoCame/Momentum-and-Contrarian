"""
Portfolio Constructor per TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012)
Implementa volatility scaling e aggregazione cross-sectional seguendo le specifiche del paper.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import warnings
import logging

warnings.filterwarnings('ignore')

class TSMOMPortfolioConstructor:
    """
    Costruttore del portafoglio TSMOM seguendo Moskowitz, Ooi & Pedersen (2012).
    
    Specifiche chiave del paper:
    - Target volatility = 40% annualizzata per singolo contratto
    - Weight = signal × (target_vol / σ_{t-1})
    - Aggregazione: equal-weight cross-sectional (media semplice)
    - Ribilanciamento mensile, holding period = 1 mese
    - Nessun overlapping di posizioni
    """
    
    def __init__(self, 
                 target_volatility: float = 0.40,
                 max_weight_per_contract: float = 10.0):
        """
        Inizializza il portfolio constructor.
        
        Args:
            target_volatility: Target vol annualizzata per contratto (default: 40% come MOP)
            max_weight_per_contract: Peso massimo per singolo contratto (safety cap)
        """
        self.target_volatility = target_volatility
        self.max_weight_per_contract = max_weight_per_contract
        self.logger = logging.getLogger(__name__)
        
        # Storage per portfolio data
        self.contract_weights = None
        self.portfolio_returns = None
        self.portfolio_weights_final = None
        self.turnover_data = None
    
    def calculate_volatility_scaled_weights(self, 
                                          signals: pd.DataFrame,
                                          volatilities: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola i pesi scalati per volatilità seguendo MOP (2012).
        
        Formula del paper: w_{s,t} = signal_{s,t} × (target_vol / σ_{s,t-1})
        
        Dove:
        - signal_{s,t} ∈ {-1, 0, +1} per contratto s al mese t
        - σ_{s,t-1} è la volatilità ex-ante (EWMA laggata)  
        - target_vol = 0.40 (40% annualizzato)
        
        Args:
            signals: DataFrame segnali TSMOM (+1, -1, 0)
            volatilities: DataFrame volatilità laggata (ex-ante)
            
        Returns:
            DataFrame pesi per singolo contratto
        """
        self.logger.info(f"📊 Calcolo volatility-scaled weights (target={self.target_volatility:.1%})...")
        
        # Allinea temporalmente signals e volatilities
        common_dates = signals.index.intersection(volatilities.index)
        common_tickers = signals.columns.intersection(volatilities.columns)
        
        if len(common_dates) == 0 or len(common_tickers) == 0:
            raise ValueError("Nessun allineamento temporale tra signals e volatilities!")
        
        # Align data
        aligned_signals = signals.loc[common_dates, common_tickers]
        aligned_volatilities = volatilities.loc[common_dates, common_tickers]
        
        # Calcola volatility scaling factor
        vol_scaling = self.target_volatility / aligned_volatilities
        
        # Applica segnali: weight = signal × scaling_factor
        contract_weights = aligned_signals * vol_scaling
        
        # Safety cap per evitare leverage estremo
        contract_weights = contract_weights.clip(-self.max_weight_per_contract, 
                                                self.max_weight_per_contract)
        
        # Handle NaN/Inf
        contract_weights = contract_weights.replace([np.inf, -np.inf], 0)
        contract_weights = contract_weights.fillna(0)
        
        # Validation
        self._validate_weights(contract_weights)
        
        self.contract_weights = contract_weights
        
        self.logger.info(f"✅ Contract weights: {contract_weights.shape}")
        self.logger.info(f"📊 Weight range: {contract_weights.min().min():.2f} to {contract_weights.max().max():.2f}")
        
        return contract_weights
    
    def construct_portfolio_returns(self, 
                                  contract_weights: pd.DataFrame,
                                  monthly_excess_returns: pd.DataFrame) -> pd.Series:
        """
        Costruisce i rendimenti del portafoglio TSMOM seguendo MOP (2012).
        
        Metodologia:
        1. Calcola rendimenti per contratto: ret_leg_{s,t+1} = w_{s,t} × rx_{s,t+1}
        2. Aggregazione equal-weight: portfolio_return = mean(ret_legs) 
        3. Skipna per gestire contratti assenti
        
        Args:
            contract_weights: DataFrame pesi per contratto al tempo t
            monthly_excess_returns: DataFrame excess returns al tempo t+1
            
        Returns:
            Serie dei rendimenti del portafoglio TSMOM
        """
        self.logger.info("📊 Costruzione portfolio returns (equal-weight aggregation)...")
        
        # Allinea per timing corretto: weights[t] × returns[t+1]
        # Shift weights forward di 1 mese per allineamento temporale
        weights_shifted = contract_weights.shift(1).dropna(how='all')
        
        # Trova overlap temporale
        common_dates = weights_shifted.index.intersection(monthly_excess_returns.index)
        common_tickers = weights_shifted.columns.intersection(monthly_excess_returns.columns)
        
        if len(common_dates) == 0:
            raise ValueError("Nessun allineamento temporale per portfolio construction!")
        
        # Align data
        aligned_weights = weights_shifted.loc[common_dates, common_tickers]
        aligned_returns = monthly_excess_returns.loc[common_dates, common_tickers]
        
        # Calcola rendimenti per leg: w[t] × r[t+1]
        contract_returns = aligned_weights * aligned_returns
        
        # Aggregazione equal-weight: media cross-sectional
        # skipna=True per gestire contratti con dati mancanti
        portfolio_returns = contract_returns.mean(axis=1, skipna=True)
        
        # Remove NaN (potrebbero esserci se tutti i contratti sono NaN in un mese)
        portfolio_returns = portfolio_returns.dropna()
        
        # Validation
        self._validate_portfolio_returns(portfolio_returns)
        
        self.portfolio_returns = portfolio_returns
        
        self.logger.info(f"✅ Portfolio returns: {len(portfolio_returns)} months")
        self.logger.info(f"📅 Period: {portfolio_returns.index.min().date()} -> {portfolio_returns.index.max().date()}")
        self.logger.info(f"📊 Mean monthly: {portfolio_returns.mean():.3%}, Std: {portfolio_returns.std():.3%}")
        
        return portfolio_returns
    
    def calculate_portfolio_turnover(self, contract_weights: pd.DataFrame) -> pd.Series:
        """
        Calcola il turnover mensile del portafoglio.
        
        Formula: turnover_t = mean(|w_{s,t} - w_{s,t-1}|) per tutti i contratti s
        
        Args:
            contract_weights: DataFrame pesi contratti
            
        Returns:
            Serie del turnover mensile
        """
        self.logger.info("📊 Calcolo portfolio turnover...")
        
        # Calcola variazioni assolute dei pesi mese su mese
        weight_changes = contract_weights.diff().abs()
        
        # Media cross-sectional del turnover
        monthly_turnover = weight_changes.mean(axis=1, skipna=True)
        
        # Remove NaN (primo mese)
        monthly_turnover = monthly_turnover.dropna()
        
        self.turnover_data = monthly_turnover
        
        avg_turnover = monthly_turnover.mean()
        self.logger.info(f"✅ Average monthly turnover: {avg_turnover:.3f}")
        
        return monthly_turnover
    
    def apply_transaction_costs(self, 
                              portfolio_returns: pd.Series,
                              turnover: pd.Series,
                              cost_bps: float = 0) -> pd.Series:
        """
        Applica costi di transazione ai rendimenti del portafoglio.
        
        Formula: net_return_t = gross_return_t - (turnover_t × cost_bps)
        
        Args:
            portfolio_returns: Serie rendimenti lordi
            turnover: Serie turnover mensile
            cost_bps: Costi in basis points per round-trip (default: 0)
            
        Returns:
            Serie rendimenti netti dopo costi
        """
        if cost_bps == 0:
            self.logger.info("📊 Nessun costo di transazione applicato")
            return portfolio_returns
        
        self.logger.info(f"📊 Applicazione costi transazione: {cost_bps} bps...")
        
        # Allinea temporalmente
        common_dates = portfolio_returns.index.intersection(turnover.index)
        aligned_returns = portfolio_returns.loc[common_dates]
        aligned_turnover = turnover.loc[common_dates]
        
        # Converti bps in decimale
        cost_rate = cost_bps / 10000
        
        # Sottrai costi: net = gross - (turnover × cost_rate)
        transaction_costs = aligned_turnover * cost_rate
        net_returns = aligned_returns - transaction_costs
        
        total_costs = transaction_costs.sum()
        self.logger.info(f"📊 Costi totali: {total_costs:.4f} ({total_costs*100:.2f}% cumulativo)")
        
        return net_returns
    
    def get_portfolio_statistics(self) -> Dict:
        """
        Calcola statistiche descrittive del portafoglio costruito.
        
        Returns:
            Dict con metriche del portafoglio
        """
        if self.portfolio_returns is None or self.contract_weights is None:
            return {"error": "Costruisci prima il portafoglio!"}
        
        returns = self.portfolio_returns
        weights = self.contract_weights
        
        stats = {
            "returns": {
                "observations": len(returns),
                "mean_monthly": returns.mean(),
                "std_monthly": returns.std(),
                "min_return": returns.min(),
                "max_return": returns.max(),
                "annualized_return": returns.mean() * 12,
                "annualized_volatility": returns.std() * np.sqrt(12)
            },
            "weights": {
                "avg_active_contracts": (weights != 0).sum(axis=1).mean(),
                "max_active_contracts": (weights != 0).sum(axis=1).max(),
                "avg_gross_exposure": weights.abs().sum(axis=1).mean(),
                "max_gross_exposure": weights.abs().sum(axis=1).max(),
                "avg_net_exposure": weights.sum(axis=1).mean()
            },
            "turnover": {},
            "period": {
                "start_date": returns.index.min().date(),
                "end_date": returns.index.max().date(),
                "total_months": len(returns)
            }
        }
        
        # Turnover stats se disponibili
        if self.turnover_data is not None:
            stats["turnover"] = {
                "mean_monthly": self.turnover_data.mean(),
                "median_monthly": self.turnover_data.median(),
                "max_monthly": self.turnover_data.max(),
                "annualized_turnover": self.turnover_data.mean() * 12
            }
        
        return stats
    
    def _validate_weights(self, weights: pd.DataFrame):
        """
        Valida i pesi calcolati per il portafoglio.
        
        Args:
            weights: DataFrame pesi contratti
        """
        if weights.empty:
            raise ValueError("Pesi contratti vuoti!")
        
        # Check per valori estremi
        max_abs_weight = weights.abs().max().max()
        if max_abs_weight > self.max_weight_per_contract * 1.1:  # Small tolerance
            self.logger.warning(f"⚠️ Peso estremo trovato: {max_abs_weight:.2f}")
        
        # Check per infinite/NaN dopo processing
        inf_count = np.isinf(weights).sum().sum()
        nan_count = weights.isnull().sum().sum()
        
        if inf_count > 0:
            raise ValueError(f"{inf_count} pesi infiniti dopo processing!")
        
        if nan_count / weights.size > 0.1:  # >10% NaN
            self.logger.warning(f"⚠️ {nan_count/weights.size:.1%} NaN nei pesi")
    
    def _validate_portfolio_returns(self, returns: pd.Series):
        """
        Valida i rendimenti del portafoglio.
        
        Args:
            returns: Serie rendimenti portafoglio
        """
        if returns.empty:
            raise ValueError("Rendimenti portafoglio vuoti!")
        
        # Check per valori estremi (>100% in un mese)
        extreme_returns = returns.abs() > 1.0
        if extreme_returns.any():
            extreme_count = extreme_returns.sum()
            self.logger.warning(f"⚠️ {extreme_count} rendimenti estremi (>100%) trovati")
        
        # Check per infinite/NaN
        if returns.isnull().any():
            nan_count = returns.isnull().sum()
            self.logger.warning(f"⚠️ {nan_count} NaN nei rendimenti portafoglio")
        
        if np.isinf(returns).any():
            raise ValueError("Rendimenti infiniti nel portafoglio!")
    
    def export_portfolio_data(self, output_path: str):
        """
        Esporta i dati del portafoglio costruito.
        
        Args:
            output_path: Path base per i file di output
        """
        if self.portfolio_returns is not None:
            # Portfolio returns
            self.portfolio_returns.to_csv(f"{output_path}_portfolio_returns.csv")
            self.portfolio_returns.to_frame('TSMOM_Returns').to_parquet(f"{output_path}_portfolio_returns.parquet")
            
        if self.contract_weights is not None:
            # Contract weights
            self.contract_weights.to_csv(f"{output_path}_contract_weights.csv")
            self.contract_weights.to_parquet(f"{output_path}_contract_weights.parquet")
            
        if self.turnover_data is not None:
            # Turnover data
            self.turnover_data.to_csv(f"{output_path}_turnover.csv")
        
        # Portfolio statistics
        stats = self.get_portfolio_statistics()
        import json
        with open(f"{output_path}_portfolio_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        self.logger.info(f"💾 Portfolio data esportati in {output_path}_*")
    
    def plot_portfolio_exposure(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualizza l'esposizione del portafoglio nel tempo.
        
        Args:
            figsize: Dimensione figura
        """
        if self.contract_weights is None:
            self.logger.error("Calcola prima i pesi contratti!")
            return
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Gross exposure over time
        ax1 = axes[0, 0]
        gross_exposure = self.contract_weights.abs().sum(axis=1)
        gross_exposure.plot(ax=ax1, color='blue')
        ax1.set_title('Gross Exposure Over Time')
        ax1.set_ylabel('Total |Weight|')
        ax1.grid(True, alpha=0.3)
        
        # 2. Net exposure over time  
        ax2 = axes[0, 1]
        net_exposure = self.contract_weights.sum(axis=1)
        net_exposure.plot(ax=ax2, color='green')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Net Exposure Over Time')
        ax2.set_ylabel('Net Weight')
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of active positions
        ax3 = axes[1, 0]
        active_positions = (self.contract_weights != 0).sum(axis=1)
        active_positions.plot(ax=ax3, color='orange')
        ax3.set_title('Active Positions Over Time')
        ax3.set_ylabel('Number of Active Contracts')
        ax3.grid(True, alpha=0.3)
        
        # 4. Weight distribution heatmap (sample)
        ax4 = axes[1, 1]
        sample_weights = self.contract_weights.iloc[-12:, :10]  # Last 12 months, first 10 tickers
        im = ax4.imshow(sample_weights.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        ax4.set_title('Weight Heatmap (Last 12M, Sample Tickers)')
        ax4.set_xlabel('Time (Months)')
        ax4.set_ylabel('Contracts')
        
        # Colorbar per heatmap
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.show()