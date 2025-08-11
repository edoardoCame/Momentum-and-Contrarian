"""
TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012) Implementation
Classe principale che integra tutti i moduli per l'esecuzione completa della strategia TSMOM.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
import logging
import os
from datetime import datetime

# Import dei moduli TSMOM
from modules.data_manager import TSMOMDataManager
from modules.returns_calculator import TSMOMReturnsCalculator
from modules.volatility_estimator import TSMOMVolatilityEstimator
from modules.signal_generator import TSMOMSignalGenerator
from modules.portfolio_constructor import TSMOMPortfolioConstructor
from modules.performance_analyzer import TSMOMPerformanceAnalyzer
from modules.visualizer import TSMOMVisualizer
from modules.validator import TSMOMValidator

warnings.filterwarnings('ignore')

class TSMOMStrategy:
    """
    Strategia TSMOM completa seguendo Moskowitz, Ooi & Pedersen (2012) - versione modificata.
    
    Integra tutti i moduli per l'esecuzione end-to-end:
    1. Download dati (futures + T-Bill)
    2. Calcolo returns e excess returns
    3. Stima volatilità EWMA ex-ante
    4. Generazione segnali momentum (12M, include ultimo mese)
    5. Costruzione portafoglio con volatility scaling (40% target)
    6. Analisi performance completa
    7. Validazione e diagnostiche
    8. Visualizzazioni professionali
    """
    
    def __init__(self,
                 start_date: str = "1985-01-01",
                 end_date: Optional[str] = None,
                 universe: Optional[Dict[str, List[str]]] = None,
                 target_volatility: float = 0.40,
                 lookback_months: int = 12,
                 transaction_cost_bps: float = 0,
                 data_cache_dir: str = "data/"):
        """
        Inizializza la strategia TSMOM.
        
        Args:
            start_date: Data inizio analisi
            end_date: Data fine (default: oggi)
            universe: Universo futures (default: universo completo MOP)
            target_volatility: Target volatility per contratto (default: 40%)
            lookback_months: Lookback period per momentum (default: 12M)
            transaction_cost_bps: Costi transazione in bps (default: 0)
            data_cache_dir: Directory per cache dei dati
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.universe = universe
        self.target_volatility = target_volatility
        self.lookback_months = lookback_months
        self.transaction_cost_bps = transaction_cost_bps
        self.data_cache_dir = data_cache_dir
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Inizializza moduli
        self._initialize_modules()
        
        # Storage per risultati
        self.results = {}
        self.is_executed = False
        
        self.logger.info(f"🚀 TSMOM Strategy inizializzata: {start_date} -> {self.end_date}")
    
    def _initialize_modules(self):
        """Inizializza tutti i moduli della strategia."""
        self.data_manager = TSMOMDataManager(
            start_date=self.start_date,
            end_date=self.end_date,
            universe=self.universe
        )
        
        self.returns_calculator = TSMOMReturnsCalculator()
        
        self.volatility_estimator = TSMOMVolatilityEstimator(
            center_of_mass=60,  # MOP (2012) specification
            annualization_factor=np.sqrt(261)
        )
        
        self.signal_generator = TSMOMSignalGenerator(
            lookback_months=self.lookback_months,
            holding_months=1  # MOP specification
        )
        
        self.portfolio_constructor = TSMOMPortfolioConstructor(
            target_volatility=self.target_volatility,
            max_weight_per_contract=10.0  # Safety cap
        )
        
        self.performance_analyzer = TSMOMPerformanceAnalyzer()
        
        self.visualizer = TSMOMVisualizer()
        
        self.validator = TSMOMValidator()
    
    def execute_full_strategy(self, validate_results: bool = True) -> Dict:
        """
        Esegue la strategia TSMOM completa end-to-end.
        
        Args:
            validate_results: Se True, esegue validazione completa
            
        Returns:
            Dict con tutti i risultati della strategia
        """
        self.logger.info("🎯 Avvio esecuzione completa strategia TSMOM...")
        
        try:
            # 1. Download e preparazione dati
            self._execute_data_preparation()
            
            # 2. Calcolo returns
            self._execute_returns_calculation()
            
            # 3. Stima volatilità
            self._execute_volatility_estimation()
            
            # 4. Generazione segnali
            self._execute_signal_generation()
            
            # 5. Costruzione portafoglio
            self._execute_portfolio_construction()
            
            # 6. Analisi performance
            self._execute_performance_analysis()
            
            # 7. Validazione (se richiesta)
            if validate_results:
                self._execute_validation()
            
            # 8. Genera visualizzazioni
            self._execute_visualization()
            
            # 9. Compila risultati finali
            self._compile_final_results()
            
            self.is_executed = True
            self.logger.info("✅ Strategia TSMOM eseguita con successo!")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Errore nell'esecuzione strategia: {e}")
            raise
    
    def _execute_data_preparation(self):
        """Fase 1: Download e preparazione dati."""
        self.logger.info("📊 Fase 1: Data Preparation...")
        
        # Carica o scarica dati usando la cache
        self.futures_data, self.tbill_data = self.data_manager.get_or_download_data(
            cache_dir=self.data_cache_dir
        )
        if not self.futures_data:
            raise ValueError("Nessun dato futures disponibile!")
        
        # Estrai matrice prezzi
        self.price_matrix = self.data_manager.get_futures_prices()
        
        # Risk-free rate mensile
        self.risk_free_monthly = self.data_manager.get_risk_free_rate_monthly()
        
        # Summary dei dati
        data_summary = self.data_manager.get_data_summary()
        self.results['data_summary'] = data_summary
        
        self.logger.info(f"✅ Dati preparati: {len(self.futures_data)} futures, {len(self.risk_free_monthly)} mesi RF")
    
    def _execute_returns_calculation(self):
        """Fase 2: Calcolo returns giornalieri e mensili."""
        self.logger.info("📊 Fase 2: Returns Calculation...")
        
        # Daily returns
        self.daily_returns = self.returns_calculator.calculate_daily_returns(self.price_matrix)
        
        # Monthly returns
        self.monthly_returns = self.returns_calculator.convert_to_monthly_returns(self.daily_returns)
        
        # Excess returns
        self.monthly_excess_returns = self.returns_calculator.calculate_excess_returns(
            self.monthly_returns, 
            self.risk_free_monthly
        )
        
        # Summary returns
        returns_summary = self.returns_calculator.get_returns_summary()
        self.results['returns_summary'] = returns_summary
        
        self.logger.info(f"✅ Returns calcolati: {self.monthly_excess_returns.shape} excess returns")
    
    def _execute_volatility_estimation(self):
        """Fase 3: Stima volatilità EWMA."""
        self.logger.info("📊 Fase 3: Volatility Estimation...")
        
        # Daily EWMA volatility
        self.daily_volatility = self.volatility_estimator.calculate_daily_ewma_volatility(self.daily_returns)
        
        # Monthly volatility
        self.monthly_volatility = self.volatility_estimator.extract_monthly_volatility(self.daily_volatility)
        
        # Lagged volatility per position sizing
        self.lagged_volatility = self.volatility_estimator.apply_lag_for_position_sizing(self.monthly_volatility)
        
        # Volatility statistics
        vol_stats = self.volatility_estimator.get_volatility_statistics()
        self.results['volatility_statistics'] = vol_stats
        
        self.logger.info(f"✅ Volatilità stimata: {self.lagged_volatility.shape} lagged vol per position sizing")
    
    def _execute_signal_generation(self):
        """Fase 4: Generazione segnali TSMOM."""
        self.logger.info("📊 Fase 4: Signal Generation...")
        
        # Cumulative momentum 12M
        self.momentum_cumulative = self.signal_generator.calculate_cumulative_momentum(self.monthly_excess_returns)
        
        # TSMOM signals
        self.tsmom_signals = self.signal_generator.generate_tsmom_signals(self.momentum_cumulative)
        
        # Look-ahead bias validation
        lookahead_validation = self.signal_generator.validate_look_ahead_bias_prevention(
            self.monthly_excess_returns, sample_months=5
        )
        
        self.results['signal_statistics'] = self.signal_generator.signal_statistics
        self.results['lookahead_validation'] = lookahead_validation
        
        self.logger.info(f"✅ Segnali generati: {self.tsmom_signals.shape} TSMOM signals")
    
    def _execute_portfolio_construction(self):
        """Fase 5: Costruzione portafoglio."""
        self.logger.info("📊 Fase 5: Portfolio Construction...")
        
        # Volatility-scaled weights
        self.contract_weights = self.portfolio_constructor.calculate_volatility_scaled_weights(
            self.tsmom_signals,
            self.lagged_volatility
        )
        
        # Portfolio returns
        self.portfolio_returns = self.portfolio_constructor.construct_portfolio_returns(
            self.contract_weights,
            self.monthly_excess_returns
        )
        
        # Turnover calculation
        self.turnover = self.portfolio_constructor.calculate_portfolio_turnover(self.contract_weights)
        
        # Apply transaction costs se specificati
        if self.transaction_cost_bps > 0:
            self.portfolio_returns_net = self.portfolio_constructor.apply_transaction_costs(
                self.portfolio_returns,
                self.turnover,
                self.transaction_cost_bps
            )
        else:
            self.portfolio_returns_net = self.portfolio_returns.copy()
        
        # Portfolio statistics
        portfolio_stats = self.portfolio_constructor.get_portfolio_statistics()
        self.results['portfolio_statistics'] = portfolio_stats
        
        self.logger.info(f"✅ Portafoglio costruito: {len(self.portfolio_returns)} mesi di returns")
    
    def _execute_performance_analysis(self):
        """Fase 6: Analisi performance."""
        self.logger.info("📊 Fase 6: Performance Analysis...")
        
        # Setup risk-free rate per l'analyzer
        self.performance_analyzer.risk_free_rate = self.risk_free_monthly
        
        # Performance metrics comprehensive
        self.performance_metrics = self.performance_analyzer.calculate_comprehensive_metrics(
            self.portfolio_returns_net
        )
        
        # Rolling metrics (36M windows)
        if len(self.portfolio_returns_net) >= 36:
            self.rolling_metrics = self.performance_analyzer.calculate_rolling_metrics(
                self.portfolio_returns_net, window_months=36
            )
        else:
            self.rolling_metrics = pd.DataFrame()
        
        # Performance report testuale
        performance_report = self.performance_analyzer.generate_performance_report()
        
        self.results['performance_metrics'] = self.performance_metrics
        self.results['rolling_metrics'] = self.rolling_metrics
        self.results['performance_report'] = performance_report
        
        self.logger.info("✅ Performance analysis completata")
    
    def _execute_validation(self):
        """Fase 7: Validazione e diagnostiche."""
        self.logger.info("📊 Fase 7: Validation & Diagnostics...")
        
        # Validazione completa
        self.validation_results = self.validator.run_comprehensive_validation(
            self.monthly_excess_returns,
            self.tsmom_signals,
            self.lagged_volatility,
            self.contract_weights,
            self.portfolio_returns_net
        )
        
        # Validation report
        validation_report = self.validator.generate_validation_report()
        
        self.results['validation_results'] = self.validation_results
        self.results['validation_report'] = validation_report
        
        # Check se validazione è passata
        validation_passed = self.validation_results.get('summary', {}).get('validation_passed', False)
        
        if validation_passed:
            self.logger.info("✅ Validazione completata con successo")
        else:
            error_count = self.validation_results.get('summary', {}).get('total_errors', 0)
            self.logger.warning(f"⚠️ Validazione completata con {error_count} errori")
    
    def _execute_visualization(self):
        """Fase 8: Generazione visualizzazioni."""
        self.logger.info("📊 Fase 8: Visualization Generation...")
        
        # Le visualizzazioni vengono create on-demand tramite i metodi plot_*()
        # Qui prepariamo solo i dati necessari
        self.visualization_data = {
            'portfolio_returns': self.portfolio_returns_net,
            'contract_weights': self.contract_weights,
            'signals': self.tsmom_signals,
            'rolling_metrics': self.rolling_metrics if hasattr(self, 'rolling_metrics') else None,
            'momentum_data': self.momentum_cumulative,
            'risk_free_rate': self.risk_free_monthly
        }
        
        self.logger.info("✅ Dati visualization preparati")
    
    def _compile_final_results(self):
        """Fase 9: Compilazione risultati finali."""
        self.logger.info("📊 Fase 9: Final Results Compilation...")
        
        # Summary esecutivo
        executive_summary = self._generate_executive_summary()
        
        # Key data per export
        key_data = {
            'portfolio_returns': self.portfolio_returns_net,
            'contract_weights': self.contract_weights,
            'tsmom_signals': self.tsmom_signals,
            'monthly_excess_returns': self.monthly_excess_returns,
            'lagged_volatility': self.lagged_volatility,
            'turnover': self.turnover
        }
        
        self.results.update({
            'executive_summary': executive_summary,
            'key_data': key_data,
            'strategy_parameters': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'target_volatility': self.target_volatility,
                'lookback_months': self.lookback_months,
                'transaction_cost_bps': self.transaction_cost_bps,
                'universe_size': len(self.futures_data)
            }
        })
        
        self.logger.info("✅ Risultati finali compilati")
    
    def _generate_executive_summary(self) -> Dict:
        """Genera summary esecutivo della strategia."""
        if not hasattr(self, 'performance_metrics'):
            return {"error": "Performance metrics non disponibili"}
        
        perf = self.performance_metrics
        basic = perf.get('basic_stats', {})
        returns = perf.get('return_metrics', {})
        risk = perf.get('risk_metrics', {})
        drawdown = perf.get('drawdown_metrics', {})
        
        return {
            "strategy_name": "TSMOM - Time Series Momentum",
            "paper_reference": "Moskowitz, Ooi & Pedersen (2012)",
            "execution_period": {
                "start": basic.get('start_date'),
                "end": basic.get('end_date'),
                "total_months": basic.get('observations', 0)
            },
            "key_performance": {
                "cagr": returns.get('cagr', 0),
                "annual_volatility": returns.get('volatility_annual', 0),
                "sharpe_ratio": risk.get('sharpe_ratio', 0),
                "max_drawdown": drawdown.get('max_drawdown', 0),
                "calmar_ratio": drawdown.get('calmar_ratio', 0)
            },
            "portfolio_characteristics": {
                "target_volatility_per_contract": self.target_volatility,
                "average_active_positions": self.results.get('portfolio_statistics', {}).get('weights', {}).get('avg_active_contracts', 0),
                "average_turnover": self.results.get('portfolio_statistics', {}).get('turnover', {}).get('mean_monthly', 0)
            }
        }
    
    # Metodi di accesso ai risultati e visualizzazioni
    
    def get_equity_curve(self) -> pd.Series:
        """Restituisce l'equity curve del portafoglio."""
        if not self.is_executed:
            raise ValueError("Esegui prima la strategia con execute_full_strategy()")
        return (1 + self.portfolio_returns_net).cumprod()
    
    def get_performance_summary(self) -> str:
        """Restituisce summary testuale delle performance."""
        if not self.is_executed:
            raise ValueError("Esegui prima la strategia con execute_full_strategy()")
        return self.results.get('performance_report', 'Performance report non disponibile')
    
    def plot_equity_curves(self, **kwargs):
        """Plotta equity curves."""
        if not self.is_executed:
            raise ValueError("Esegui prima la strategia con execute_full_strategy()")
        
        return self.visualizer.plot_equity_curves(
            self.portfolio_returns_net,
            risk_free_rate=self.risk_free_monthly,
            **kwargs
        )
    
    def plot_drawdown_analysis(self, **kwargs):
        """Plotta analisi drawdown."""
        if not self.is_executed:
            raise ValueError("Esegui prima la strategia con execute_full_strategy()")
        
        return self.visualizer.plot_drawdown_analysis(self.portfolio_returns_net, **kwargs)
    
    def plot_rolling_metrics(self, **kwargs):
        """Plotta metriche rolling."""
        if not self.is_executed or self.rolling_metrics.empty:
            raise ValueError("Esegui prima la strategia o dati insufficienti per rolling metrics")
        
        return self.visualizer.plot_rolling_metrics(self.rolling_metrics, **kwargs)
    
    def plot_commodity_heatmap(self, **kwargs):
        """Plotta heatmap commodities."""
        if not self.is_executed:
            raise ValueError("Esegui prima la strategia con execute_full_strategy()")
        
        return self.visualizer.plot_commodity_heatmap(
            self.contract_weights,
            self.portfolio_returns_net,
            **kwargs
        )
    
    def plot_signal_analysis(self, **kwargs):
        """Plotta analisi segnali."""
        if not self.is_executed:
            raise ValueError("Esegui prima la strategia con execute_full_strategy()")
        
        return self.visualizer.plot_signal_analysis(
            self.tsmom_signals,
            self.momentum_cumulative,
            **kwargs
        )
    
    def save_all_results(self, output_dir: str = "results/tsmom_output"):
        """
        Salva tutti i risultati, grafici e report.
        
        Args:
            output_dir: Directory di output
        """
        if not self.is_executed:
            raise ValueError("Esegui prima la strategia con execute_full_strategy()")
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"💾 Salvataggio risultati completi in {output_dir}...")
        
        # 1. Salva dati chiave
        self._save_key_data(output_dir)
        
        # 2. Salva reports
        self._save_reports(output_dir)
        
        # 3. Salva grafici
        self._save_visualizations(output_dir)
        
        self.logger.info("✅ Tutti i risultati salvati!")
    
    def _save_key_data(self, output_dir: str):
        """Salva i dati chiave della strategia."""
        # Portfolio returns
        self.portfolio_returns_net.to_csv(f"{output_dir}/portfolio_returns.csv")
        self.portfolio_returns_net.to_frame('TSMOM_Returns').to_parquet(f"{output_dir}/portfolio_returns.parquet")
        
        # Contract weights
        self.contract_weights.to_csv(f"{output_dir}/contract_weights.csv")
        self.contract_weights.to_parquet(f"{output_dir}/contract_weights.parquet")
        
        # Signals
        self.tsmom_signals.to_csv(f"{output_dir}/tsmom_signals.csv")
        
        # Excess returns
        self.monthly_excess_returns.to_csv(f"{output_dir}/monthly_excess_returns.csv")
        
        # Volatility
        self.lagged_volatility.to_csv(f"{output_dir}/volatility_lagged.csv")
    
    def _save_reports(self, output_dir: str):
        """Salva i reports testuali."""
        import json
        
        # Performance report
        with open(f"{output_dir}/performance_report.txt", 'w') as f:
            f.write(self.get_performance_summary())
        
        # Executive summary
        with open(f"{output_dir}/executive_summary.json", 'w') as f:
            json.dump(self.results['executive_summary'], f, indent=2, default=str)
        
        # Validation report se disponibile
        if 'validation_report' in self.results:
            with open(f"{output_dir}/validation_report.txt", 'w') as f:
                f.write(self.results['validation_report'])
        
        # Performance metrics completi
        with open(f"{output_dir}/performance_metrics.json", 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
    
    def _save_visualizations(self, output_dir: str):
        """Salva tutte le visualizzazioni."""
        viz_dir = f"{output_dir}/visualizations"
        
        self.visualizer.save_all_plots(
            portfolio_returns=self.portfolio_returns_net,
            contract_weights=self.contract_weights,
            signals=self.tsmom_signals,
            output_dir=viz_dir,
            rolling_metrics=self.rolling_metrics if hasattr(self, 'rolling_metrics') else None,
            momentum_data=self.momentum_cumulative,
            risk_free_rate=self.risk_free_monthly
        )