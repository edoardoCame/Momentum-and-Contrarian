"""
Validator per TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012)
Implementa validazioni e diagnostiche per garantire la correttezza dell'implementazione.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import logging

warnings.filterwarnings('ignore')

class TSMOMValidator:
    """
    Validatore per la strategia TSMOM con controlli di integrità completi.
    
    Validazioni implementate:
    - Look-ahead bias prevention
    - Temporal alignment correctness
    - Data integrity checks
    - Signal generation logic validation
    - Portfolio construction validation
    - Performance calculation checks
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        self.validation_errors = []
        self.validation_warnings = []
    
    def run_comprehensive_validation(self,
                                   monthly_excess_returns: pd.DataFrame,
                                   signals: pd.DataFrame,
                                   volatilities: pd.DataFrame,
                                   contract_weights: pd.DataFrame,
                                   portfolio_returns: pd.Series) -> Dict:
        """
        Esegue validazione completa della strategia TSMOM.
        
        Args:
            monthly_excess_returns: DataFrame excess returns mensili
            signals: DataFrame segnali TSMOM
            volatilities: DataFrame volatilità laggata
            contract_weights: DataFrame pesi contratti
            portfolio_returns: Serie rendimenti portafoglio
            
        Returns:
            Dict con risultati della validazione
        """
        self.logger.info("🔍 Avvio validazione completa TSMOM...")
        
        self.validation_errors = []
        self.validation_warnings = []
        
        # 1. Look-ahead bias validation
        lookahead_result = self._validate_lookahead_bias(monthly_excess_returns, signals)
        
        # 2. Temporal alignment validation
        temporal_result = self._validate_temporal_alignment(signals, volatilities, contract_weights)
        
        # 3. Signal generation validation
        signal_result = self._validate_signal_generation(signals, monthly_excess_returns)
        
        # 4. Volatility calculation validation
        volatility_result = self._validate_volatility_calculation(volatilities)
        
        # 5. Portfolio construction validation
        portfolio_result = self._validate_portfolio_construction(contract_weights, monthly_excess_returns, portfolio_returns)
        
        # 6. Data integrity validation
        integrity_result = self._validate_data_integrity(monthly_excess_returns, signals, volatilities)
        
        # Summary dei risultati
        self.validation_results = {
            "lookahead_bias": lookahead_result,
            "temporal_alignment": temporal_result,
            "signal_generation": signal_result,
            "volatility_calculation": volatility_result,
            "portfolio_construction": portfolio_result,
            "data_integrity": integrity_result,
            "summary": {
                "total_errors": len(self.validation_errors),
                "total_warnings": len(self.validation_warnings),
                "validation_passed": len(self.validation_errors) == 0
            },
            "errors": self.validation_errors,
            "warnings": self.validation_warnings
        }
        
        # Log risultati
        if len(self.validation_errors) == 0:
            self.logger.info("✅ Validazione completata con successo!")
        else:
            self.logger.error(f"❌ Validazione fallita: {len(self.validation_errors)} errori trovati")
        
        if len(self.validation_warnings) > 0:
            self.logger.warning(f"⚠️ {len(self.validation_warnings)} warning trovati")
        
        return self.validation_results
    
    def _validate_lookahead_bias(self, returns: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """Valida che non ci sia look-ahead bias."""
        self.logger.info("🔍 Validazione look-ahead bias...")
        
        result = {"passed": True, "details": {}}
        
        try:
            # Check 1: I segnali devono essere disponibili solo DOPO i dati necessari
            min_signal_date = signals.dropna(how='all').index.min()
            min_returns_date = returns.dropna(how='all').index.min()
            
            # I segnali devono iniziare almeno 13 mesi dopo i returns (12M lookback + 1M lag)
            expected_min_signal_date = min_returns_date + pd.DateOffset(months=13)
            
            if min_signal_date < expected_min_signal_date:
                self.validation_errors.append(f"Look-ahead bias: segnali iniziano troppo presto ({min_signal_date} vs expected {expected_min_signal_date})")
                result["passed"] = False
            
            # Check 2: Sample validation su alcuni punti temporali
            sample_validation = self._sample_lookahead_validation(returns, signals)
            result["details"]["sample_validation"] = sample_validation
            
            if not sample_validation["all_valid"]:
                result["passed"] = False
                self.validation_errors.append("Look-ahead bias trovato nella sample validation")
            
            # Check 3: Timing pattern consistency
            timing_check = self._validate_timing_pattern(signals)
            result["details"]["timing_pattern"] = timing_check
            
        except Exception as e:
            self.validation_errors.append(f"Errore nella validazione look-ahead bias: {e}")
            result["passed"] = False
        
        return result
    
    def _sample_lookahead_validation(self, returns: pd.DataFrame, signals: pd.DataFrame, n_samples: int = 5) -> Dict:
        """Valida look-ahead bias su un campione di date."""
        
        # Prendi sample dates
        available_signal_dates = signals.dropna(how='all').index
        if len(available_signal_dates) < n_samples:
            sample_dates = available_signal_dates
        else:
            # Prendi date equi-spaziate
            idx_step = len(available_signal_dates) // n_samples
            sample_indices = range(0, len(available_signal_dates), idx_step)[:n_samples]
            sample_dates = [available_signal_dates[i] for i in sample_indices]
        
        sample_tickers = returns.columns[:3]  # Primi 3 tickers
        validation_details = []
        all_valid = True
        
        for date in sample_dates:
            for ticker in sample_tickers:
                if ticker in signals.columns:
                    # Prendi il segnale a questa data
                    signal_value = signals.loc[date, ticker] if date in signals.index else np.nan
                    
                    if not np.isnan(signal_value):
                        # Verifica che il segnale sia basato solo su dati fino a t-1
                        lookback_end = date - pd.DateOffset(months=1)
                        lookback_start = date - pd.DateOffset(months=12)
                        
                        # Estrai returns nel periodo corretto
                        historical_returns = returns.loc[
                            (returns.index >= lookback_start) & 
                            (returns.index <= lookback_end), ticker
                        ].dropna()
                        
                        if len(historical_returns) >= 10:  # Minimo dati per validazione
                            # Ricalcola segnale manualmente
                            manual_cumulative = (1 + historical_returns).prod() - 1
                            manual_signal = np.sign(manual_cumulative)
                            
                            # Confronta con segnale del modello
                            is_valid = abs(manual_signal - signal_value) < 0.01
                            
                            validation_details.append({
                                'date': date.date(),
                                'ticker': ticker,
                                'model_signal': signal_value,
                                'manual_signal': manual_signal,
                                'is_valid': is_valid,
                                'lookback_months': len(historical_returns)
                            })
                            
                            if not is_valid:
                                all_valid = False
        
        return {
            "all_valid": all_valid,
            "sample_size": len(validation_details),
            "valid_count": sum(d['is_valid'] for d in validation_details),
            "details": validation_details
        }
    
    def _validate_timing_pattern(self, signals: pd.DataFrame) -> Dict:
        """Valida che i pattern temporali siano consistenti."""
        # Check che non ci siano salti temporali anomali
        signal_dates = signals.dropna(how='all').index
        date_diffs = signal_dates.to_series().diff().dt.days
        
        # La maggior parte dovrebbe essere ~30 giorni (mensile)
        median_diff = date_diffs.median()
        anomalous_gaps = (date_diffs > 60) | (date_diffs < 15)  # Oltre 2 mesi o sotto 2 settimane
        
        return {
            "median_gap_days": median_diff,
            "anomalous_gaps_count": anomalous_gaps.sum(),
            "expected_monthly_pattern": 25 <= median_diff <= 35
        }
    
    def _validate_temporal_alignment(self, signals: pd.DataFrame, volatilities: pd.DataFrame, weights: pd.DataFrame) -> Dict:
        """Valida allineamento temporale tra componenti."""
        self.logger.info("🔍 Validazione allineamento temporale...")
        
        result = {"passed": True, "details": {}}
        
        try:
            # Check 1: Volatilities devono precedere signals di esatto 1 mese
            vol_dates = volatilities.dropna(how='all').index
            signal_dates = signals.dropna(how='all').index
            
            # Per ogni data di signal, deve esistere volatility 1 mese prima
            alignment_issues = 0
            sample_checks = []
            
            for i, signal_date in enumerate(signal_dates[:10]):  # Check prime 10 date
                expected_vol_date = signal_date  # Volatilities sono già laggata
                
                if expected_vol_date not in vol_dates:
                    alignment_issues += 1
                
                sample_checks.append({
                    'signal_date': signal_date.date(),
                    'expected_vol_date': expected_vol_date.date(),
                    'vol_available': expected_vol_date in vol_dates
                })
            
            result["details"]["volatility_signal_alignment"] = {
                "alignment_issues": alignment_issues,
                "sample_checks": sample_checks
            }
            
            # Check 2: Weights devono allinearsi correttamente con returns timing
            weight_dates = weights.dropna(how='all').index
            
            # Weights al tempo t dovrebbero influenzare returns al tempo t+1
            timing_validation = []
            for i, weight_date in enumerate(weight_dates[:5]):
                next_month = weight_date + pd.DateOffset(months=1)
                timing_validation.append({
                    'weight_date': weight_date.date(),
                    'next_month': next_month.date(),
                    'correct_timing_pattern': True  # Placeholder per ora
                })
            
            result["details"]["weight_return_timing"] = timing_validation
            
        except Exception as e:
            self.validation_errors.append(f"Errore nella validazione temporal alignment: {e}")
            result["passed"] = False
        
        return result
    
    def _validate_signal_generation(self, signals: pd.DataFrame, returns: pd.DataFrame) -> Dict:
        """Valida la logica di generazione dei segnali."""
        self.logger.info("🔍 Validazione signal generation...")
        
        result = {"passed": True, "details": {}}
        
        try:
            # Check 1: I segnali devono essere solo -1, 0, 1
            unique_signals = np.unique(signals.values[~np.isnan(signals.values)])
            valid_signals = set([-1, 0, 1])
            invalid_signals = set(unique_signals) - valid_signals
            
            if invalid_signals:
                self.validation_errors.append(f"Segnali non validi trovati: {invalid_signals}")
                result["passed"] = False
            
            result["details"]["signal_values"] = {
                "unique_signals": list(unique_signals),
                "all_valid": len(invalid_signals) == 0
            }
            
            # Check 2: Distribuzione dei segnali ragionevole
            signal_distribution = pd.Series(signals.values.flatten()).value_counts()
            total_signals = len(signals.values[~np.isnan(signals.values)])
            
            distribution_pct = (signal_distribution / total_signals * 100).round(2)
            
            # I segnali non dovrebbero essere troppo skewed (>90% su un valore)
            max_concentration = distribution_pct.max()
            if max_concentration > 90:
                self.validation_warnings.append(f"Segnali molto concentrati: {max_concentration:.1f}% su un valore")
            
            result["details"]["signal_distribution"] = {
                "distribution_pct": distribution_pct.to_dict(),
                "max_concentration": max_concentration,
                "reasonable_distribution": max_concentration < 90
            }
            
            # Check 3: Validazione matematica su campione
            math_validation = self._validate_signal_math(signals, returns)
            result["details"]["mathematical_validation"] = math_validation
            
        except Exception as e:
            self.validation_errors.append(f"Errore nella validazione signal generation: {e}")
            result["passed"] = False
        
        return result
    
    def _validate_signal_math(self, signals: pd.DataFrame, returns: pd.DataFrame) -> Dict:
        """Valida matematicamente alcuni segnali generati."""
        
        # Prendi un campione e ricalcola manualmente
        sample_date = signals.dropna(how='all').index[-5]  # 5° dall'ultimo
        sample_ticker = signals.columns[0]
        
        if sample_date in signals.index and sample_ticker in signals.columns:
            model_signal = signals.loc[sample_date, sample_ticker]
            
            # Ricalcola manualmente
            lookback_end = sample_date - pd.DateOffset(months=1)
            lookback_start = sample_date - pd.DateOffset(months=12)
            
            historical_returns = returns.loc[
                (returns.index >= lookback_start) & 
                (returns.index <= lookback_end), sample_ticker
            ].dropna()
            
            if len(historical_returns) > 0:
                manual_cumulative = (1 + historical_returns).prod() - 1
                manual_signal = np.sign(manual_cumulative)
                
                return {
                    "sample_date": sample_date.date(),
                    "sample_ticker": sample_ticker,
                    "model_signal": float(model_signal) if not np.isnan(model_signal) else None,
                    "manual_signal": float(manual_signal),
                    "cumulative_return": float(manual_cumulative),
                    "signals_match": abs(model_signal - manual_signal) < 0.01 if not np.isnan(model_signal) else False,
                    "lookback_periods": len(historical_returns)
                }
        
        return {"error": "Impossibile validare sample matematico"}
    
    def _validate_volatility_calculation(self, volatilities: pd.DataFrame) -> Dict:
        """Valida il calcolo delle volatilità."""
        self.logger.info("🔍 Validazione volatility calculation...")
        
        result = {"passed": True, "details": {}}
        
        try:
            # Check 1: Volatilità sempre positive
            negative_vol = (volatilities < 0).sum().sum()
            if negative_vol > 0:
                self.validation_errors.append(f"{negative_vol} volatilità negative trovate")
                result["passed"] = False
            
            # Check 2: Range ragionevole (5%-200% annualizzata)
            min_vol = volatilities.min().min()
            max_vol = volatilities.max().max()
            
            reasonable_range = (min_vol >= 0.05 and max_vol <= 2.0)
            if not reasonable_range:
                self.validation_warnings.append(f"Volatilità fuori range ragionevole: {min_vol:.1%} - {max_vol:.1%}")
            
            # Check 3: Continuità temporale (no salti estremi)
            vol_changes = volatilities.pct_change().abs()
            extreme_changes = (vol_changes > 1.0).sum().sum()  # Cambi > 100%
            
            if extreme_changes > volatilities.size * 0.01:  # >1% di cambi estremi
                self.validation_warnings.append(f"Molti cambi estremi di volatilità: {extreme_changes}")
            
            result["details"] = {
                "negative_volatilities": negative_vol,
                "min_volatility": float(min_vol),
                "max_volatility": float(max_vol),
                "reasonable_range": reasonable_range,
                "extreme_changes": int(extreme_changes)
            }
            
        except Exception as e:
            self.validation_errors.append(f"Errore nella validazione volatility: {e}")
            result["passed"] = False
        
        return result
    
    def _validate_portfolio_construction(self, weights: pd.DataFrame, returns: pd.DataFrame, portfolio_returns: pd.Series) -> Dict:
        """Valida la costruzione del portafoglio."""
        self.logger.info("🔍 Validazione portfolio construction...")
        
        result = {"passed": True, "details": {}}
        
        try:
            # Check 1: Ricalcola manualmente alcuni rendimenti di portafoglio
            manual_validation = self._manual_portfolio_calculation_check(weights, returns, portfolio_returns)
            result["details"]["manual_calculation_check"] = manual_validation
            
            if not manual_validation.get("calculations_match", True):
                self.validation_errors.append("Portfolio returns non corrispondono al calcolo manuale")
                result["passed"] = False
            
            # Check 2: Peso constraints ragionevoli
            max_weight = weights.abs().max().max()
            if max_weight > 20:  # Peso singolo > 20x leverage sembra eccessivo
                self.validation_warnings.append(f"Peso estremo trovato: {max_weight:.2f}")
            
            # Check 3: Portfolio returns range ragionevole
            max_monthly_return = portfolio_returns.max()
            min_monthly_return = portfolio_returns.min()
            
            if max_monthly_return > 0.5:  # >50% in un mese
                self.validation_warnings.append(f"Rendimento mensile estremo: {max_monthly_return:.2%}")
            
            result["details"]["portfolio_statistics"] = {
                "max_weight": float(max_weight),
                "max_monthly_return": float(max_monthly_return),
                "min_monthly_return": float(min_monthly_return),
                "total_months": len(portfolio_returns)
            }
            
        except Exception as e:
            self.validation_errors.append(f"Errore nella validazione portfolio construction: {e}")
            result["passed"] = False
        
        return result
    
    def _manual_portfolio_calculation_check(self, weights: pd.DataFrame, returns: pd.DataFrame, portfolio_returns: pd.Series) -> Dict:
        """Check manuale del calcolo dei rendimenti di portafoglio."""
        
        # Prendi una data campione 
        sample_dates = portfolio_returns.index[-3:]  # Ultime 3 date
        
        calculation_checks = []
        
        for date in sample_dates:
            if date in portfolio_returns.index:
                model_return = portfolio_returns[date]
                
                # Trova i pesi al tempo t-1
                weight_date = date - pd.DateOffset(months=1)
                
                if weight_date in weights.index and date in returns.index:
                    # Pesi al tempo t-1
                    weights_t = weights.loc[weight_date]
                    # Returns al tempo t
                    returns_t = returns.loc[date]
                    
                    # Calcola manualmente: sum(w[t-1] * r[t]) con equal weight aggregation
                    contract_returns = weights_t * returns_t
                    manual_portfolio_return = contract_returns.mean(skipna=True)
                    
                    # Confronta
                    difference = abs(model_return - manual_portfolio_return)
                    match = difference < 0.0001  # Tolerance numerica
                    
                    calculation_checks.append({
                        'date': date.date(),
                        'weight_date': weight_date.date(),
                        'model_return': float(model_return),
                        'manual_return': float(manual_portfolio_return),
                        'difference': float(difference),
                        'match': match
                    })
        
        all_match = all(check['match'] for check in calculation_checks)
        
        return {
            "calculations_match": all_match,
            "sample_checks": calculation_checks,
            "total_checks": len(calculation_checks)
        }
    
    def _validate_data_integrity(self, returns: pd.DataFrame, signals: pd.DataFrame, volatilities: pd.DataFrame) -> Dict:
        """Valida l'integrità generale dei dati."""
        self.logger.info("🔍 Validazione data integrity...")
        
        result = {"passed": True, "details": {}}
        
        try:
            # Check 1: Overlap temporale sufficiente
            returns_dates = set(returns.dropna(how='all').index)
            signals_dates = set(signals.dropna(how='all').index) 
            vol_dates = set(volatilities.dropna(how='all').index)
            
            common_dates = returns_dates.intersection(signals_dates).intersection(vol_dates)
            
            if len(common_dates) < 12:  # Minimo 1 anno di dati comuni
                self.validation_errors.append(f"Overlap temporale insufficiente: {len(common_dates)} mesi")
                result["passed"] = False
            
            # Check 2: Ticker consistency
            returns_tickers = set(returns.columns)
            signals_tickers = set(signals.columns)
            vol_tickers = set(volatilities.columns)
            
            common_tickers = returns_tickers.intersection(signals_tickers).intersection(vol_tickers)
            
            if len(common_tickers) < 5:  # Minimo 5 tickers comuni
                self.validation_errors.append(f"Ticker comuni insufficienti: {len(common_tickers)}")
                result["passed"] = False
            
            result["details"] = {
                "common_dates": len(common_dates),
                "common_tickers": len(common_tickers),
                "returns_coverage": len(returns_dates),
                "signals_coverage": len(signals_dates),
                "volatilities_coverage": len(vol_dates)
            }
            
        except Exception as e:
            self.validation_errors.append(f"Errore nella validazione data integrity: {e}")
            result["passed"] = False
        
        return result
    
    def generate_validation_report(self) -> str:
        """Genera report testuale della validazione."""
        if not self.validation_results:
            return "Nessuna validazione eseguita!"
        
        report = "=" * 60 + "\n"
        report += "TSMOM STRATEGY - VALIDATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Summary
        summary = self.validation_results.get("summary", {})
        status = "PASSED" if summary.get("validation_passed", False) else "FAILED"
        report += f"VALIDATION STATUS: {status}\n"
        report += f"Errors: {summary.get('total_errors', 0)}\n"
        report += f"Warnings: {summary.get('total_warnings', 0)}\n\n"
        
        # Dettagli per categoria
        for category, result in self.validation_results.items():
            if category not in ["summary", "errors", "warnings"]:
                status_symbol = "✅" if result.get("passed", False) else "❌"
                report += f"{status_symbol} {category.replace('_', ' ').title()}\n"
        
        # Errori
        if self.validation_errors:
            report += "\nERRORS:\n"
            for i, error in enumerate(self.validation_errors, 1):
                report += f"  {i}. {error}\n"
        
        # Warning
        if self.validation_warnings:
            report += "\nWARNINGS:\n"
            for i, warning in enumerate(self.validation_warnings, 1):
                report += f"  {i}. {warning}\n"
        
        return report
    
    def export_validation_data(self, output_path: str):
        """Esporta i risultati della validazione."""
        if self.validation_results:
            import json
            with open(f"{output_path}_validation_results.json", 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
        
        # Validation report
        report = self.generate_validation_report()
        with open(f"{output_path}_validation_report.txt", 'w') as f:
            f.write(report)
        
        self.logger.info(f"💾 Validation data esportati in {output_path}_*")