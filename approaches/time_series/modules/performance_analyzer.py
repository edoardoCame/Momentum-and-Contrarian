"""
Performance Analyzer per TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012)
Calcola metriche di performance comprehensive per l'analisi della strategia.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings
from scipy import stats
import logging

warnings.filterwarnings('ignore')

class TSMOMPerformanceAnalyzer:
    """
    Analizzatore di performance per TSMOM seguendo le best practices quantitative.
    
    Metriche implementate:
    - Return metrics: CAGR, volatility, Sharpe ratio
    - Risk metrics: Maximum drawdown, Calmar ratio, VaR, CVaR  
    - Distribution metrics: Skewness, kurtosis, hit ratio
    - Advanced metrics: Information ratio, Sortino ratio
    """
    
    def __init__(self, risk_free_rate: Optional[pd.Series] = None):
        """
        Inizializza il performance analyzer.
        
        Args:
            risk_free_rate: Serie del risk-free rate mensile (opzionale)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        
        # Storage per performance metrics
        self.performance_metrics = {}
        self.rolling_metrics = {}
    
    def calculate_comprehensive_metrics(self, 
                                      portfolio_returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calcola set completo di metriche di performance.
        
        Args:
            portfolio_returns: Serie rendimenti mensili del portafoglio
            benchmark_returns: Serie rendimenti benchmark (opzionale)
            
        Returns:
            Dict con tutte le metriche calcolate
        """
        self.logger.info("📊 Calcolo comprehensive performance metrics...")
        
        if portfolio_returns.empty:
            raise ValueError("Portfolio returns vuoto!")
        
        # Clean data
        returns = portfolio_returns.dropna()
        
        # Calcola tutte le metriche
        metrics = {
            "basic_stats": self._calculate_basic_statistics(returns),
            "return_metrics": self._calculate_return_metrics(returns),
            "risk_metrics": self._calculate_risk_metrics(returns),
            "distribution_metrics": self._calculate_distribution_metrics(returns),
            "drawdown_metrics": self._calculate_drawdown_metrics(returns)
        }
        
        # Benchmark comparison se disponibile
        if benchmark_returns is not None:
            metrics["relative_metrics"] = self._calculate_relative_metrics(returns, benchmark_returns)
        
        # Risk-adjusted metrics
        if self.risk_free_rate is not None:
            metrics["risk_adjusted"] = self._calculate_risk_adjusted_metrics(returns)
        
        self.performance_metrics = metrics
        
        self.logger.info("✅ Performance metrics calcolate")
        return metrics
    
    def _calculate_basic_statistics(self, returns: pd.Series) -> Dict:
        """Calcola statistiche di base."""
        return {
            "observations": len(returns),
            "start_date": returns.index.min().date(),
            "end_date": returns.index.max().date(),
            "mean_monthly": returns.mean(),
            "median_monthly": returns.median(),
            "std_monthly": returns.std(),
            "min_return": returns.min(),
            "max_return": returns.max()
        }
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """Calcola metriche di rendimento annualizzate."""
        # Annualization
        mean_annual = returns.mean() * 12
        std_annual = returns.std() * np.sqrt(12)
        
        # CAGR (Compound Annual Growth Rate)
        total_periods = len(returns)
        years = total_periods / 12
        cumulative_return = (1 + returns).prod()
        cagr = (cumulative_return ** (1/years)) - 1 if years > 0 else 0
        
        return {
            "mean_annual": mean_annual,
            "volatility_annual": std_annual,
            "cagr": cagr,
            "total_return": cumulative_return - 1,
            "best_month": returns.max(),
            "worst_month": returns.min(),
            "positive_months": (returns > 0).sum(),
            "negative_months": (returns < 0).sum(),
            "hit_ratio": (returns > 0).mean()
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calcola metriche di rischio."""
        # Sharpe ratio (usando excess returns)
        if self.risk_free_rate is not None:
            # Allinea temporalmente
            common_dates = returns.index.intersection(self.risk_free_rate.index)
            if len(common_dates) > 0:
                aligned_returns = returns.loc[common_dates]
                aligned_rf = self.risk_free_rate.loc[common_dates]
                excess_returns = aligned_returns - aligned_rf
                sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(12) if excess_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            # Assume risk-free = 0 se non disponibile
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(12) if returns.std() > 0 else 0
        
        # VaR e CVaR
        var_95 = returns.quantile(0.05)  # 5% VaR
        var_99 = returns.quantile(0.01)  # 1% VaR
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else returns.std()
        sortino_ratio = (returns.mean() / downside_std) * np.sqrt(12) if downside_std > 0 else 0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "downside_deviation": downside_std
        }
    
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict:
        """Calcola metriche distribuzione."""
        return {
            "skewness": stats.skew(returns.dropna()),
            "kurtosis": stats.kurtosis(returns.dropna()),
            "jarque_bera_stat": stats.jarque_bera(returns.dropna())[0],
            "jarque_bera_pvalue": stats.jarque_bera(returns.dropna())[1],
            "normality_test": stats.jarque_bera(returns.dropna())[1] > 0.05  # True if normal at 5%
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calcola metriche di drawdown."""
        # Equity curve
        equity_curve = (1 + returns).cumprod()
        
        # Running maximum (peak)
        running_max = equity_curve.expanding().max()
        
        # Drawdown series
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        # Find quando siamo in drawdown
        in_drawdown = drawdown < 0
        
        # Trova periodi di drawdown consecutivi
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        # Aggiungi ultimo periodo se non chiuso
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Calmar ratio = CAGR / |Max Drawdown|
        cagr = ((1 + returns).prod() ** (12 / len(returns))) - 1 if len(returns) > 0 else 0
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "avg_drawdown_duration": avg_drawdown_duration,
            "max_drawdown_duration": max_drawdown_duration,
            "current_drawdown": drawdown.iloc[-1] if len(drawdown) > 0 else 0,
            "drawdown_periods_count": len(drawdown_periods)
        }
    
    def _calculate_relative_metrics(self, returns: pd.Series, benchmark: pd.Series) -> Dict:
        """Calcola metriche relative vs benchmark."""
        # Allinea temporalmente
        common_dates = returns.index.intersection(benchmark.index)
        if len(common_dates) == 0:
            return {"error": "Nessun allineamento temporale con benchmark"}
        
        aligned_returns = returns.loc[common_dates]
        aligned_benchmark = benchmark.loc[common_dates]
        
        # Active returns (excess vs benchmark)
        active_returns = aligned_returns - aligned_benchmark
        
        # Information ratio
        information_ratio = (active_returns.mean() / active_returns.std()) * np.sqrt(12) if active_returns.std() > 0 else 0
        
        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(12)
        
        # Beta vs benchmark
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        return {
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "beta": beta,
            "correlation": aligned_returns.corr(aligned_benchmark),
            "active_return_annual": active_returns.mean() * 12,
            "outperformance_months": (active_returns > 0).sum(),
            "outperformance_ratio": (active_returns > 0).mean()
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Calcola metriche risk-adjusted usando risk-free rate."""
        # Allinea con risk-free rate
        common_dates = returns.index.intersection(self.risk_free_rate.index)
        if len(common_dates) == 0:
            return {"error": "Nessun allineamento con risk-free rate"}
        
        aligned_returns = returns.loc[common_dates]
        aligned_rf = self.risk_free_rate.loc[common_dates]
        
        # Excess returns
        excess_returns = aligned_returns - aligned_rf
        
        # Sharpe ratio più preciso
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(12) if excess_returns.std() > 0 else 0
        
        # Treynor ratio (se abbiamo benchmark per beta)
        # Per ora, assumiamo beta = 1 (solo per reference)
        treynor_ratio = excess_returns.mean() * 12  # Semplificato
        
        return {
            "excess_return_annual": excess_returns.mean() * 12,
            "excess_volatility_annual": excess_returns.std() * np.sqrt(12),
            "sharpe_ratio_precise": sharpe_ratio,
            "treynor_ratio": treynor_ratio,
            "average_risk_free_rate": aligned_rf.mean() * 12
        }
    
    def calculate_rolling_metrics(self, 
                                returns: pd.Series,
                                window_months: int = 36) -> pd.DataFrame:
        """
        Calcola metriche rolling per analisi di stabilità.
        
        Args:
            returns: Serie rendimenti
            window_months: Finestra rolling in mesi
            
        Returns:
            DataFrame con metriche rolling
        """
        self.logger.info(f"📊 Calcolo rolling metrics (window={window_months}M)...")
        
        if len(returns) < window_months:
            self.logger.warning(f"⚠️ Dati insufficienti per rolling {window_months}M")
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window_months - 1, len(returns)):
            window_returns = returns.iloc[i-window_months+1:i+1]
            date = returns.index[i]
            
            # Calcola metriche per questa finestra
            metrics = {
                'Date': date,
                'Return_Annual': window_returns.mean() * 12,
                'Volatility_Annual': window_returns.std() * np.sqrt(12),
                'Sharpe_Ratio': (window_returns.mean() / window_returns.std()) * np.sqrt(12) if window_returns.std() > 0 else 0,
                'Max_Drawdown': self._calculate_rolling_drawdown(window_returns),
                'Hit_Ratio': (window_returns > 0).mean()
            }
            
            rolling_data.append(metrics)
        
        rolling_df = pd.DataFrame(rolling_data)
        rolling_df.set_index('Date', inplace=True)
        
        self.rolling_metrics = rolling_df
        
        self.logger.info(f"✅ Rolling metrics: {len(rolling_df)} periods")
        return rolling_df
    
    def _calculate_rolling_drawdown(self, returns: pd.Series) -> float:
        """Calcola max drawdown per finestra rolling."""
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
    
    def generate_performance_report(self) -> str:
        """
        Genera report testuale delle performance.
        
        Returns:
            Stringa con report formattato
        """
        if not self.performance_metrics:
            return "Nessuna metrica calcolata!"
        
        metrics = self.performance_metrics
        
        report = "=" * 60 + "\n"
        report += "TSMOM STRATEGY - PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Basic stats
        basic = metrics.get("basic_stats", {})
        report += f"PERIOD: {basic.get('start_date')} to {basic.get('end_date')}\n"
        report += f"OBSERVATIONS: {basic.get('observations')} months\n\n"
        
        # Return metrics
        returns_m = metrics.get("return_metrics", {})
        report += "RETURN METRICS:\n"
        report += f"  CAGR:                {returns_m.get('cagr', 0):.2%}\n"
        report += f"  Total Return:        {returns_m.get('total_return', 0):.2%}\n"
        report += f"  Annual Volatility:   {returns_m.get('volatility_annual', 0):.2%}\n"
        report += f"  Hit Ratio:           {returns_m.get('hit_ratio', 0):.2%}\n"
        report += f"  Best Month:          {returns_m.get('best_month', 0):.2%}\n"
        report += f"  Worst Month:         {returns_m.get('worst_month', 0):.2%}\n\n"
        
        # Risk metrics
        risk = metrics.get("risk_metrics", {})
        report += "RISK METRICS:\n"
        report += f"  Sharpe Ratio:        {risk.get('sharpe_ratio', 0):.2f}\n"
        report += f"  Sortino Ratio:       {risk.get('sortino_ratio', 0):.2f}\n"
        report += f"  VaR (95%):           {risk.get('var_95', 0):.2%}\n"
        report += f"  CVaR (95%):          {risk.get('cvar_95', 0):.2%}\n\n"
        
        # Drawdown metrics
        dd = metrics.get("drawdown_metrics", {})
        report += "DRAWDOWN METRICS:\n"
        report += f"  Maximum Drawdown:    {dd.get('max_drawdown', 0):.2%}\n"
        report += f"  Calmar Ratio:        {dd.get('calmar_ratio', 0):.2f}\n"
        report += f"  Avg DD Duration:     {dd.get('avg_drawdown_duration', 0):.1f} months\n"
        report += f"  Max DD Duration:     {dd.get('max_drawdown_duration', 0)} months\n\n"
        
        # Distribution metrics
        dist = metrics.get("distribution_metrics", {})
        report += "DISTRIBUTION METRICS:\n"
        report += f"  Skewness:            {dist.get('skewness', 0):.3f}\n"
        report += f"  Kurtosis:            {dist.get('kurtosis', 0):.3f}\n"
        report += f"  Normality Test:      {'Pass' if dist.get('normality_test', False) else 'Fail'}\n\n"
        
        return report
    
    def export_performance_data(self, output_path: str):
        """
        Esporta tutti i dati di performance.
        
        Args:
            output_path: Path base per export
        """
        # Performance metrics
        if self.performance_metrics:
            import json
            with open(f"{output_path}_performance_metrics.json", 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
        
        # Rolling metrics
        if not self.rolling_metrics.empty:
            self.rolling_metrics.to_csv(f"{output_path}_rolling_metrics.csv")
            self.rolling_metrics.to_parquet(f"{output_path}_rolling_metrics.parquet")
        
        # Performance report
        report = self.generate_performance_report()
        with open(f"{output_path}_performance_report.txt", 'w') as f:
            f.write(report)
        
        self.logger.info(f"💾 Performance data esportati in {output_path}_*")