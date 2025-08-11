"""
Visualizer per TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012)
Crea visualizzazioni professionali per l'analisi della strategia TSMOM.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict
import warnings
import logging

warnings.filterwarnings('ignore')

class TSMOMVisualizer:
    """
    Visualizzatore per analisi TSMOM con grafici professionali.
    
    Grafici implementati:
    - Equity curves (excess & total returns)
    - Drawdown analysis
    - Rolling performance metrics
    - Commodity contribution heatmaps
    - Signal distribution analysis
    - Volatility time series
    """
    
    def __init__(self, style: str = 'whitegrid'):
        """
        Inizializza il visualizzatore.
        
        Args:
            style: Stile seaborn per i grafici
        """
        # Setup plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_style(style)
        sns.set_palette("husl")
        
        # Configurazione generale
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        
        self.logger = logging.getLogger(__name__)
    
    def plot_equity_curves(self, 
                          portfolio_returns: pd.Series,
                          benchmark_returns: Optional[pd.Series] = None,
                          risk_free_rate: Optional[pd.Series] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plotta equity curves comprehensive.
        
        Args:
            portfolio_returns: Serie rendimenti portafoglio
            benchmark_returns: Serie rendimenti benchmark (opzionale)
            risk_free_rate: Serie risk-free rate (opzionale)
            figsize: Dimensione figura
            
        Returns:
            Figura matplotlib
        """
        self.logger.info("📊 Generazione equity curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('TSMOM Strategy - Equity Curves Analysis', fontsize=16, y=0.95)
        
        # 1. Cumulative returns comparison
        ax1 = axes[0, 0]
        
        # TSMOM equity curve
        tsmom_equity = (1 + portfolio_returns).cumprod()
        ax1.plot(tsmom_equity.index, tsmom_equity.values, 
                label='TSMOM Strategy', linewidth=2, color='navy')
        
        # Benchmark se disponibile
        if benchmark_returns is not None:
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                benchmark_equity = (1 + benchmark_returns.loc[common_dates]).cumprod()
                ax1.plot(benchmark_equity.index, benchmark_equity.values,
                        label='Benchmark', linewidth=2, color='red', alpha=0.7)
        
        # Risk-free se disponibile
        if risk_free_rate is not None:
            common_dates = portfolio_returns.index.intersection(risk_free_rate.index)
            if len(common_dates) > 0:
                rf_equity = (1 + risk_free_rate.loc[common_dates]).cumprod()
                ax1.plot(rf_equity.index, rf_equity.values,
                        label='Risk-Free', linewidth=1, color='green', alpha=0.5)
        
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale per meglio visualizzare compound returns
        
        # 2. Monthly returns distribution
        ax2 = axes[0, 1]
        ax2.hist(portfolio_returns.dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(portfolio_returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {portfolio_returns.mean():.3%}')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Monthly Returns Distribution')
        ax2.set_xlabel('Monthly Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis as percentage
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # 3. Rolling returns (12M)
        ax3 = axes[1, 0]
        rolling_12m = portfolio_returns.rolling(12).sum()
        ax3.plot(rolling_12m.index, rolling_12m.values, color='purple', linewidth=1.5)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.fill_between(rolling_12m.index, rolling_12m.values, 0, 
                        where=(rolling_12m.values >= 0), color='green', alpha=0.3, label='Positive')
        ax3.fill_between(rolling_12m.index, rolling_12m.values, 0,
                        where=(rolling_12m.values < 0), color='red', alpha=0.3, label='Negative')
        ax3.set_title('Rolling 12-Month Returns')
        ax3.set_ylabel('12M Return')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 4. Excess returns vs risk-free
        ax4 = axes[1, 1]
        if risk_free_rate is not None:
            common_dates = portfolio_returns.index.intersection(risk_free_rate.index)
            if len(common_dates) > 0:
                excess_returns = portfolio_returns.loc[common_dates] - risk_free_rate.loc[common_dates]
                excess_equity = (1 + excess_returns).cumprod()
                ax4.plot(excess_equity.index, excess_equity.values, 
                        color='darkgreen', linewidth=2, label='Excess Returns')
                ax4.set_title('Excess Returns vs Risk-Free')
                ax4.set_ylabel('Cumulative Excess Return')
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'No Risk-Free Rate Data', 
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            # Plot portfolio returns se non c'è risk-free
            tsmom_equity.plot(ax=ax4, color='navy', linewidth=2)
            ax4.set_title('Portfolio Returns')
            ax4.set_ylabel('Cumulative Return')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown_analysis(self, 
                             portfolio_returns: pd.Series,
                             figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Plotta analisi dettagliata dei drawdowns.
        
        Args:
            portfolio_returns: Serie rendimenti portafoglio
            figsize: Dimensione figura
            
        Returns:
            Figura matplotlib
        """
        self.logger.info("📊 Generazione drawdown analysis...")
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle('TSMOM Strategy - Drawdown Analysis', fontsize=16, y=0.95)
        
        # Calcola equity curve e drawdown
        equity_curve = (1 + portfolio_returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        # 1. Equity curve con peaks
        ax1 = axes[0]
        ax1.plot(equity_curve.index, equity_curve.values, 
                label='Equity Curve', linewidth=2, color='navy')
        ax1.plot(running_max.index, running_max.values,
                label='Running Maximum', linewidth=1, color='red', alpha=0.7)
        
        # Highlight major drawdown periods
        major_dd_threshold = -0.10  # -10%
        major_dd_periods = drawdown < major_dd_threshold
        if major_dd_periods.any():
            ax1.fill_between(drawdown.index, equity_curve.values, running_max.values,
                           where=major_dd_periods, color='red', alpha=0.3, 
                           label='Major Drawdowns (>10%)')
        
        ax1.set_title('Equity Curve with Drawdown Periods')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Drawdown time series
        ax2 = axes[1]
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        color='red', alpha=0.7, label='Drawdown')
        ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        
        # Mark maximum drawdown
        max_dd_date = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax2.scatter(max_dd_date, max_dd_value, color='red', s=100, zorder=5)
        ax2.annotate(f'Max DD: {max_dd_value:.1%}\n{max_dd_date.strftime("%Y-%m")}',
                    xy=(max_dd_date, max_dd_value), 
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax2.set_title('Drawdown Time Series')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_metrics(self, 
                           rolling_metrics: pd.DataFrame,
                           figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Plotta metriche rolling per analisi di stabilità.
        
        Args:
            rolling_metrics: DataFrame con metriche rolling (da PerformanceAnalyzer)
            figsize: Dimensione figura
            
        Returns:
            Figura matplotlib
        """
        self.logger.info("📊 Generazione rolling metrics...")
        
        if rolling_metrics.empty:
            self.logger.warning("⚠️ Nessuna metrica rolling disponibile")
            return plt.figure()
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('TSMOM Strategy - Rolling Performance Metrics (36M Windows)', 
                    fontsize=16, y=0.95)
        
        # 1. Rolling returns
        ax1 = axes[0, 0]
        if 'Return_Annual' in rolling_metrics.columns:
            ax1.plot(rolling_metrics.index, rolling_metrics['Return_Annual'], 
                    color='blue', linewidth=1.5)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax1.set_title('Rolling Annual Returns')
            ax1.set_ylabel('Annual Return')
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 2. Rolling volatility
        ax2 = axes[0, 1]
        if 'Volatility_Annual' in rolling_metrics.columns:
            ax2.plot(rolling_metrics.index, rolling_metrics['Volatility_Annual'], 
                    color='red', linewidth=1.5)
            ax2.set_title('Rolling Annual Volatility')
            ax2.set_ylabel('Annual Volatility')
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 3. Rolling Sharpe ratio
        ax3 = axes[1, 0]
        if 'Sharpe_Ratio' in rolling_metrics.columns:
            ax3.plot(rolling_metrics.index, rolling_metrics['Sharpe_Ratio'], 
                    color='green', linewidth=1.5)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Sharpe = 1')
            ax3.set_title('Rolling Sharpe Ratio')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Rolling max drawdown
        ax4 = axes[1, 1]
        if 'Max_Drawdown' in rolling_metrics.columns:
            ax4.fill_between(rolling_metrics.index, rolling_metrics['Max_Drawdown'], 0,
                           color='red', alpha=0.7)
            ax4.plot(rolling_metrics.index, rolling_metrics['Max_Drawdown'], 
                    color='darkred', linewidth=1)
            ax4.set_title('Rolling Maximum Drawdown')
            ax4.set_ylabel('Max Drawdown')
            ax4.grid(True, alpha=0.3)
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 5. Rolling hit ratio
        ax5 = axes[2, 0]
        if 'Hit_Ratio' in rolling_metrics.columns:
            ax5.plot(rolling_metrics.index, rolling_metrics['Hit_Ratio'], 
                    color='purple', linewidth=1.5)
            ax5.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50%')
            ax5.set_title('Rolling Hit Ratio')
            ax5.set_ylabel('Hit Ratio')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 6. Risk-return scatter (ultima finestra vs storica)
        ax6 = axes[2, 1]
        if 'Return_Annual' in rolling_metrics.columns and 'Volatility_Annual' in rolling_metrics.columns:
            # Scatter plot risk-return per ogni periodo
            ax6.scatter(rolling_metrics['Volatility_Annual'], rolling_metrics['Return_Annual'],
                       alpha=0.6, s=20, c=range(len(rolling_metrics)), cmap='viridis')
            
            # Media storica
            mean_vol = rolling_metrics['Volatility_Annual'].mean()
            mean_ret = rolling_metrics['Return_Annual'].mean()
            ax6.scatter(mean_vol, mean_ret, color='red', s=100, marker='*', 
                       label='Average', zorder=5)
            
            ax6.set_title('Risk-Return Profile Over Time')
            ax6.set_xlabel('Volatility (Annual)')
            ax6.set_ylabel('Return (Annual)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        return fig
    
    def plot_commodity_heatmap(self, 
                             contract_weights: pd.DataFrame,
                             portfolio_returns: pd.Series,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plotta heatmap delle contribuzioni per commodity.
        
        Args:
            contract_weights: DataFrame pesi contratti
            portfolio_returns: Serie rendimenti portafoglio
            figsize: Dimensione figura
            
        Returns:
            Figura matplotlib
        """
        self.logger.info("📊 Generazione commodity contribution heatmap...")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('TSMOM Strategy - Commodity Analysis', fontsize=16, y=0.95)
        
        # 1. Average weights heatmap
        ax1 = axes[0, 0]
        
        # Calcola pesi medi per anno e commodity
        weights_annual = contract_weights.groupby(contract_weights.index.year).mean()
        
        # Prendi top commodities per peso medio
        top_commodities = contract_weights.abs().mean().nlargest(10).index
        weights_for_heatmap = weights_annual[top_commodities]
        
        sns.heatmap(weights_for_heatmap.T, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, ax=ax1, cbar_kws={'label': 'Average Weight'})
        ax1.set_title('Average Weights by Year (Top 10 Commodities)')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Commodity')
        
        # 2. Position frequency heatmap  
        ax2 = axes[0, 1]
        
        # Calcola frequenza posizioni (long/short) per anno
        long_freq = (contract_weights > 0).groupby(contract_weights.index.year).mean()
        short_freq = (contract_weights < 0).groupby(contract_weights.index.year).mean()
        
        position_freq = long_freq - short_freq  # Net long frequency
        position_freq_top = position_freq[top_commodities]
        
        sns.heatmap(position_freq_top.T, annot=True, fmt='.2f', cmap='RdBu', 
                   center=0, ax=ax2, cbar_kws={'label': 'Net Long Frequency'})
        ax2.set_title('Net Long Frequency by Year')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Commodity')
        
        # 3. Commodity contribution to returns
        ax3 = axes[1, 0]
        
        # Calcola contribuzione approssimativa (peso medio × return medio)
        if len(portfolio_returns) > 0:
            monthly_data = pd.DataFrame({
                'portfolio_return': portfolio_returns,
                'year': portfolio_returns.index.year
            })
            annual_returns = monthly_data.groupby('year')['portfolio_return'].sum()
            
            # Bar plot dei rendimenti annuali
            bars = ax3.bar(annual_returns.index, annual_returns.values,
                          color=['green' if x >= 0 else 'red' for x in annual_returns.values],
                          alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_title('Annual Portfolio Returns')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Annual Return')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 4. Active positions over time
        ax4 = axes[1, 1]
        
        active_long = (contract_weights > 0).sum(axis=1)
        active_short = (contract_weights < 0).sum(axis=1)
        
        ax4.plot(active_long.index, active_long.values, 
                label='Long Positions', color='green', linewidth=1.5)
        ax4.plot(active_short.index, active_short.values,
                label='Short Positions', color='red', linewidth=1.5)
        ax4.fill_between(active_long.index, active_long.values, 0, 
                        color='green', alpha=0.3)
        ax4.fill_between(active_short.index, active_short.values, 0,
                        color='red', alpha=0.3)
        
        ax4.set_title('Active Positions Over Time')
        ax4.set_ylabel('Number of Positions')
        ax4.set_xlabel('Date')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_signal_analysis(self, 
                           signals: pd.DataFrame,
                           momentum_data: Optional[pd.DataFrame] = None,
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plotta analisi dettagliata dei segnali TSMOM.
        
        Args:
            signals: DataFrame segnali TSMOM
            momentum_data: DataFrame momentum cumulativo (opzionale)
            figsize: Dimensione figura
            
        Returns:
            Figura matplotlib
        """
        self.logger.info("📊 Generazione signal analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('TSMOM Strategy - Signal Analysis', fontsize=16, y=0.95)
        
        # 1. Signal distribution over time
        ax1 = axes[0, 0]
        
        monthly_long = (signals == 1).sum(axis=1)
        monthly_short = (signals == -1).sum(axis=1)
        monthly_neutral = (signals == 0).sum(axis=1)
        
        ax1.plot(monthly_long.index, monthly_long.values, 
                label='Long Signals', color='green', linewidth=1.5)
        ax1.plot(monthly_short.index, monthly_short.values,
                label='Short Signals', color='red', linewidth=1.5)
        ax1.plot(monthly_neutral.index, monthly_neutral.values,
                label='Neutral', color='gray', linewidth=1, alpha=0.7)
        
        ax1.set_title('Signal Distribution Over Time')
        ax1.set_ylabel('Number of Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Signal frequency by commodity
        ax2 = axes[0, 1]
        
        # Calcola % long per ogni commodity
        long_frequency = (signals == 1).mean() * 100
        long_frequency = long_frequency.sort_values(ascending=True)
        
        # Prendi top 10 commodities
        top_10 = long_frequency.tail(10)
        colors = ['green' if x > 50 else 'red' for x in top_10.values]
        
        bars = ax2.barh(range(len(top_10)), top_10.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_10)))
        ax2.set_yticklabels(top_10.index, fontsize=8)
        ax2.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='50%')
        ax2.set_title('Long Signal Frequency (Top 10)')
        ax2.set_xlabel('Long Signal %')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Signal correlation heatmap
        ax3 = axes[1, 0]
        
        # Prendi sample di commodities per correlation
        sample_tickers = signals.columns[:8]  # Prime 8 commodities
        signal_corr = signals[sample_tickers].corr()
        
        sns.heatmap(signal_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax3, 
                   cbar_kws={'label': 'Signal Correlation'})
        ax3.set_title('Signal Correlation Matrix (Sample)')
        
        # 4. Momentum distribution (se disponibile)
        ax4 = axes[1, 1]
        
        if momentum_data is not None:
            # Flatten tutti i momentum values
            all_momentum = momentum_data.values.flatten()
            all_momentum = all_momentum[~np.isnan(all_momentum)]
            
            ax4.hist(all_momentum, bins=50, alpha=0.7, color='skyblue', 
                    density=True, edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                       label='Zero Line')
            ax4.set_title('12M Momentum Distribution')
            ax4.set_xlabel('12M Cumulative Return')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        else:
            # Signal time series per alcuni tickers se no momentum
            sample_tickers_ts = signals.columns[:3]
            for i, ticker in enumerate(sample_tickers_ts):
                offset = i * 0.1
                ax4.scatter(signals.index, signals[ticker] + offset, 
                           alpha=0.6, s=15, label=ticker)
            ax4.set_title('Signal Time Series (Sample)')
            ax4.set_ylabel('Signal Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, 
                      portfolio_returns: pd.Series,
                      contract_weights: pd.DataFrame,
                      signals: pd.DataFrame,
                      output_dir: str,
                      rolling_metrics: Optional[pd.DataFrame] = None,
                      momentum_data: Optional[pd.DataFrame] = None,
                      benchmark_returns: Optional[pd.Series] = None,
                      risk_free_rate: Optional[pd.Series] = None):
        """
        Salva tutti i grafici in una directory.
        
        Args:
            portfolio_returns: Serie rendimenti portafoglio
            contract_weights: DataFrame pesi contratti
            signals: DataFrame segnali
            output_dir: Directory output
            rolling_metrics: DataFrame metriche rolling (opzionale)
            momentum_data: DataFrame momentum (opzionale)
            benchmark_returns: Serie benchmark (opzionale)
            risk_free_rate: Serie risk-free (opzionale)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"💾 Salvataggio grafici in {output_dir}...")
        
        # 1. Equity curves
        fig1 = self.plot_equity_curves(portfolio_returns, benchmark_returns, risk_free_rate)
        fig1.savefig(f"{output_dir}/equity_curves.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Drawdown analysis  
        fig2 = self.plot_drawdown_analysis(portfolio_returns)
        fig2.savefig(f"{output_dir}/drawdown_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Rolling metrics (se disponibili)
        if rolling_metrics is not None and not rolling_metrics.empty:
            fig3 = self.plot_rolling_metrics(rolling_metrics)
            fig3.savefig(f"{output_dir}/rolling_metrics.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        # 4. Commodity heatmap
        fig4 = self.plot_commodity_heatmap(contract_weights, portfolio_returns)
        fig4.savefig(f"{output_dir}/commodity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # 5. Signal analysis
        fig5 = self.plot_signal_analysis(signals, momentum_data)
        fig5.savefig(f"{output_dir}/signal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        self.logger.info("✅ Tutti i grafici salvati!")