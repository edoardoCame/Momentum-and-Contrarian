"""
Data Manager per TSMOM Strategy - Moskowitz, Ooi & Pedersen (2012)
Gestisce il download dei dati dai futures su commodities e T-Bill da Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import time
import logging

warnings.filterwarnings('ignore')

class TSMOMDataManager:
    """
    Gestore dati per la strategia TSMOM seguendo esattamente le specifiche del paper MOP (2012).
    
    Caratteristiche chiave:
    - Download robusto dei continuous futures da Yahoo Finance
    - T-Bill 3-month (^IRX) per risk-free rate
    - Gestione automatica degli errori e retry logic
    - Validazione qualità dati e controlli di integrità
    """
    
    # Universo futures con dati storici estesi (2000-2024, 20+ anni)
    # Basato su verifica dati storici disponibili su Yahoo Finance
    DEFAULT_UNIVERSE = {
        'Energy': ["CL=F", "NG=F", "HO=F", "RB=F"],  # Rimosso BZ=F (solo dal 2007)
        'Metals_Precious': ["GC=F", "SI=F", "PL=F", "PA=F"],  # Esteso con Platino/Palladio
        'Metals_Industrial': ["HG=F"], 
        'Agriculture_Softs': ["KC=F", "CC=F", "SB=F", "CT=F", "OJ=F"],  # Rimosso LB=F (non disponibile)
        'Agriculture_Grains': ["ZS=F", "ZC=F", "ZW=F", "ZM=F", "ZL=F", "ZO=F", "KE=F", "ZR=F"],  # Esteso
        'Livestock': ["HE=F", "LE=F", "GF=F"]  # Nuovo settore aggiunto
    }
    
    def __init__(self, 
                 start_date: str = "2000-01-01",
                 end_date: Optional[str] = None,
                 universe: Optional[Dict[str, List[str]]] = None):
        """
        Inizializza il data manager.
        
        Args:
            start_date: Data inizio (formato YYYY-MM-DD)  
            end_date: Data fine (default: oggi)
            universe: Dizionario con futures per categoria (default: universo completo)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.universe = universe or self.DEFAULT_UNIVERSE
        
        # Flatten universe per facilitare il processing
        self.all_tickers = []
        self.ticker_to_sector = {}
        for sector, tickers in self.universe.items():
            self.all_tickers.extend(tickers)
            for ticker in tickers:
                self.ticker_to_sector[ticker] = sector
                
        # Storage per i dati scaricati
        self.futures_data = {}
        self.tbill_data = None
        self.failed_downloads = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_futures_data(self, 
                            max_retries: int = 3,
                            retry_delay: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Scarica i dati dei futures con retry logic robusta.
        
        Args:
            max_retries: Numero massimo di tentativi per ticker falliti
            retry_delay: Delay tra tentativi (secondi)
            
        Returns:
            Dict con ticker -> DataFrame (OHLCV data)
        """
        self.logger.info(f"📊 Avvio download di {len(self.all_tickers)} futures...")
        self.logger.info(f"📅 Periodo: {self.start_date} -> {self.end_date}")
        
        successful_downloads = {}
        failed_tickers = []
        
        for i, ticker in enumerate(self.all_tickers):
            self.logger.info(f"Downloading {ticker} ({i+1}/{len(self.all_tickers)})...")
            
            success = False
            for attempt in range(max_retries):
                try:
                    # Download con yfinance
                    ticker_obj = yf.Ticker(ticker)
                    data = ticker_obj.history(
                        start=self.start_date,
                        end=self.end_date,
                        auto_adjust=True,  # Usa prezzi aggiustati
                        back_adjust=True
                    )
                    
                    if data.empty or len(data) < 100:  # Controllo minimo dati
                        raise ValueError(f"Dati insufficienti per {ticker}: {len(data)} observations")
                    
                    # yfinance ora restituisce: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                    # Mantieni solo le colonne OHLCV che ci servono
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    optional_cols = ['Volume']
                    
                    # Verifica che abbiamo almeno OHLC
                    missing_required = [col for col in required_cols if col not in data.columns]
                    if missing_required:
                        raise ValueError(f"Missing required columns for {ticker}: {missing_required}")
                    
                    # Seleziona solo le colonne che ci servono
                    cols_to_keep = required_cols.copy()
                    for col in optional_cols:
                        if col in data.columns:
                            cols_to_keep.append(col)
                    
                    data = data[cols_to_keep]
                    data.index.name = 'Date'
                    
                    # Converti in timezone naive come richiesto
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Validazioni base
                    self._validate_price_data(data, ticker)
                    
                    successful_downloads[ticker] = data
                    success = True
                    self.logger.info(f"✅ {ticker}: {len(data)} observations from {data.index.min().date()} to {data.index.max().date()}")
                    break
                    
                except Exception as e:
                    self.logger.warning(f"❌ Attempt {attempt+1} failed for {ticker}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        failed_tickers.append(ticker)
                        self.failed_downloads.append((ticker, str(e)))
            
            # Breve delay tra download per evitare rate limiting
            if i < len(self.all_tickers) - 1:
                time.sleep(0.2)
        
        self.futures_data = successful_downloads
        
        self.logger.info(f"\n📈 Download completato!")
        self.logger.info(f"✅ Successi: {len(successful_downloads)}")
        self.logger.info(f"❌ Falliti: {len(failed_tickers)}")
        if failed_tickers:
            self.logger.warning(f"Failed tickers: {failed_tickers}")
            
        return successful_downloads
    
    def download_tbill_data(self) -> pd.Series:
        """
        Scarica dati T-Bill 3-month (^IRX) per il risk-free rate.
        
        Returns:
            Serie con yield annualizzati (%) del T-Bill
        """
        self.logger.info("📊 Download T-Bill 3-month (^IRX)...")
        
        try:
            tbill = yf.Ticker("^IRX")
            data = tbill.history(
                start=self.start_date,
                end=self.end_date
            )
            
            if data.empty:
                raise ValueError("Nessun dato T-Bill scaricato")
            
            # Usa Close price (yield %)
            tbill_yields = data['Close']
            
            # Converti in timezone naive
            if tbill_yields.index.tz is not None:
                tbill_yields.index = tbill_yields.index.tz_localize(None)
            
            # Valida dati
            if tbill_yields.isnull().sum() > len(tbill_yields) * 0.1:
                self.logger.warning("⚠️ Molti valori NaN nei dati T-Bill")
            
            # Forward fill per valori mancanti (pratica standard)
            tbill_yields = tbill_yields.fillna(method='ffill')
            
            self.tbill_data = tbill_yields
            self.logger.info(f"✅ T-Bill: {len(tbill_yields)} observations, range: {tbill_yields.min():.2f}% - {tbill_yields.max():.2f}%")
            
            return tbill_yields
            
        except Exception as e:
            self.logger.error(f"❌ Errore download T-Bill: {e}")
            # Fallback: usa rate fisso del 2%
            self.logger.warning("📊 Usando risk-free rate fisso del 2%")
            
            # Crea serie con date business dei futures
            if self.futures_data:
                sample_data = list(self.futures_data.values())[0]
                fallback_rf = pd.Series(2.0, index=sample_data.index)
                self.tbill_data = fallback_rf
                return fallback_rf
            else:
                raise ValueError("Impossibile creare fallback T-Bill senza dati futures")
    
    def get_futures_prices(self, price_type: str = 'Close') -> pd.DataFrame:
        """
        Estrae matrice dei prezzi dai dati scaricati.
        
        Args:
            price_type: Tipo di prezzo ('Close', 'Open', etc.)
            
        Returns:
            DataFrame con date x tickers dei prezzi
        """
        if not self.futures_data:
            raise ValueError("Nessun dato futures scaricato. Esegui download_futures_data() prima.")
        
        price_matrix = pd.DataFrame()
        
        for ticker, data in self.futures_data.items():
            if price_type in data.columns:
                price_matrix[ticker] = data[price_type]
            else:
                self.logger.warning(f"⚠️ {price_type} non trovato per {ticker}")
        
        # Remove rows con tutti NaN
        price_matrix = price_matrix.dropna(how='all')
        
        self.logger.info(f"📊 Matrice prezzi: {price_matrix.shape[0]} dates x {price_matrix.shape[1]} tickers")
        
        return price_matrix
    
    def get_risk_free_rate_monthly(self) -> pd.Series:
        """
        Converte T-Bill yields in rendimenti mensili risk-free.
        
        Formula MOP (2012): rf_monthly = (IRX_annual / 100) / 12
        Allineato a fine mese business day come specificato.
        
        Returns:
            Serie dei rendimenti mensili risk-free
        """
        if self.tbill_data is None:
            raise ValueError("Nessun dato T-Bill disponibile. Esegui download_tbill_data() prima.")
        
        # Converti yield annuale % in rendimento mensile semplice
        # IRX è in %, quindi dividi per 100 prima
        monthly_rf = (self.tbill_data / 100) / 12
        
        # Resample all'ultimo giorno business del mese
        monthly_rf_resampled = monthly_rf.resample('BM').last()
        
        self.logger.info(f"📊 Risk-free mensile: {len(monthly_rf_resampled)} observations, media: {monthly_rf_resampled.mean()*100:.3f}% mensile")
        
        return monthly_rf_resampled
    
    def get_data_summary(self) -> Dict:
        """
        Genera summary statistico dei dati scaricati.
        
        Returns:
            Dict con statistiche descrittive
        """
        if not self.futures_data:
            return {"error": "Nessun dato disponibile"}
        
        summary = {
            "total_tickers": len(self.futures_data),
            "failed_tickers": len(self.failed_downloads),
            "sectors": {},
            "date_range": {},
            "data_quality": {}
        }
        
        # Summary per settore
        for sector, tickers in self.universe.items():
            successful_in_sector = [t for t in tickers if t in self.futures_data]
            summary["sectors"][sector] = {
                "total": len(tickers),
                "successful": len(successful_in_sector),
                "tickers": successful_in_sector
            }
        
        # Date range
        all_start_dates = [data.index.min() for data in self.futures_data.values()]
        all_end_dates = [data.index.max() for data in self.futures_data.values()]
        
        summary["date_range"] = {
            "earliest_start": min(all_start_dates).date(),
            "latest_start": max(all_start_dates).date(), 
            "earliest_end": min(all_end_dates).date(),
            "latest_end": max(all_end_dates).date()
        }
        
        # Data quality
        price_matrix = self.get_futures_prices()
        summary["data_quality"] = {
            "total_observations": len(price_matrix),
            "missing_data_pct": (price_matrix.isnull().sum().sum() / price_matrix.size) * 100,
            "tickers_with_missing": (price_matrix.isnull().any()).sum()
        }
        
        return summary
    
    def _validate_price_data(self, data: pd.DataFrame, ticker: str):
        """
        Valida la qualità dei dati scaricati (versione rilassata).
        
        Args:
            data: DataFrame dei prezzi
            ticker: Nome del ticker
        """
        # Check solo per NaN eccessivi (soglia rilassata)
        nan_pct = data.isnull().sum().sum() / data.size * 100
        if nan_pct > 50:  # Soglia più permissiva
            raise ValueError(f"Troppi NaN ({nan_pct:.1f}%) in {ticker}")
        
        # Ignora check per prezzi negativi (petrolio 2020 aveva prezzi negativi legittimi)
        # Ignora check per high/low inconsistencies (potrebbero essere split/aggiustamenti)
        # Ignora check per outlier (sono normali nei commodity durante crisi)
        
        # Solo warning per informare, senza bloccare download
        if (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            self.logger.warning(f"⚠️ Prezzi negativi/zero trovati in {ticker} (normale per alcuni periodi)")
        
        if (data['High'] < data['Low']).any():
            self.logger.warning(f"⚠️ High < Low trovato in {ticker} (possibili aggiustamenti)")
        
        daily_returns = data['Close'].pct_change().abs()
        if (daily_returns > 0.5).any():
            self.logger.warning(f"⚠️ Outlier estremi trovati in {ticker} (normale in periodi volatili)")
    
    def load_cached_data(self, cache_dir: str = "data/") -> bool:
        """
        Carica i dati dalla cache se disponibili.
        
        Args:
            cache_dir: Directory della cache
            
        Returns:
            True se i dati sono stati caricati, False altrimenti
        """
        import os
        
        if not os.path.exists(cache_dir):
            return False
        
        # Controlla se abbiamo i file cached per tutti i tickers
        cached_files = []
        for ticker in self.all_tickers:
            filename = f"{cache_dir}/{ticker.replace('=', '_')}.parquet"
            if os.path.exists(filename):
                cached_files.append((ticker, filename))
        
        tbill_file = f"{cache_dir}/tbill_data.parquet"
        
        if len(cached_files) < len(self.all_tickers) * 0.8:  # Se meno dell'80% è cached, rifai download
            return False
        
        try:
            # Carica futures data
            self.futures_data = {}
            for ticker, filename in cached_files:
                data = pd.read_parquet(filename)
                self.futures_data[ticker] = data
                
            # Carica T-Bill data se disponibile
            if os.path.exists(tbill_file):
                tbill_df = pd.read_parquet(tbill_file)
                self.tbill_data = tbill_df['TBill_Yield']
                
            self.logger.info(f"📂 Caricati {len(self.futures_data)} futures dalla cache: {cache_dir}")
            return True
            
        except Exception as e:
            self.logger.warning(f"❌ Errore caricamento cache: {e}")
            return False

    def save_data(self, output_dir: str = "data/"):
        """
        Salva i dati scaricati in formato parquet per riuso futuro.
        
        Args:
            output_dir: Directory di output
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva futures data
        for ticker, data in self.futures_data.items():
            filename = f"{output_dir}/{ticker.replace('=', '_')}.parquet"
            data.to_parquet(filename)
            
        # Salva T-Bill data
        if self.tbill_data is not None:
            self.tbill_data.to_frame('TBill_Yield').to_parquet(f"{output_dir}/tbill_data.parquet")
            
        self.logger.info(f"💾 Dati salvati in {output_dir}")
    
    def get_or_download_data(self, cache_dir: str = "data/") -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
        """
        Carica dati dalla cache o li scarica se necessario.
        
        Args:
            cache_dir: Directory della cache
            
        Returns:
            Tuple (futures_data, tbill_data)
        """
        # Prova a caricare dalla cache
        if self.load_cached_data(cache_dir):
            self.logger.info("✅ Dati caricati dalla cache")
        else:
            self.logger.info("📥 Cache non disponibile, download in corso...")
            # Download nuovi dati
            self.download_futures_data()
            self.download_tbill_data()
            # Salva in cache
            self.save_data(cache_dir)
            
        return self.futures_data, self.tbill_data