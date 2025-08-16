#!/usr/bin/env python3
"""
Script per scaricare coppie valutarie aggiuntive da Yahoo Finance
e salvarle in formato parquet nella directory data/forex/

Aggiunge 14 nuove coppie liquide per espandere il dataset forex.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Lista delle nuove coppie valutarie da scaricare
NEW_FOREX_PAIRS = [
    # Major Pairs con valute europee
    'EURCZK=X',  # EUR/CZK - Corona ceca
    'EURPLN=X',  # EUR/PLN - Zloty polacco  
    'EURSEK=X',  # EUR/SEK - Corona svedese
    'EURNOK=X',  # EUR/NOK - Corona norvegese
    
    # Cross Pairs liquidi
    'CADCHF=X',  # CAD/CHF 
    'CADJPY=X',  # CAD/JPY
    'CHFJPY=X',  # CHF/JPY
    'AUDCAD=X',  # AUD/CAD
    
    # Valute emergenti liquide
    'USDSGD=X',  # USD/SGD - Dollaro Singapore
    'USDHKD=X',  # USD/HKD - Dollaro Hong Kong
    'USDZAR=X',  # USD/ZAR - Rand sudafricano
    'USDMXN=X',  # USD/MXN - Peso messicano
    
    # Additional Cross pairs
    'NOKJPY=X',  # NOK/JPY
    'SEKJPY=X',  # SEK/JPY
]

def find_data_directory():
    """Trova la directory data/forex/ utilizzando vari percorsi possibili."""
    possible_paths = [
        "../../../data/forex",  # Da approaches/time_series_momentum/
        "../../data/forex",     # Da notebooks
        "../data/forex",        # Da approaches
        "data/forex",           # Da root
        "../../../../data/forex"  # Percorso alternativo
    ]
    
    for path in possible_paths:
        test_path = Path(path)
        if test_path.exists():
            print(f"✅ Directory data trovata: {test_path.resolve()}")
            return test_path
    
    # Se non trova nessuna directory, crea la struttura
    data_dir = Path("../../../data/forex")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Directory creata: {data_dir.resolve()}")
    return data_dir

def download_forex_pair(symbol, data_dir, start_date="2000-01-01"):
    """
    Scarica una singola coppia valutaria da Yahoo Finance.
    
    Args:
        symbol: Simbolo Yahoo Finance (es. EURUSD=X)
        data_dir: Directory dove salvare il file
        start_date: Data di inizio per il download
        
    Returns:
        bool: True se il download è riuscito, False altrimenti
    """
    try:
        print(f"📥 Scaricando {symbol}...", end=" ")
        
        # Download dati da Yahoo Finance
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, auto_adjust=True, back_adjust=True)
        
        if data.empty:
            print("❌ Nessun dato disponibile")
            return False
        
        # Pulisci i dati
        data.index = pd.to_datetime(data.index)
        data = data.dropna()
        
        # Verifica che abbiamo dati sufficienti
        if len(data) < 100:  # Almeno 100 giorni di dati
            print(f"❌ Dati insufficienti ({len(data)} giorni)")
            return False
        
        # Converti il nome del simbolo per il salvataggio
        filename = symbol.replace('=', '_') + '.parquet'
        filepath = data_dir / filename
        
        # Salva in formato parquet
        data.to_parquet(filepath)
        
        # Statistiche
        duration_years = (data.index[-1] - data.index[0]).days / 365.25
        print(f"✅ {len(data)} giorni ({duration_years:.1f} anni: {data.index[0].strftime('%Y-%m-%d')} → {data.index[-1].strftime('%Y-%m-%d')})")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        return False

def main():
    """Funzione principale per scaricare tutte le nuove coppie valutarie."""
    
    print("🚀 Download Coppie Valutarie Aggiuntive")
    print("=" * 60)
    
    # Trova directory data
    data_dir = find_data_directory()
    
    # Verifica coppie già esistenti
    existing_files = list(data_dir.glob("*_X.parquet"))
    existing_pairs = [f.stem for f in existing_files]
    
    print(f"📊 Coppie esistenti: {len(existing_pairs)}")
    print(f"🆕 Nuove coppie da scaricare: {len(NEW_FOREX_PAIRS)}")
    print()
    
    # Scarica ogni coppia
    successful_downloads = 0
    failed_downloads = []
    
    for i, symbol in enumerate(NEW_FOREX_PAIRS, 1):
        print(f"[{i:2d}/{len(NEW_FOREX_PAIRS)}] ", end="")
        
        # Controlla se esiste già
        filename_check = symbol.replace('=', '_') + '.parquet'
        if (data_dir / filename_check).exists():
            print(f"⏭️  {symbol} già esistente, saltato")
            successful_downloads += 1
            continue
        
        # Scarica
        if download_forex_pair(symbol, data_dir):
            successful_downloads += 1
        else:
            failed_downloads.append(symbol)
        
        # Piccola pausa per non sovraccaricare Yahoo Finance
        time.sleep(0.5)
    
    # Riepilogo finale
    print()
    print("=" * 60)
    print("📊 RIEPILOGO DOWNLOAD")
    print("=" * 60)
    print(f"✅ Download riusciti: {successful_downloads}/{len(NEW_FOREX_PAIRS)}")
    
    if failed_downloads:
        print(f"❌ Download falliti: {len(failed_downloads)}")
        print("   Simboli falliti:", ", ".join(failed_downloads))
    
    # Conta totale coppie dopo download
    total_files = list(data_dir.glob("*_X.parquet"))
    print(f"📈 Totale coppie disponibili: {len(total_files)}")
    
    # Suggerimenti per retry
    if failed_downloads:
        print()
        print("💡 Per riprovare i download falliti, esegui nuovamente lo script")
        print("   oppure prova manualmente con simboli alternativi")
    
    print("\n🎉 Download completato!")

if __name__ == "__main__":
    main()