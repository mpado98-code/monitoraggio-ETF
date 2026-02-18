import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE TELEGRAM ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# --- CONFIGURAZIONE TICKER PER CATEGORIA ---
TICKERS_CONFIG = {
    'INDICI': [
        '^GSPC', '^NDX', 'GC=F', 'SI=F', 'BTC-USD', '000001.SS',
        '^HSI', '^N225', '^GDAXI', '^FTSE', '^VIX'
    ],
    'VALUTE': [
        'JPY=X', 'EURUSD=X', 'GBPUSD=X', 'CNY=X', 'CHF=X', 'NOKUSD=X'
    ],
    'AZIONARIO': [
        'VWCE.MI', 'CSSPX.MI', 'CSNDX.MI', 'SWDA.MI', 'EIMI.MI'
    ],
    'AZIONARIO_GEO': [
        'XSX6.DE', 'SJPA.MI', 'XCS6.DE', 'XMBR.DE', 'XFVT.DE', 'XMIN.MI'
    ],
    'AZIONARIO_SET_US': [
        'SXLE.MI', 'SXLU.MI', 'SXLV.MI', 'SXLI.MI'
    ],
    'AZIONARIO_EW': [
        'MWEQ.MI', 'XDEW.MI'
    ],
    'OBBLIGAZIONI': [
        'CSBGU7.MI', 'CSBGE7.MI', 'SXRC.MU', 'X25E.MI'
    ],
    'ALTRO': [
        'SGLD.MI', 'PPFD.SG', 'CL=F'
    ]
}

PERIODI_GIORNI = {
    '1 Settimana': 7, '1 Mese': 30, '3 Mesi': 90,
    '6 Mesi': 180, '1 Anno': 365, '3 Anni': 1095, '5 Anni': 1825
}

# --- FUNZIONI ---
def get_freccia(rendimento):
    if rendimento is None or pd.isna(rendimento):
        return "⏸️"
    elif rendimento > 0:
        return "🟢 ↑"
    elif rendimento < 0:
        return "🔴 ↓"
    else:
        return "⚪ →"

def calcola_deviazione_std(storico, giorni=30):
    if len(storico) < giorni:
        return None
    rendimenti_giornalieri = storico['Close'].pct_change().dropna().tail(giorni)
    if len(rendimenti_giornalieri) == 0:
        return None
    std_annualizzata = rendimenti_giornalieri.std() * np.sqrt(252) * 100
    return round(std_annualizzata, 2)

def verifica_incrocio_medie_mobili(storico):
    if len(storico) < 200:
        return "Dati insufficienti"
    storico['MA50'] = storico['Close'].rolling(window=50).mean()
    storico['MA200'] = storico['Close'].rolling(window=200).mean()
    prezzo_ultimo = storico['Close'].iloc[-1]
    ma50_ultimo = storico['MA50'].iloc[-1]
    ma200_ultimo = storico['MA200'].iloc[-1]
    if pd.isna(ma50_ultimo) or pd.isna(ma200_ultimo):
        return "N/D"
    if prezzo_ultimo > ma50_ultimo and prezzo_ultimo > ma200_ultimo:
        return "🟢 SOPRA MA50/200"
    elif prezzo_ultimo > ma50_ultimo and prezzo_ultimo < ma200_ultimo:
        return "🟡 SOPRA MA50, SOTTO MA200"
    elif prezzo_ultimo < ma50_ultimo and prezzo_ultimo > ma200_ultimo:
        return "🟠 SOTTO MA50, SOPRA MA200"
    elif prezzo_ultimo < ma50_ultimo and prezzo_ultimo < ma200_ultimo:
        return "🔴 SOTTO MA50/200"
    else:
        return "⚪ MISTO"

def calcola_rendimenti(ticker, data_inizio, data_fine):
    risultati = {'Ticker': ticker}
    try:
        azione = yf.Ticker(ticker)
        storico = azione.history(start=data_inizio, end=data_fine)
        if storico.empty:
            return {**risultati, **{nome: None for nome in PERIODI_GIORNI.keys()}, 
                    'DevStd 30gg': None, 'MA50/200': 'N/D'}
        if storico.index.tz is not None:
            storico.index = storico.index.tz_localize(None)
        prezzo_attuale = storico['Close'].iloc[-1]
        data_ultima = storico.index[-1]
        for nome_periodo, giorni in PERIODI_GIORNI.items():
            data_target = data_ultima - timedelta(days=giorni)
            storico_periodo = storico[storico.index <= data_target]
            if not storico_periodo.empty:
                prezzo_passato = storico_periodo['Close'].iloc[-1]
                rendimento = ((prezzo_attuale - prezzo_passato) / prezzo_passato) * 100
                risultati[nome_periodo] = round(rendimento, 2)
            else:
                risultati[nome_periodo] = None
        risultati['DevStd 30gg'] = calcola_deviazione_std(storico)
        risultati['MA50/200'] = verifica_incrocio_medie_mobili(storico)
        for periodo in ['1 Settimana', '1 Mese', '3 Mesi']:
            if periodo in risultati:
                risultati[f'Freccia_{periodo}'] = get_freccia(risultati[periodo])
    except Exception as e:
        risultati = {**risultati, **{nome: None for nome in PERIODI_GIORNI.keys()},
                    'DevStd 30gg': None, 'MA50/200': 'N/D'}
        for periodo in ['1 Settimana', '1 Mese', '3 Mesi']:
            risultati[f'Freccia_{periodo}'] = "⚠️"
    return risultati

def formatta_categoria(categoria, df):
    """Formatta UNA SOLA categoria per Telegram"""
    if df.empty:
        return ""
    
    messaggio = f"<b>🏷️ {categoria}</b>\n─────────────────\n"
    
    for _, row in df.iterrows():
        ticker = row['Ticker']
        ma_status = row.get('MA50/200', 'N/D')
        devstd = row.get('DevStd 30gg', 'N/D')
        riga = f"<b>{ticker}</b> | {ma_status} | Vol: {devstd if devstd else 'N/D'}%\n"
        for periodo in ['1 Settimana', '1 Mese', '3 Mesi']:
            freccia = row.get(f'Freccia_{periodo}', '')
            valore = row.get(periodo, None)
            if valore is not None and not pd.isna(valore):
                riga += f"{periodo[:3]}: {freccia} {valore:+.2f}%  "
            else:
                riga += f"{periodo[:3]}: ⏸️ N/D  "
        messaggio += riga + "\n"
    
    return messaggio

def formatta_top_performer(df_completo):
    """Formatta solo la sezione TOP PERFORMER"""
    messaggio = "<b>🏆 TOP PERFORMER</b>\n─────────────────\n"
    
    for periodo in ['1 Settimana', '1 Mese', '3 Mesi']:
        if periodo in df_completo.columns:
            df_validi = df_completo[df_completo[periodo].notna()]
            if not df_validi.empty:
                top3 = df_validi.nlargest(3, periodo)[['Ticker', periodo]]
                messaggio += f"\n<b>{periodo}:</b>\n"
                for _, row in top3.iterrows():
                    freccia = get_freccia(row[periodo])
                    messaggio += f"  {freccia} {row['Ticker']}: {row[periodo]:+.2f}%\n"
    
    return messaggio

def invia_telegram(messaggio):
    """Invia un messaggio via Telegram"""
    if not messaggio.strip():
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': messaggio,
        'parse_mode': 'HTML'
    }
    response = requests.post(url, json=payload)
    print(f"Inviato: {len(messaggio)} caratteri - Status: {response.status_code}")
    return response

def main():
    print("🚀 Avvio monitoraggio...")
    
    data_fine = datetime.now()
    data_inizio = data_fine - timedelta(days=5*365 + 100)
    reports_dict = {}
    
    # Raccogli dati
    for categoria, tickers in TICKERS_CONFIG.items():
        if tickers:
            print(f"📊 Analizzo {categoria}...")
            risultati_categoria = []
            for ticker in tickers:
                risultati = calcola_rendimenti(ticker, data_inizio, data_fine)
                risultati_categoria.append(risultati)
            df_categoria = pd.DataFrame(risultati_categoria)
            if not df_categoria.empty:
                reports_dict[categoria] = df_categoria
    
    if not reports_dict:
        print("❌ Nessun report generato")
        return
    
    # Invia intestazione
    data_str = data_fine.strftime('%d/%m/%Y %H:%M')
    invia_telegram(f"<b>📈 REPORT MERCATI - {data_str}</b>\n\n(Invio in più parti...)")
    
    # Invia una categoria alla volta
    for categoria, df in reports_dict.items():
        msg_categoria = formatta_categoria(categoria, df)
        invia_telegram(msg_categoria)
    
    # Invia top performer
    df_completo = pd.concat(reports_dict.values(), ignore_index=True)
    msg_top = formatta_top_performer(df_completo)
    invia_telegram(msg_top)
    
    print("✅ Report completato e inviato!")

if __name__ == "__main__":
    main()
