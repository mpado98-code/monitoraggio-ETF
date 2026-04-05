import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import warnings
import google.generativeai as genai
warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE TELEGRAM ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# --- CONFIGURAZIONE GEMINI AI ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- DIZIONARIO PER I NOMI LEGGIBILI ---
NOMI_LEGGIBILI = {
    # INDICI
    '^GSPC': 'S&P500',
    '^NDX': 'Nasdaq',
    'GC=F': 'Oro',
    'SI=F': 'Argento',
    'BTC-USD': 'Bitcoin',
    '000001.SS': 'Shanghai',
    '^HSI': 'Hang Seng',
    '^N225': 'Nikkei',
    '^GDAXI': 'DAX',
    '^FTSE': 'FTSE100',
    '^VIX': 'VIX',
    
    # VALUTE (con nuovi ticker)
    'JPYUSD=X': 'Yen$',
    'EURUSD=X': 'Euro$',
    'GBPUSD=X': 'GBP$',
    'CNYUSD=X': 'Yuan$',
    'CHFUSD=X': 'Franco$',
    'NOKUSD=X': 'NOK$',
    
    # AZIONARIO
    'VWCE.MI': 'ALL WORLD',
    'CSSPX.MI': 'SP500€',
    'CSNDX.MI': 'NASDAQ€',
    'SWDA.MI': 'SVILUPPATI',
    'EIMI.MI': 'EMERGENTI',
    
    # AZIONARIO GEO
    'XSX6.DE': 'Euro Stoxx',
    'SJPA.MI': 'Giappone',
    'XCS6.DE': 'Cina',
    'XMBR.DE': 'Brasile',
    'XFVT.DE': 'Vietnam',
    'XMIN.MI': 'Indonesia',
    
    # AZIONARIO SETTORIALE USA
    'SXLE.MI': 'Energia',
    'SXLU.MI': 'Utilities',
    'SXLV.MI': 'Sanità',
    'SXLI.MI': 'Industriali',
    'SXLK.MI': 'Tech',
    
    # AZIONARIO EW
    'MWEQ.MI': 'ALL WORLD EW',
    'XDEW.MI': 'SP500 EW',
    
    # OBBLIGAZIONI
    'CSBGU7.MI': 'Treasury 7-10',
    'CSBGE7.MI': 'Euro Govt 7-10',
    'SXRC.MU': 'Treasury 20+',
    'X25E.MI': 'Euro Govt 25+',
    
    # ALTRO
    'SGLD.MI': 'Gold ETC',
    'PPFD.SG': 'Silver ETC',
    'CL=F': 'Petrolio'
}

# --- CONFIGURAZIONE TICKER PER CATEGORIA ---
TICKERS_CONFIG = {
    'INDICI': [
        '^GSPC', '^NDX', 'GC=F', 'SI=F', 'BTC-USD', '000001.SS',
        '^HSI', '^N225', '^GDAXI', '^FTSE', '^VIX'
    ],
    'VALUTE': [
        'JPYUSD=X', 'EURUSD=X', 'GBPUSD=X', 'CNYUSD=X', 'CHFUSD=X', 'NOKUSD=X'
    ],
    'AZIONARIO': [
        'VWCE.MI', 'CSSPX.MI', 'CSNDX.MI', 'SWDA.MI', 'EIMI.MI'
    ],
    'AZIONARIO_GEO': [
        'XSX6.DE', 'SJPA.MI', 'XCS6.DE', 'XMBR.DE', 'XFVT.DE', 'XMIN.MI'
    ],
    'AZIONARIO_SET_US': [
        'SXLE.MI', 'SXLU.MI', 'SXLV.MI', 'SXLI.MI', 'SXLK.MI'
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
    '1 Settimana': 7, 
    '1 Mese': 30, 
    '3 Mesi': 90,
    '6 Mesi': 180, 
    '1 Anno': 365, 
    '3 Anni': 1095, 
    '5 Anni': 1825
}

# --- FUNZIONI ---
def get_nome_leggibile(ticker):
    return NOMI_LEGGIBILI.get(ticker, ticker)

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
    risultati = {'Ticker': ticker, 'Nome': get_nome_leggibile(ticker)}
    
    try:
        azione = yf.Ticker(ticker)
        storico = azione.history(start=data_inizio, end=data_fine)
        
        if storico.empty:
            print(f"⚠️ Nessun dato per {ticker}")
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
        
        for periodo in ['1 Settimana', '1 Mese', '3 Mesi', '6 Mesi', '1 Anno']:
            if periodo in risultati:
                risultati[f'Freccia_{periodo}'] = get_freccia(risultati[periodo])
        
        print(f"✅ {ticker} -> {risultati['Nome']}: OK")
                
    except Exception as e:
        print(f"❌ Errore con {ticker}: {str(e)[:50]}...")
        risultati = {**risultati, **{nome: None for nome in PERIODI_GIORNI.keys()},
                    'DevStd 30gg': None, 'MA50/200': 'N/D'}
        for periodo in ['1 Settimana', '1 Mese', '3 Mesi', '6 Mesi', '1 Anno']:
            risultati[f'Freccia_{periodo}'] = "⚠️"
    
    return risultati

def genera_riassunto_ai(df_completo):
    """Genera un riassunto strategico usando Google Gemini AI"""
    
    # Prepara i dati principali per il prompt
    top_1_settimana = df_completo.nlargest(5, '1 Settimana')[['Nome', '1 Settimana', 'MA50/200']].to_string(index=False)
    top_1_mese = df_completo.nlargest(5, '1 Mese')[['Nome', '1 Mese', 'MA50/200']].to_string(index=False)
    top_3_mesi = df_completo.nlargest(5, '3 Mesi')[['Nome', '3 Mesi', 'MA50/200']].to_string(index=False)
    top_6_mesi = df_completo.nlargest(5, '6 Mesi')[['Nome', '6 Mesi', 'MA50/200']].to_string(index=False)
    top_1_anno = df_completo.nlargest(5, '1 Anno')[['Nome', '1 Anno', 'MA50/200']].to_string(index=False)
    
    # Asset con trend forte (sopra medie mobili)
    trend_forte = df_completo[df_completo['MA50/200'].str.contains('SOPRA', na=False)][['Nome', 'MA50/200', '1 Mese', '3 Mesi']].head(5).to_string(index=False)
    
    prompt = f"""Sei un analista finanziario esperto. Basandoti sui dati seguenti, rispondi a queste domande:

DATI PERFORMANCE (rendimenti % e trend):
TOP 5 ULTIMA SETTIMANA (1S):
{top_1_settimana}

TOP 5 ULTIMO MESE (1M):
{top_1_mese}

TOP 5 ULTIMI 3 MESI (3M):
{top_3_mesi}

TOP 5 ULTIMI 6 MESI (6M):
{top_6_mesi}

TOP 5 ULTIMO ANNO (1A):
{top_1_anno}

ASSET CON TREND PIÙ FORTE (sopra medie mobili):
{trend_forte}

RICHIESTA:
Fammi un riassunto di massimo 8 righe (medio) in italiano che risponda a:

1. DOVE VANNO I FLUSSI? - Quali asset/mercati stanno raccogliendo più capitali (quelli con rendimenti migliori e trend positivi)

2. BREVE TERMINE (1-6 mesi) - Quale potrebbe essere l'opzione di acquisto migliore? Perché?

3. MEDIO TERMINE (6-12 mesi) - Quale asset ha le caratteristiche migliori per un orizzonte medio? Perché?

4. LUNGO TERMINE (>3 anni) - Quale mercato/ETF consiglieresti per una strategia di lungo periodo? Perché?

Usa un tono professionale ma chiaro. Evita elenchi puntati, usa frasi fluide. Non superare le 8 righe totali."""
    
    try:
        response = model.generate_content(prompt)
        riassunto = response.text.strip()
        print("✅ Riassunto AI generato con successo")
        return riassunto
    except Exception as e:
        print(f"❌ Errore generazione AI: {e}")
        return "🤖 Servizio AI temporaneamente non disponibile. Report standard in arrivo."

def formatta_categoria(categoria, df):
    if df.empty:
        return ""
    
    messaggio = f"<b>🏷️ {categoria}</b>\n─────────────────\n"
    
    for _, row in df.iterrows():
        nome = row['Nome']
        ma_status = row.get('MA50/200', 'N/D')
        devstd = row.get('DevStd 30gg', 'N/D')
        
        riga = f"<b>{nome}</b> | {ma_status} | Vol: {devstd if devstd else 'N/D'}%\n"
        
        for periodo in ['1 Settimana', '1 Mese', '3 Mesi', '6 Mesi', '1 Anno']:
            freccia = row.get(f'Freccia_{periodo}', '')
            valore = row.get(periodo, None)
            if periodo == '1 Settimana':
                abbr = '1S'
            elif periodo == '1 Mese':
                abbr = '1M'
            elif periodo == '3 Mesi':
                abbr = '3M'
            elif periodo == '6 Mesi':
                abbr = '6M'
            elif periodo == '1 Anno':
                abbr = '1A'
            else:
                abbr = periodo[:3]
                
            if valore is not None and not pd.isna(valore):
                riga += f"{abbr}: {freccia} {valore:+.2f}%  "
            else:
                riga += f"{abbr}: ⏸️ N/D  "
        
        messaggio += riga + "\n"
    
    return messaggio

def formatta_top_performer(df_completo):
    messaggio = "<b>🏆 TOP PERFORMER</b>\n─────────────────\n"
    
    for periodo in ['1 Settimana', '1 Mese', '3 Mesi', '6 Mesi', '1 Anno']:
        if periodo in df_completo.columns:
            df_validi = df_completo[df_completo[periodo].notna()]
            if not df_validi.empty:
                top3 = df_validi.nlargest(3, periodo)[['Nome', periodo]]
                if periodo == '1 Settimana':
                    abbr = '1S'
                elif periodo == '1 Mese':
                    abbr = '1M'
                elif periodo == '3 Mesi':
                    abbr = '3M'
                elif periodo == '6 Mesi':
                    abbr = '6M'
                elif periodo == '1 Anno':
                    abbr = '1A'
                else:
                    abbr = periodo
                    
                messaggio += f"\n<b>{abbr}:</b>\n"
                for _, row in top3.iterrows():
                    freccia = get_freccia(row[periodo])
                    messaggio += f"  {freccia} {row['Nome']}: {row[periodo]:+.2f}%\n"
    
    return messaggio

def invia_telegram(messaggio):
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
    
    # Unisci tutti i dati per l'AI
    df_completo = pd.concat(reports_dict.values(), ignore_index=True)
    
    # Genera riassunto AI
    print("🤖 Generazione riassunto con Gemini AI...")
    riassunto_ai = genera_riassunto_ai(df_completo)
    
    # Invia riassunto AI su Telegram
    data_str = data_fine.strftime('%d/%m/%Y %H:%M')
    invia_telegram(f"<b>📈 REPORT MERCATI - {data_str}</b>\n\n<b>🤖 ANALISI AI:</b>\n{riassunto_ai}")
    
    # Invia report completo (dettagli)
    invia_telegram(f"\n<b>📊 DATI DETTAGLIATI</b>")
    
    for categoria, df in reports_dict.items():
        msg_categoria = formatta_categoria(categoria, df)
        invia_telegram(msg_categoria)
    
    msg_top = formatta_top_performer(df_completo)
    invia_telegram(msg_top)
    
    print("✅ Report completo inviato!")

if __name__ == "__main__":
    main()
