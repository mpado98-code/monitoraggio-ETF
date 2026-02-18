import requests
import os

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def invia_telegram(messaggio):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': messaggio,
        'parse_mode': 'HTML'
    }
    response = requests.post(url, json=payload)
    print(f"Status code: {response.status_code}")
    print(f"Risposta: {response.text}")
    return response

def main():
    messaggio = "<b>🧪 TEST</b>\n\nSe ricevi questo messaggio, il bot funziona!"
    invia_telegram(messaggio)
    print("Test completato")

if __name__ == "__main__":
    main()
