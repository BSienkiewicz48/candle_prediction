#Kliknij play żeby połączyć się z serwerem XTB
import websocket
import json
from datetime import datetime
import pandas as pd
import streamlit as st
import joblib

# Dane do logowania
login_data = {
    "command": "login",
    "arguments": {
        "userId": st.secrets["userId"],  # Zastąp własnym ID
        "password": st.secrets["password"]  # Zastąp własnym hasłem
    }
}
login_json = json.dumps(login_data)

# Tworzenie połączenia WebSocket
ws = websocket.create_connection("wss://ws.xtb.com/demo")  # Użycie create_connection

# Wysyłanie danych logowania
ws.send(login_json)

# Odbieranie odpowiedzi
response = ws.recv()
result = json.loads(response)
status = result.get("status")

# Obsługa odpowiedzi
if str(status) == "True":
    print("Logowanie zakończone sukcesem! 🎉")
else:
    print("Błąd logowania.")

# Ustalanie czasu rozpoczęcia 
two_days_ago = pd.Timestamp.now() - pd.Timedelta(days=0.25)
start_timestamp = int(two_days_ago.timestamp() * 1000)  # Przekształcenie na milisekundy

# Dane do zapytania o wykres
chart_last_request = {
    "command": "getChartLastRequest",
    "arguments": {
        "info": {
            "period": 15,  # jaka świeczka warjacie
            "start": start_timestamp,
            "symbol": "USDPLN"  # Zmień symbol, jeśli potrzebujesz innego
        }
    }
}
request_json = json.dumps(chart_last_request)

# Wysyłanie danych zapytania o wykres
ws.send(request_json)

# Odbieranie odpowiedzi
response_chart = ws.recv()
result_chart = json.loads(response_chart)

# Obsługa odpowiedzi
if result_chart.get("status"):
    return_data = result_chart.get("returnData", {})
    digits = return_data.get("digits", 0)
    rate_infos = return_data.get("rateInfos", [])
    
    # Tworzenie DataFrame z listy danych
    data = []
    for rate in rate_infos:
        data.append({
            "Open": rate['open'] / (10 ** digits),
            "Close": rate['close'] / (10 ** digits),
            "High": rate['high'] / (10 ** digits),
            "Low": rate['low'] / (10 ** digits),
            "Volume": rate['vol'],
            "Time": rate['ctmString']
        })
    
    df = pd.DataFrame(data)
    print("Pobrano dane do df")  # Wyświetlenie danych DataFrame
else:
    print("Błąd pobierania danych wykresu.")

df['Time'] = pd.to_datetime(df['Time'])
df['Close_Price'] = df['Close'] + df['Open']
df['max_price'] = df['Open'] + df['High']  
df['min_price'] = df['Open'] + df['Low']  
df['PriceChange'] = df['Close_Price'] - df['Open']  
df['Volatility'] = df['max_price'] - df['min_price']

# Zaokrąglanie kolumn do 4 miejsc po przecinku
df[['Open', 'Close', 'High', 'Low', 'Close_Price', 'max_price', 'min_price', 'PriceChange', 'Volatility']] = df[['Open', 'Close', 'High', 'Low', 'Close_Price', 'max_price', 'min_price', 'PriceChange', 'Volatility']].round(4)

symbol = chart_last_request['arguments']['info']['symbol']

# Dodanie symbolu do nazw kolumn
df = df.rename(columns={
    'Open': f'Open_{symbol}',
    'Close': f'Close_{symbol}',
    'High': f'High_{symbol}',
    'Low': f'Low_{symbol}',
    'Volume': f'Volume_{symbol}',
    'Close_Price': f'Close_Price_{symbol}',
    'max_price': f'max_price_{symbol}',
    'min_price': f'min_price_{symbol}',
    'PriceChange': f'PriceChange_{symbol}',
    'Volatility': f'Volatility_{symbol}'
})


#DANE DRUGIE

# Dane do zapytania o wykres
chart_last_request = {
    "command": "getChartLastRequest",
    "arguments": {
        "info": {
            "period": 15,  # jaka świeczka warjacie
            "start": start_timestamp,
            "symbol": "EURPLN"  # Zmień symbol, jeśli potrzebujesz innego
        }
    }
}
request_json = json.dumps(chart_last_request)

# Wysyłanie danych zapytania o wykres
ws.send(request_json)

# Odbieranie odpowiedzi
response_chart = ws.recv()
result_chart = json.loads(response_chart)

# Obsługa odpowiedzi
if result_chart.get("status"):
    return_data = result_chart.get("returnData", {})
    digits = return_data.get("digits", 0)
    rate_infos = return_data.get("rateInfos", [])
    
    # Tworzenie DataFrame z listy danych
    data = []
    for rate in rate_infos:
        data.append({
            "Open": rate['open'] / (10 ** digits),
            "Close": rate['close'] / (10 ** digits),
            "High": rate['high'] / (10 ** digits),
            "Low": rate['low'] / (10 ** digits),
            "Volume": rate['vol'],
            "Time": rate['ctmString']
        })
    
    df_1 = pd.DataFrame(data)
    print("Pobrano dane do df_1")  # Wyświetlenie danych DataFrame
else:
    print("Błąd pobierania danych wykresu.")

df_1['Time'] = pd.to_datetime(df_1['Time'])
df_1['Close_Price'] = df_1['Close'] + df_1['Open']
df_1['max_price'] = df_1['Open'] + df_1['High']  
df_1['min_price'] = df_1['Open'] + df_1['Low']  
df_1['PriceChange'] = df_1['Close_Price'] - df_1['Open']  
df_1['Volatility'] = df_1['max_price'] - df_1['min_price']

# Zaokrąglanie kolumn do 4 miejsc po przecinku
df_1[['Open', 'Close', 'High', 'Low', 'Close_Price', 'max_price', 'min_price', 'PriceChange', 'Volatility']] = df_1[['Open', 'Close', 'High', 'Low', 'Close_Price', 'max_price', 'min_price', 'PriceChange', 'Volatility']].round(4)

symbol1 = chart_last_request['arguments']['info']['symbol']

# Dodanie symbolu do nazw kolumn
df_1 = df_1.rename(columns={
    'Open': f'Open_{symbol1}',
    'Close': f'Close_{symbol1}',
    'High': f'High_{symbol1}',
    'Low': f'Low_{symbol1}',
    'Volume': f'Volume_{symbol1}',
    'Close_Price': f'Close_Price_{symbol1}',
    'max_price': f'max_price_{symbol1}',
    'min_price': f'min_price_{symbol1}',
    'PriceChange': f'PriceChange_{symbol1}',
    'Volatility': f'Volatility_{symbol1}'
})


# Sprawdzenie, który DataFrame ma mniej danych
if len(df) < len(df_1):
    smaller_df = df
    larger_df = df_1
else:
    smaller_df = df_1
    larger_df = df

# Dołączenie kolumn z większego DataFrame do mniejszego po kolumnie Time
prediction_df = pd.merge(smaller_df, larger_df, on='Time', how='left')
# Przesunięcie kolumny 'Time' na pierwszą pozycję
cols = prediction_df.columns.tolist()
cols.insert(0, cols.pop(cols.index('Time')))
prediction_df = prediction_df[cols]

# Usunięcie kolumn Close_XXXXXX, High_XXXXXX, Low_XXXXXX, PriceChange_XXXXXX
columns_to_drop = [col for col in prediction_df.columns if col.startswith(f'Close_{symbol1}') or col.startswith(f'High_{symbol1}') or col.startswith(f'Low_{symbol1}') or col.startswith(f'PriceChange_{symbol1}') or col.startswith(f'Close_{symbol}') or col.startswith(f'High_{symbol}') or col.startswith(f'Low_{symbol}') or col.startswith(f'PriceChange_{symbol}')]
prediction_df.drop(columns=columns_to_drop, inplace=True)


# Pobranie najnowszego wiersza z merged_df
latest_data = prediction_df.sort_values(by='Time', ascending=False).iloc[0]

# Przygotowanie danych wejściowych dla modeli
# Usuń jeśli isnieją, ale jeśli nie istnieją to nic nie rób
columns_to_drop = ['Time', 'Close_Price_USDPLN_Lag1', 'Open_Price_USDPLN_Lag1', 'Max_Price_USDPLN_Lag1', 'Min_Price_USDPLN_Lag1']
X_latest = latest_data.drop(labels=[col for col in columns_to_drop if col in latest_data]).values.reshape(1, -1)

# Wczytanie modeli
try:
    model_xgboost_close = joblib.load('model_xgboost_close.pkl')
    model_xgboost_Open = joblib.load('model_xgboost_Open.pkl')
    model_xgboost_Max = joblib.load('model_xgboost_Max.pkl')
    model_xgboost_Min = joblib.load('model_xgboost_Min.pkl')
except FileNotFoundError as e:
    st.error(f"Model file not found: {e.filename}")
    st.stop()

# Prognozowanie wartości za pomocą modeli
pred_close_price = model_xgboost_close.predict(X_latest)
pred_open_price = model_xgboost_Open.predict(X_latest)
pred_max_price = model_xgboost_Max.predict(X_latest)
pred_min_price = model_xgboost_Min.predict(X_latest)

# Tworzenie interfejsu Streamlit
st.title("Wykres świecowy USDPLN z prognozą")

import plotly.graph_objects as go

# Tworzenie wykresu świecowego
fig = go.Figure(data=[go.Candlestick(
    x=prediction_df['Time'],
    open=prediction_df[f'Open_{symbol}'],
    high=prediction_df[f'max_price_{symbol}'],
    low=prediction_df[f'min_price_{symbol}'],
    close=prediction_df[f'Close_Price_{symbol}'],
    name=f'{symbol} Actual'
)])

# Dodanie danych prognozowanych jako kolejna świeczka
fig.add_trace(go.Candlestick(
    x=[prediction_df['Time'].iloc[-1] + pd.Timedelta(minutes=15)],
    open=[pred_open_price[0]],
    high=[pred_max_price[0]],
    low=[pred_min_price[0]],
    close=[pred_close_price[0]],
    name='Predicted',
    increasing_fillcolor='rgba(0, 0, 255, 0.5)',  
    decreasing_fillcolor='rgba(0, 0, 255, 0.5)'   
))

# Ustawienia wykresu
fig.update_layout(
    xaxis_title='Czas',
    yaxis_title='Cena',
    xaxis_rangeslider_visible=False
)

# Wyświetlenie wykresu w Streamlit
st.plotly_chart(fig)