# Importa bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input, Activation
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam

# Baixa dados de ações da internet
data = yf.download(tickers='PETR4.SA', start='2012-03-11', end='2024-08-29')
print(data)

# Calcula índices técnicos
data['RSI'] = ta.rsi(data.Close, length=15)  # Índice de Força Relativa
data['EMAF'] = ta.ema(data.Close, length=20)  # Média Móvel Exponencial
data['EMAM'] = ta.ema(data.Close, length=100)  # Média Móvel Exponencial
data['EMAS'] = ta.ema(data.Close, length=150)  # Média Móvel Exponencial

# Remove linhas com valores faltantes
data.dropna(inplace=True)

# Seleciona as colunas necessárias
data_set = data[['Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']]

# Escala os dados
sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(data_set)

# Prepara os dados para o modelo LSTM
X = []
y = []
backcandles = 30

for i in range(backcandles, len(data_set_scaled)):
    X.append(data_set_scaled[i-backcandles:i, :])  # Dados de entrada
    y.append(data_set_scaled[i, 0])  # Valor de fechamento correspondente

X, y = np.array(X), np.array(y)

# Separa os dados de treinamento e teste
splitlimit = int(len(X) * 0.8)  # 80% para treinamento
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

# Cria o modelo
model = Sequential()
model.add(LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=150, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))

# Compila o modelo
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Treina o modelo
model.fit(X_train, y_train, epochs=30, batch_size=15, validation_split=0.1, shuffle=True)

# Faz previsões
y_pred = model.predict(X_test)

# Desfaz a escala das previsões
y_pred = sc.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], 4))), axis=1))[:,0]
y_test = sc.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1))[:,0]

# Obtém as datas correspondentes às previsões
dates = data.index[splitlimit + backcandles:]

# Plota as previsões e os valores reais com as datas no eixo x
plt.figure(figsize=(16, 8))
plt.plot(dates, y_test, color='black', label='Valor Real')
plt.plot(dates, y_pred, color='green', label='Previsão')
plt.title('Previsão de Fechamento de Ação')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.legend()
plt.show()

# Cria um DataFrame com as previsões e os valores reais
df_result = pd.DataFrame({
    'Data': dates,
    'Valor Real': y_test,
    'Valor Previsto': y_pred
})

# Define a coluna 'Data' como índice
df_result.set_index('Data', inplace=True)

# Exibe o DataFrame resultante
print(df_result)

# Opcional: Salva o DataFrame em um arquivo CSV
df_result.to_excel('previsoes_acoes.xlsx')

