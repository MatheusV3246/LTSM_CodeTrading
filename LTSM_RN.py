# Importa bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from VECM import VECM_Model

# Carrega o arquivo Excel e seleciona as colunas necessárias
data = pd.read_excel("dados.xlsx")
data.set_index('Ano-Mês', inplace=True)
data_set = data[["Crédito", "CDI a.m", "IPCA a.m"]]

# Escala os dados
sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(data_set)

# Prepara os dados para o modelo LSTM
X = []
y = []
backcandles = 30

for i in range(backcandles, len(data_set_scaled)):
    X.append(data_set_scaled[i-backcandles:i, :])  # Dados de entrada
    y.append(data_set_scaled[i, 0])  # Valor de 'Crédito' correspondente

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

# Obtenha o número de features
num_features = data_set.shape[1]  # Deve ser 3 no seu caso

# Desfaz a escala das previsões
y_pred_extended = np.concatenate((y_pred, np.zeros((y_pred.shape[0], num_features - 1))), axis=1)
y_pred = sc.inverse_transform(y_pred_extended)[:, 0]

y_test_extended = np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], num_features - 1))), axis=1)
y_test = sc.inverse_transform(y_test_extended)[:, 0]

# Obtém as datas correspondentes às previsões
dates = data.index[splitlimit + backcandles:]

# Plota as previsões e os valores reais com as datas no eixo x
plt.figure(figsize=(16, 8))
plt.plot(dates, y_test, color='black', label='Valor Real')
plt.plot(dates, y_pred, color='green', label='Previsão')
plt.title('Previsão de Crédito')
plt.xlabel('Data')
plt.ylabel('Crédito')
plt.legend()
plt.show()

cdi = data_set['CDI a.m']

# Cria um DataFrame com as previsões e os valores reais
df_result = pd.DataFrame({
    'Data': dates,
    'Crédito': y_test,
    'Crédito Previsto': y_pred,
    'CDI a.m': cdi
})

# Define a coluna 'Data' como índice
df_result.set_index('Data', inplace=True)

df_result_set = df_result[["Crédito Previsto", "CDI a.m"]]

modelo_vecm = VECM_Model(df_result_set, target="Crédito Previsto", index=0, diff=3, coint=1, deterministic="li")
modelo_vecm.fit_model()
dados_finais = modelo_vecm.predict_model(pfrente=5)

dados_finais.to_excel('previsoes_crédito.xlsx')