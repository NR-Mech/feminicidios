# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

# %%

df = pd.read_csv("data/feminicidios.csv")
df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')
df = df.sort_values(by='data')
print(df.head())

# %%
# Carregar os dados
serie = df.groupby('data')['vitimas'].sum()

# %%
# Separar os dados em treino e teste
corte_data = '2023-01-01'
treino = serie[serie.index < corte_data]
teste = serie[serie.index >= corte_data]

# %%
# Ajustar o modelo SARIMA
# Definindo os parâmetros do modelo (substitua por parâmetros ajustados para seu caso)
s = 12  # Sazonalidade mensal
model = SARIMAX(treino, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, s), 
                enforce_stationarity=False, 
                enforce_invertibility=False)

sarima_model = model.fit(disp=False)

# %%
# Exibir resumo do modelo
print(sarima_model.summary())

# %%
# Fazer previsões no conjunto de teste
forecast = sarima_model.get_forecast(steps=len(teste))
previsoes = forecast.predicted_mean
intervalo_confianca = forecast.conf_int()

# %%
# Avaliar o modelo
rmse = np.sqrt(mean_squared_error(teste, previsoes))
print(f"RMSE: {rmse}")

# %%
# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(treino, label='Treino')
plt.plot(teste, label='Teste')
plt.plot(previsoes, label='Previsões', linestyle='--')
plt.fill_between(intervalo_confianca.index,
                 intervalo_confianca.iloc[:, 0],
                 intervalo_confianca.iloc[:, 1], 
                 color='pink', alpha=0.3, label='Intervalo de Confiança')
plt.legend()
plt.show()

# %%
