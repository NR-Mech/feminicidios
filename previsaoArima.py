# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# %%

df = pd.read_csv("data/feminicidios.csv")
df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')
df = df.sort_values(by='data')
print(df.head())

# %%
serie = df.groupby('data')['vitimas'].sum()

print(serie.head())
print(serie.shape[0])
# %%
from statsmodels.tsa.stattools import adfuller

# Teste ADF para verificar estacionaridade
resultado = adfuller(serie)
print(f"ADF Statistic: {resultado[0]}")
print(f"p-value: {resultado[1]}")

if resultado[1] < 0.05:
    print("A série é estacionária")
else:
    print("A série NÃO é estacionária")
# %%
# Diferenciação
serie_diff = serie.diff().dropna()

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ACF e PACF para a série diferenciada
plot_acf(serie_diff, lags=20)
plot_pacf(serie_diff, lags=20)
plt.show()
# %%
# Filtrar os dados de treino (2015-2022) e teste (2023-2024)
serie_treino = serie[serie.index < '2023-01-01']
serie_teste = serie[serie.index >= '2023-01-01']

# Exibir os tamanhos dos conjuntos
print("Treino:")
print(serie_treino)
print("\nTeste:")
print(serie_teste)

# %%
from statsmodels.tsa.arima.model import ARIMA

# Ajustar o modelo nos dados de treino
p, d, q = 1, 1, 2  # Escolha os valores apropriados após análise de ACF/PACF
modelo = ARIMA(serie_treino, order=(p, d, q))
resultado = modelo.fit()

# Exibir o resumo do modelo
print(resultado.summary())

# %%
# Previsão para o mesmo intervalo do conjunto de teste
previsoes = resultado.get_forecast(steps=len(serie_teste))
previsoes_ic = previsoes.conf_int()  # Intervalo de confiança
previsoes_valores = previsoes.predicted_mean  # Valores previstos

# Exibir as previsões
print(previsoes_valores)
# %%
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Calcular o erro quadrático médio (MSE)
mse = mean_squared_error(serie_teste, previsoes_valores)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(serie_treino, label='Treino')
plt.plot(serie_teste, label='Teste', color='green')
plt.plot(previsoes_valores, label='Previsões', color='red')
plt.fill_between(
    previsoes_ic.index,
    previsoes_ic.iloc[:, 0],
    previsoes_ic.iloc[:, 1],
    color='pink', alpha=0.3, label="Intervalo de Confiança"
)
plt.legend()
plt.title("ARIMA - Previsão de Feminicídios")
plt.show()

# %%
serie_teste.shape
# %%
serie_treino.shape
# %%
