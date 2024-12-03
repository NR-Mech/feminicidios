# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import timedelta


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
# Ajustar o modelo nos dados completos
modelo_completo = ARIMA(serie, order=(1, 1, 2))  # Use os mesmos parâmetros (p, d, q) encontrados anteriormente
resultado_completo = modelo_completo.fit()

# %%
# Prever de outubro deste ano até dezembro do próximo ano (15 meses)
passos_previsao = 15
previsao_futura = resultado_completo.get_forecast(steps=passos_previsao)

# %%
# Valores previstos
valores_previstos = previsao_futura.predicted_mean
# Intervalos de confiança
intervalo_confianca = previsao_futura.conf_int()

# %%
# Exibir os resultados
print("Previsões futuras:")
print(valores_previstos)
print("\nIntervalo de Confiança:")
print(intervalo_confianca)

# Plotar as previsões futuras
plt.figure(figsize=(10, 6))
plt.plot(serie, label='Histórico', color='blue')
plt.plot(valores_previstos.index, valores_previstos, label='Previsão Futura', color='red')
plt.fill_between(
    intervalo_confianca.index,
    intervalo_confianca.iloc[:, 0],
    intervalo_confianca.iloc[:, 1],
    color='pink', alpha=0.3, label="Intervalo de Confiança"
)
plt.legend()
plt.title("ARIMA - Previsão de Feminicídios (Outubro 2024 a Dezembro 2025)")
plt.xlabel("Tempo")
plt.ylabel("Vítimas")
plt.grid()
plt.show()
# %%
