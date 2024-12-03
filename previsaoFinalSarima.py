# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %%

df = pd.read_csv("data/feminicidios.csv")
df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')
df = df.sort_values(by='data')

# Carregar os dados
serie = df.groupby('data')['vitimas'].sum()

# %%
# Ajustar o modelo SARIMA com toda a base de dados
s = 12  # Sazonalidade mensal
model = SARIMAX(serie, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, s), 
                enforce_stationarity=False, 
                enforce_invertibility=False)

sarima_model = model.fit(disp=False)

# %%
# Exibir resumo do modelo
print(sarima_model.summary())

# %%
# Fazer previsões para 15 meses à frente (Outubro de 2024 a Dezembro de 2025)
steps_ahead = 15
forecast = sarima_model.get_forecast(steps=steps_ahead)
previsoes = forecast.predicted_mean
intervalo_confianca = forecast.conf_int()

# %%
# Criar índice de datas para previsões
ultimo_mes = serie.index[-1]
datas_previsoes = pd.date_range(start=ultimo_mes + pd.offsets.MonthBegin(1), periods=steps_ahead, freq='MS')

# Adicionar as previsões ao DataFrame
previsoes.index = datas_previsoes
intervalo_confianca.index = datas_previsoes

# %%
# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.plot(serie, label='Histórico')
plt.plot(previsoes, label='Previsão (Out/2024 - Dez/2025)', linestyle='--', color='red')
plt.fill_between(intervalo_confianca.index,
                 intervalo_confianca.iloc[:, 0],
                 intervalo_confianca.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo de Confiança')
plt.axvline(ultimo_mes, color='black', linestyle='--', alpha=0.7, label='Início da Previsão')
plt.legend()
plt.title('Previsão SARIMA - Feminicídios')
plt.xlabel('Data')
plt.ylabel('Número de Vítimas')
plt.grid()
plt.show()

# %%
# Exibir previsões e intervalo de confiança
print("Previsões futuras:")
print(previsoes)
print("\nIntervalos de confiança:")
print(intervalo_confianca)

# %%
