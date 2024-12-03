# %%
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import numpy as np

# %%
# Carregar os dados
df = pd.read_csv("data/feminicidios.csv")
df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')
df = df.sort_values(by='data')
print(df.head())

# Criar a série temporal
serie = df.groupby('data')['vitimas'].sum()

# %%
# Separar os dados em treino e teste
corte_data = '2023-01-01'
serie_treino = serie[serie.index < corte_data]
serie_teste = serie[serie.index >= corte_data]

# %%
# Ajustar o modelo automaticamente com auto_arima
arima_model = auto_arima(serie_treino, 
                         seasonal=False,  # Indica que é ARIMA, não SARIMA
                         stepwise=True,   # Busca passo-a-passo mais eficiente
                         suppress_warnings=True, 
                         trace=True,      # Exibe progresso
                         error_action='ignore', 
                         max_order=10)   # Limita busca para evitar excesso de combinações

# Exibir os parâmetros escolhidos
print(arima_model.summary())

# %%
# Fazer previsões no conjunto de teste
previsoes = arima_model.predict(n_periods=len(serie_teste))
print(previsoes)

# %%
# Calcular o RMSE
rmse = np.sqrt(mean_squared_error(serie_teste, previsoes))
print(f"RMSE: {rmse}")

# %%
# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(serie_treino, label='Treino')
plt.plot(serie_teste, label='Teste', color='green')
plt.plot(serie_teste.index, previsoes, label='Previsões', color='red', linestyle='--')
plt.legend()
plt.title("ARIMA Automático - Previsão de Feminicídios")
plt.show()
