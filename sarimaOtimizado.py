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
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Definir os parâmetros do modelo
p, d, q = 1, 1, 2
P, D, Q, s = 1, 0, 1, 12  # Parâmetros sazonais

# Listas para armazenar os erros
rmses = []
maes = []

# Número de divisões na validação cruzada
n_splits = 5
step = len(serie) // n_splits

# Validação cruzada com janela expansiva
for i in range(step, len(treino) + 1, step):
    treino_cv = treino.iloc[:i]
    teste_cv = treino.iloc[i:i + step]  # Próxima janela para validação

    # Ajustar o modelo SARIMA nos dados de treino
    model = SARIMAX(
        treino_cv, 
        order=(p, d, q), 
        seasonal_order=(P, D, Q, s), 
        enforce_stationarity=False, 
        enforce_invertibility=False
    )
    result = model.fit(disp=False)

    # Fazer previsões
    forecast = result.get_forecast(steps=len(teste_cv))
    previsoes = forecast.predicted_mean

    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(teste_cv, previsoes))
    mae = mean_absolute_error(teste_cv, previsoes)

    # Armazenar os erros
    rmses.append(rmse)
    maes.append(mae)

    print(f"Divisão {i // step}: RMSE = {rmse}, MAE = {mae}")

# Exibir os resultados médios
print(f"Média RMSE: {np.mean(rmses)}")
print(f"Média MAE: {np.mean(maes)}")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rmses) + 1), rmses, label="RMSE", marker='o')
plt.plot(range(1, len(maes) + 1), maes, label="MAE", marker='x')
plt.xlabel("Divisão")
plt.ylabel("Erro")
plt.title("Validação Cruzada - RMSE e MAE por Divisão")
plt.legend()
plt.show()

# %%
