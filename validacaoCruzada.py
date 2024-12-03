# %%
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("data/feminicidios.csv")
df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')
df = df.sort_values(by='data')
print(df.head())

# %%
# Função para validação cruzada em séries temporais
def cross_validation_arima(data, order, n_splits):
    """
    Realiza validação cruzada para modelos ARIMA e calcula RMSE e MAE.
    
    Args:
        data (pd.Series): Série temporal.
        order (tuple): Parâmetros do modelo ARIMA (p, d, q).
        n_splits (int): Número de divisões para validação cruzada.
    
    Returns:
        tuple: Listas de RMSEs e MAEs para cada divisão.
    """
    rmse_list = []
    mae_list = []
    n = len(data)
    split_size = n // (n_splits + 1)

    for i in range(1, n_splits + 1):
        # Dividir em treino e teste
        train = data[:split_size * i]
        test = data[split_size * i : split_size * (i + 1)]
        
        # Ajustar o modelo ARIMA
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        
        # Fazer previsões
        predictions = model_fit.forecast(steps=len(test))
        
        # Calcular RMSE para esta divisão
        rmse = np.sqrt(mean_squared_error(test, predictions))
        rmse_list.append(rmse)
        
        # Calcular MAE para esta divisão
        mae = mean_absolute_error(test, predictions)
        mae_list.append(mae)
    
    return rmse_list, mae_list


# %%
# Exemplo de uso
serie = df.groupby('data')['vitimas'].sum()
arima_order = (1, 1, 2)  # Substitua pelos melhores parâmetros encontrados
n_splits = 5  # Número de divisões para validação cruzada

rmse_scores, mae_scores = cross_validation_arima(serie, order=arima_order, n_splits=n_splits)
rmse_mean = np.mean(rmse_scores)
mae_mean = np.mean(mae_scores)

print(f"RMSE por divisão: {rmse_scores}")
print(f"RMSE médio: {rmse_mean}")
print(f"MAE por divisão: {mae_scores}")
print(f"MAE médio: {mae_mean}")
# %%
