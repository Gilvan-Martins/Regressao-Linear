from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

# Cria um dataset chamado 'df' que receberá os dados do csv
df = pd.read_csv("FuelConsumptionCo2.csv")

# Exibe a estrutura do DataFrame
print(df.head())

# Resumo dos dados csv
print(df.describe())

# Separando os motores e a emissão de CO2
motores = df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]
print(motores.head())
print(co2.head())

# Separando os grupos de treinamento e de teste
motores_treino, motores_test, co2_treino, co2_teste = train_test_split(motores, co2, test_size=0.2, random_state=42)
print(type(motores_treino))

# Plotando o gráfico de correlação motores e CO2
plt.scatter(motores_treino, co2_treino, color='blue')
plt.xlabel("Motor")
plt.ylabel("Emissão de CO2")
plt.show()

# Criar um modelo de tipo de regressão linear
modelo = linear_model.LinearRegression()

# Treinar o modelo usando o dataset de treino
modelo.fit(motores_treino, co2_treino)

print('(A) Intercepto: ', modelo.intercept_)
print('(B) Inclinação: ', modelo.coef_)

# Plotando o gráfico com a reta de regressão
plt.scatter(motores_treino, co2_treino, color='blue')
plt.plot(motores_treino, modelo.coef_[0][0] * motores_treino + modelo.intercept_[0], '-r')
plt.xlabel("Motor")
plt.ylabel("Emissão de CO2")
plt.show()

# Executando o modelo no dataset de teste
predicoesCo2 = modelo.predict(motores_test)

# Plotando o gráfico com a linha de regressão no dataset de teste
plt.scatter(motores_test, co2_teste, color='green')
plt.plot(motores_test, modelo.coef_[0][0] * motores_test + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de CO2")
plt.xlabel("Motores")
plt.show()

# Agora é mostrar as métricas
sse = np.sum((predicoesCo2 - co2_teste) ** 2, axis=0)
print("Soma dos Erros ao Quadrado (SSE): %.2f" % float(sse.iloc[0]))  # Correção aqui, não precisa converter para float
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(co2_teste, predicoesCo2))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(co2_teste, predicoesCo2))
print("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(co2_teste, predicoesCo2)))
print("R2-score: %.2f" % r2_score(co2_teste, predicoesCo2))
