from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

data_set = pd.read_csv("housing.csv")

rm = data_set[["RM"]]

medv = data_set[["MEDV"]]

print(rm.head())
print(medv.head())

rm_treino, rm_teste, medv_treino, medv_teste = train_test_split(rm, medv, test_size=0.2, random_state=42)

plt.scatter(rm, medv, color ='blue')
plt.xlabel("rm")
plt.ylabel("medv")
plt.show()


modelo = linear_model.LinearRegression()

modelo.fit(rm_treino, medv_treino)

print("A = ", modelo.intercept_)
print("B = ", modelo.coef_)

plt.scatter(rm_treino, medv_treino, color ='blue')
plt.plot(rm_treino, modelo.coef_[0][0] * rm_treino + modelo.intercept_[0], '-r')
plt.xlabel("rm")
plt.ylabel("medv")
plt.show()


predicao_preco = modelo.predict(rm_teste)


plt.scatter(rm_teste, medv_teste, color ='blue')
plt.plot(rm_teste, modelo.coef_[0][0] * rm_teste + modelo.intercept_[0], '-r')
plt.xlabel("rm")
plt.ylabel("medv")
plt.show()


sse = np.sum((predicao_preco - medv_teste) ** 2, axis=0)
print("Soma dos Erros ao Quadrado (SSE): %.2f" % float(sse.iloc[0]))  # Correção aqui, não precisa converter para float
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(medv_teste, predicao_preco))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(medv_teste, predicao_preco))
print("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(medv_teste, predicao_preco)))
print("R2-score: %.2f" % r2_score(medv_teste, predicao_preco))
