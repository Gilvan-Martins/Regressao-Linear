from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

# Criando o data_set com os dados do arquivo csv
data_set = pd.read_csv("kc_house_data.csv")

# Mostrando o data_set
print(data_set.head())

# Separando as colunas que iram ser estudadas
tam = data_set[["sqft_living"]]

preco = data_set[["price"]]

# Mostrando as respctivas colunas
print(data_set[["sqft_living"]])
print(data_set[["price"]])

# Separando o data_set em treino e teste
tam_treino, tam_teste, preco_treino, preco_teste = train_test_split(tam, preco, test_size=0.3, random_state=0)

# Plotando o grafico de cep/preco
plt.scatter(tam_treino, preco_treino, color = 'blue')
plt.xlabel("tam")
plt.ylabel("price")
plt.show()

modelo = linear_model.LinearRegression()

modelo.fit(tam_treino, preco_treino)

print("A = ", modelo.intercept_)
print("B = ", modelo.coef_)

plt.scatter(tam_treino, preco_treino, color = 'blue')
plt.plot(tam_treino, modelo.coef_[0][0] * tam_treino + modelo.intercept_[0], '-r')
plt.xlabel("tam")
plt.ylabel("price")
plt.show()


predicao_preco = modelo.predict(tam_teste)

plt.scatter(tam_teste, preco_teste, color = 'blue')
plt.plot(tam_teste, modelo.coef_[0][0] * tam_teste + modelo.intercept_[0], '-r')
plt.xlabel("tam")
plt.ylabel("price")
plt.show()


sse = np.sum((predicao_preco - tam_teste) ** 2, axis=0)
print("Soma dos Erros ao Quadrado (SSE): %.2f" % float(sse.iloc[0])) 
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(tam_teste, predicao_preco))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(tam_teste, predicao_preco))
print("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(tam_teste, predicao_preco)))
print("R2-score: %.2f" % r2_score(tam_teste, predicao_preco))


score = modelo.score(tam_teste, preco_teste)
print(score)