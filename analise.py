# Site do download do db
# https://www.kaggle.com/datasets/prakashraushan/loan-dataset

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=
# Será analisado a Idade do Cliente em relação a sua renda anual
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
# Necessario usar pip install scikit-learn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Função para leitura do bando de dados
df = pd.read_csv("LoanDatasetLoansDatasest.csv")

print("Head do dataframe:")
print(df[['customer_age', 'customer_income']].rename(
    columns={'customer_age': 'Idade', 'customer_income': 'Salario Anual'}).head(3))

# Tratamento de Colunas, transformando em tipo int ou float
# Pois esta entendendo como string e ao fazer .sum(), concatenou
df['customer_age'] = df['customer_age'].astype(int)
# Além disso, esta com virgula na casa de milhar, coloquei tipo float, devido à capacidade de dado
df['customer_income'] = df['customer_income'].str.replace(',', '').astype(float)

df['loan_amnt'] = df['loan_amnt'].str.replace('£', '')
df['loan_amnt'] = df['loan_amnt'].str.replace(',', '').astype(float)

print("\nTabela de Correlação:")
correlacao = df[['customer_age', 'employment_duration', 'customer_income',
                'loan_amnt', 'term_years', 'cred_hist_length']].corr()
print(correlacao)

# Exibir a correlação específica entre customer_income e customer_age
correlacao_Idade_Salario = df['customer_age'].corr(df['customer_income'])
print("\nCorrelação entre a Idade e Salario Atual (£): r = {:.3f}%".format(correlacao_Idade_Salario*100))

# Prova Real para a Coluna Idade e Salario Anual irei Utilizar
a = df[['customer_age', 'customer_income','employment_duration', 'loan_amnt', 'term_years']].corr()
somaX = df['customer_age'].sum()
somaY = df['customer_income'].sum()
XX = (df['customer_age'] ** 2).sum()
YY = (df['customer_income'] ** 2).sum()
somaXY = (df['customer_age'] * df['customer_income']).sum()

# Número de linhas do DataFrame
n = len(df)

# Calculo para a Correlação
r = (n*somaXY - somaX*somaY) / (math.sqrt((n*XX - somaX*somaX) * (n*YY - somaY*somaY)))

print("Correlação entre a Idade e Salario Anual (£): r = {:.3f}%".format(r*100))

# Regressão Linear
    # Definindo os eixos
x = df[['customer_age']]
y = df['customer_income']

# Criar um modelo de regressão linear
modelo = LinearRegression()
modelo.fit(x, y)

# Coeficientes
a_coeff = modelo.coef_
l_coeff = modelo.intercept_

#yRegressao = a_coeff * x + l_coeff
# Previsões feitas pelo modelo
yRegressao = modelo.predict(x)

# Erro -
print("\nErros:")
    # Erro Absoluto Médio (MAE): Calcula o erro absoluto entre os valores reais de renda (y) e os valores previstos (yRegressao).
MAE = ((abs(y - yRegressao)).sum()) / n
print("Erro Absoluto Médio: ", MAE)
    # MSE: Calcula o erro quadrático entre os valores reais de renda e os previstos.
MSE = ((abs(y - yRegressao)**2).sum()) / n
print("Erro Quadratico Médio: ", MSE)

print("\nRegressão Linear: ")
print("Coeficiente Angular: {:.3f} \nCoeficiente Linear {:.3f}".format(a_coeff[0], l_coeff))
print("Y^= {:.3f}x + {:.3f}".format(a_coeff[0], l_coeff))

# Plotar a reta com os coeficientes
plt.scatter(x, y, color='blue')
plt.plot(x, yRegressao, color='red')
plt.title('Regressão Linear: Idade x Renda Anual (£)')
plt.xlabel('Idade')
plt.ylabel('Renda Anual (£)')
plt.show() 