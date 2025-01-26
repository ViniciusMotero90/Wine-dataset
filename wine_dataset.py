import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# Carregar os dados do arquivo CSV
# Certifique-se de que o arquivo 'wine_dataset.csv' está no mesmo diretório ou forneça o caminho completo
wine = pd.read_csv('wine_dataset.csv')

y = wine['style']
x = wine.drop('style',axis=1)

# Dividir os dados em conjunto de treinamento e teste

# Treinar o modelo Naive Bayes
modelo = GaussianNB()
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo,x, y, cv=skfold)
print(resultado.mean())
