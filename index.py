import Functions
import string
import pandas as pd
import random
import seaborn as sns

dataframe = pd.read_csv('/Dataset/training.csv')

grau_SVC = []
grau_NNK = []
frases_modificadas = []
 
for i in range(0,100):
  randomico = ramdom.randint(0,len(dataframe)
  frase = dataframe['text'][randomico]
  modificada = muda_sinonimo(frase)
  modificada = muda_genero(modificada)
  classificacao_SVC = predicao(modificada)[0]
  classificacao_NNK = predicao(modificada)[1]
  frases_modificadas.append(modificada)
  grau_NNK.append(dataframe['level'][randomico] - classificacao_SVC)
  grau_SVC.append(dataframe['level'][randomico] - classificacao_NNK)
  sns.countplot(grau_SVC, label = 'Contagem');

print(frases_modificadas)
