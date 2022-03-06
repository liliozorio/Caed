import Functions
import string
import pandas as pd
import random
import seaborn as sns

dataframe = pd.read_csv('/Dataset/dataset_caed/training.csv')

grau_SVC = []
grau_NNK = []
frases_modificadas = []
 
for i in range(0,100):
  randomico = random.randint(0,len(dataframe)-1)
  frase = dataframe['text'][randomico]
  modificada = muda_sinonimo(frase)
  modificada = muda_genero(modificada)
  classificacao_geral = predicao(modificada)
  classificacao_SVC = classificacao_geral[0]
  classificacao_NNK = classificacao_geral[1]
  frases_modificadas.append(modificada)
  grau_NNK.append(dataframe['level'][randomico] - classificacao_SVC)
  grau_SVC.append(dataframe['level'][randomico] - classificacao_NNK)
  sns.countplot(grau_SVC, label = 'Contagem');

print(frases_modificadas)
