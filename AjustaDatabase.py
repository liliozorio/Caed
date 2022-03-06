Import Functions
Import Pandas as pd
Import sys

local = str(sys.argv[1])
#Cria bases
lista1 = []
lista2 = []
lista3 = []
lista4 = []

for i in range(1,827):
  if i < 298:
    lista1.append(i)
    lista2.append(i)
    lista3.append(i)
    lista4.append(i)
  elif i < 326:
    lista2.append(i)
    lista3.append(i)
    lista4.append(i)
  elif i < 629:
    lista3.append(i)
    lista4.append(i)
  else:
    lista4.append(i)

random.shuffle(lista1)
random.shuffle(lista2)
random.shuffle(lista3)
random.shuffle(lista4)

traning = pd.DataFrame(columns=['text', 'level'])
test = pd.DataFrame(columns=['text', 'level'])

for i in range(0,20):
  frase1 = open("/Dataset/dataset_caed/1_Ensino_Fundamental_I/" + str(lista1[i]) + ".txt")
  frase1 = frase1.read()
  frases = frase1.split('.')
  for frase in frases:
    frase = frase.replace('\n', '')
    if len(frase) > 2:
      newrow1 = {'text':frase,'level':'1'}
      traning = traning.append(newrow1, ignore_index=True) 
  frase2 = open("/Dataset/dataset_caed/2_Ensino_Fundamental_II/" + str(lista2[i]) + ".txt")
  frase2 = frase2.read()
  frases = frase2.split('.')
  for frase in frases:
    frase = frase.replace('\n', '')
    if len(frase) > 2:
      newrow2 = {'text':frase,'level':'2'}
      traning = traning.append(newrow2, ignore_index=True) 
  frase3 = open("/Dataset/dataset_caed/3_Ensino_Medio/" + str(lista3[i]) + ".txt")
  frase3 = frase3.read()
  frases = frase3.split('.')
  for frase in frases:
    frase = frase.replace('\n', '')
    if len(frase) > 2:
      newrow3 = {'text':frase,'level':'3'}
      traning = traning.append(newrow3, ignore_index=True) 
  frase4 = open("/Dataset/dataset_caed/4_Ensino_Superior/" + str(lista4[i]) + ".txt")
  frase4 = frase4.read()
  frases = frase4.split('.')
  for frase in frases:
    frase = frase.replace('\n', '')
    if len(frase) > 2:
      newrow4 = {'text':frase,'level':'4'}
      traning = traning.append(newrow4, ignore_index=True) 
traning.to_csv('/Dataset/training_ ' + local +'.csv', index = False)

#Ajusta Dataset
dataframe = pd.read_csv('/Dataset/training_ ' + local +'.csv')
dataframe['quantPalavras'] = dataframe['text'].apply(conta_palavras)
dataframe['virgula'] = dataframe['text'].apply(quantidade_virgulas)
dataframe['quantStopWords'] = dataframe['text'].apply(conta_stop_words)
dataframe['canonicas'] = dataframe['text'].apply(canonicidade)
dataframe.to_csv('/Dataset/train_final_ ' + local +'.csv', index = False)
