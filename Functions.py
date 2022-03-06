import pandas as pd
import string
import random
import spacy
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import seaborn as sns
import nltk
from pysinonimos.sinonimos import Search, historic

#Calcula quantos % da frase é de palavras canonicas
def canonicidade(texto):
  
  pln = spacy.load('pt')
  texto = texto.lower()
  documento = pln(texto)

  lista = []
  for token in documento:
    lista.append(token.text)

  quantidade = 0;
  frase = []
  for word in lista:
    if len(word)%2 == 0:
      canonica = True
      letras = 0;
      for char in word:
        if letras%2 == 0 and (char == 'a' or char == 'e' or char == 'i' or char == 'o' or char == 'u'):
          canonica = False
          break
        elif not (letras%2 == 0) and not (char == 'a' or char == 'e' or char == 'i' or char == 'o' or char == 'u'):
          canonica = False
          break
        letras = letras + 1 
      if canonica:
        quantidade = quantidade+1
  
  return 100*(quantidade/len(lista))

#Calcula quantas palavras a frase possui
def conta_palavras(texto):

  pln = spacy.load('pt')
  text = pln(texto)
  return len(text)


#Calcula quantas virgulas a frase possui
def quantidade_virgulas(texto):
  
  pln = spacy.load('pt')
  text = pln(texto)
  conta = 0
  counts = Counter(text.text)
  conta = counts[',']
  return conta


#Calcula quantas stop words a frase possui
def conta_stop_words(texto):
  
  pln = spacy.load('pt')
  text = pln(texto)
  stop_words = spacy.lang.pt.stop_words.STOP_WORDS
  conta = 0
  for token in text:   
    if token.text in stop_words:
      conta = conta+1
  return 100*(conta/len(text))

def predicao(texto):

  data = pd.DataFrame()
  text = {}
  text['virgula'] = quantidade_virgulas(texto)
  text['quantPalavras'] = conta_palavras(texto)
  text['canonicas'] = canonicidade(texto)
  text['quantStopWords'] = conta_stop_words(texto)
  data = data.append(text, ignore_index=True) 

  file_KNN = open('/Models/KNN_model.sav', 'rb')
  file_SVC = open('/Models/KNN_model.sav', 'rb')

  file_KNN = file_KNN.read()
  file_SVC = file_SVC.read()
  KNN_prediction = pickle.loads(file_KNN)
  SVC_prediction = pickle.loads(file_SVC)

  return SVC_prediction.predict(data)[0], KNN_prediction.predict(data)[0]
#Cria grafico para visualizar distribuições na base
def Graficos(dataframe) 
   
  sns.countplot(dataframe['level'], label = 'Contagem');
  sns.heatmap(pd.isnull(dataframe));


def descobre_genero(texto):
  if texto[len(texto)-5:len(texto)] == 'grama' or texto[len(texto)-6:len(texto)] == 'gramas':
    return 'masculino'
  elif texto[len(texto)-5:len(texto)] == 'mento' or texto[len(texto)-6:len(texto)] == 'mento':
    return 'masculino'
  elif texto[len(texto)-4:len(texto)] == 'agem' or texto[len(texto)-5:len(texto)] == 'agens':
    if not texto == 'personagem' or not texto == 'personagens':
      return 'feminina'
    else:
      return 'neutra'
  elif texto[len(texto)-4:len(texto)] == 'ista' or texto[len(texto)-5:len(texto)] == 'istas':
    return 'neutra'
  elif texto[len(texto)-4:len(texto)] == 'tica' or texto[len(texto)-5:len(texto)] == 'ticas':
    return 'feminina'
  elif texto[len(texto)-4:len(texto)] == 'ície' or texto[len(texto)-5:len(texto)] == 'ícies':
    return 'feminina'
  elif texto[len(texto)-4:len(texto)] == 'ente' or texto[len(texto)-5:len(texto)] == 'entes':
    return 'neutra'
  elif texto[len(texto)-4:len(texto)] == 'ante' or texto[len(texto)-5:len(texto)] == 'antes':
    return 'neutra'
  elif texto[len(texto)-3:len(texto)] == 'ema' or texto[len(texto)-4:len(texto)] == 'emas':
    return 'masculino'
  elif texto[len(texto)-3:len(texto)] == 'oma' or texto[len(texto)-4:len(texto)] == 'omas':
    return 'masculino'
  elif texto[len(texto)-3:len(texto)] == 'ade' or texto[len(texto)-4:len(texto)] == 'ades':
    if not texto == 'nômade' or not texto == 'nômades':
      return 'feminina'
    else:
      return 'neutra'
  elif texto[len(texto)-3:len(texto)] == 'ção' or texto[len(texto)-4:len(texto)] == 'ções':
    if not texto == 'coração' or not texto == 'corações':
      return 'feminina'
    else:
      return 'masculino'
  elif texto[len(texto)-2:len(texto)] == 'ão' or texto[len(texto)-3:len(texto)] == 'ões':
    return 'masculino'
  elif texto[len(texto)-3:len(texto)] == 'ora' or texto[len(texto)-4:len(texto)] == 'oras':
    return 'feminina'
  elif texto[len(texto)-2:len(texto)] == 'or' or texto[len(texto)-4:len(texto)] == 'ores':
    return 'masculino'
  elif texto[len(texto)-1] == 'á' or texto[len(texto)-2:len(texto)] == 'ás':
    return 'masculino'
  elif texto[len(texto)-1] == 'o' or texto[len(texto)-2:len(texto)] == 'os':
    return 'masculino'
  elif texto[len(texto)-1] == 'a' or texto[len(texto)-2:len(texto)] == 'as':
    return 'feminina'

#Tenta trocar o genero da palavra passada por parametro sendo ela um substantivo ou pronome
def descobre_oposto_substantivo_pronome(texto):
  if texto[len(texto)-5:len(texto)] == 'grama' or texto[len(texto)-6:len(texto)] == 'gramas':
    return texto, 'nada'
  elif texto[len(texto)-5:len(texto)] == 'mento' or texto[len(texto)-6:len(texto)] == 'mento':
    return texto, 'nada'
  elif texto[len(texto)-4:len(texto)] == 'agem' or texto[len(texto)-5:len(texto)] == 'agens':
    if texto == 'personagem' or texto == 'personagens':
      return texto, 'neutra'
    else:
      return texto, 'nada'
  elif texto[len(texto)-4:len(texto)] == 'ista' or texto[len(texto)-5:len(texto)] == 'istas':
    return texto, 'nada'
  elif texto[len(texto)-4:len(texto)] == 'tica' or texto[len(texto)-5:len(texto)] == 'ticas':
    return texto, 'nada'
  elif texto[len(texto)-4:len(texto)] == 'ície' or texto[len(texto)-5:len(texto)] == 'ícies':
    return texto,'nada'
  elif texto[len(texto)-4:len(texto)] == 'ista' or texto[len(texto)-5:len(texto)] == 'istas':
    return texto, 'neutra'
  elif texto[len(texto)-4:len(texto)] == 'ente' or texto[len(texto)-5:len(texto)] == 'entes':
    return texto, 'neutra'
  elif texto[len(texto)-4:len(texto)] == 'ante' or texto[len(texto)-5:len(texto)] == 'antes':
    return texto, 'neutra'
  elif texto[len(texto)-3:len(texto)] == 'ema' or texto[len(texto)-4:len(texto)] == 'emas':
    return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'oma' or texto[len(texto)-4:len(texto)] == 'omas':
    return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'ade' or texto[len(texto)-4:len(texto)] == 'ades':
    if texto == 'nômade' or texto == 'nômades':
      return texto, 'neutra'
    else:
      return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'ção' or texto[len(texto)-4:len(texto)] == 'ções':
    return texto, 'nada'
  elif texto[len(texto)-2:len(texto)] == 'ão' or texto[len(texto)-3:len(texto)] == 'ões':
    return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'rio' or texto[len(texto)-4:len(texto)] == 'rios':
    return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'soa' or texto[len(texto)-4:len(texto)] == 'soas':
    return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'ora' or texto[len(texto)-4:len(texto)] == 'oras':
    if texto[len(texto)-3:len(texto)] == 'ora':
      texto = texto[:len(texto)-1]
      return texto, 'masculino'
    else:
      texto = texto[:len(texto)-2]
      texto = texto + 'es'
      return texto, 'masculino'
  elif texto[len(texto)-2:len(texto)] == 'or' or texto[len(texto)-4:len(texto)] == 'ores':
    if texto[len(texto)-2:len(texto)] == 'or':
      texto = texto + 'a'
      return texto, 'feminina'
    else:
      texto = texto[:len(texto)-2]
      texto = texto + 'as'
      return texto, 'feminina'
  elif texto[len(texto)-3:len(texto)] == 'ano' or texto[len(texto)-4:len(texto)] == 'anos' or texto[len(texto)-3:len(texto)] == 'Ano' or texto[len(texto)-4:len(texto)] == 'Anos':
    return texto, 'nada'
  elif texto[len(texto)-1] == 'á' or texto[len(texto)-2:len(texto)] == 'ás':
    return texto, 'nada'
  elif texto[len(texto)-1] == 'o' or texto[len(texto)-2:len(texto)] == 'os' or texto[len(texto)-1] == 'O' or texto[len(texto)-2:len(texto)] == 'Os':
    if texto[len(texto)-1] == 'o':
      texto = texto[:len(texto)-1]
      texto = texto + 'a'
      return texto, 'feminina'
    elif texto[len(texto)-1] == 'O':
      texto = texto[:len(texto)-1]
      texto = texto + 'A'
      return texto, 'feminina'
    elif texto[len(texto)-2:len(texto)] == 'Os':
      texto = texto[:len(texto)-2]
      texto = texto + 'As'
      return texto, 'feminina'
    else:
      texto = texto[:len(texto)-2]
      texto = texto + 'as'
      return texto, 'feminina'
  elif texto[len(texto)-1] == 'a' or texto[len(texto)-2:len(texto)] == 'as' or texto[len(texto)-1] == 'A' or texto[len(texto)-2:len(texto)] == 'As':
    if texto[len(texto)-1] == 'a':
      texto = texto[:len(texto)-1]
      texto = texto + 'o'
      return texto, 'masculino'
    elif texto[len(texto)-1] == 'A':
      texto = texto[:len(texto)-1]
      texto = texto + 'O'
      return texto, 'masculino'
    elif texto[len(texto)-2:len(texto)] == 'As':
      texto = texto[:len(texto)-2]
      texto = texto + 'Os'
      return texto, 'masculino'
    else:
      texto = texto[:len(texto)-2]
      texto = texto + 'os'
      return texto, 'masculino'
  return texto, 'nada'


#Tenta trocar o genero da palavra passada por parametro sendo ela um adjetivo
def descobre_oposto_adjetivo(texto):
  if texto[len(texto)-5:len(texto)] == 'grama' or texto[len(texto)-6:len(texto)] == 'gramas':
    return texto, 'nada'
  elif texto[len(texto)-5:len(texto)] == 'mento' or texto[len(texto)-6:len(texto)] == 'mento':
    return texto, 'nada'
  elif texto[len(texto)-4:len(texto)] == 'agem' or texto[len(texto)-5:len(texto)] == 'agens':
    if texto == 'personagem' or texto == 'personagens':
      return texto, 'neutra'
    else:
      return texto, 'nada'
  elif texto[len(texto)-4:len(texto)] == 'ista' or texto[len(texto)-5:len(texto)] == 'istas':
    return texto, 'nada'
  elif texto[len(texto)-4:len(texto)] == 'tica' or texto[len(texto)-5:len(texto)] == 'ticas':
    return texto, 'nada'
  elif texto[len(texto)-4:len(texto)] == 'ície' or texto[len(texto)-5:len(texto)] == 'ícies':
    return texto,'nada'
  elif texto[len(texto)-4:len(texto)] == 'ista' or texto[len(texto)-5:len(texto)] == 'istas':
    return texto, 'neutra'
  elif texto[len(texto)-4:len(texto)] == 'ente' or texto[len(texto)-5:len(texto)] == 'entes':
    return texto, 'neutra'
  elif texto[len(texto)-4:len(texto)] == 'ante' or texto[len(texto)-5:len(texto)] == 'antes':
    return texto, 'neutra'
  elif texto[len(texto)-3:len(texto)] == 'ema' or texto[len(texto)-4:len(texto)] == 'emas':
    return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'oma' or texto[len(texto)-4:len(texto)] == 'omas':
    return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'ade' or texto[len(texto)-4:len(texto)] == 'ades':
    if texto == 'nômade' or texto == 'nômades':
      return texto, 'neutra'
    else:
      return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'ção' or texto[len(texto)-4:len(texto)] == 'ções':
    return texto, 'nada'
  elif texto[len(texto)-2:len(texto)] == 'ão' or texto[len(texto)-3:len(texto)] == 'ões':
    return texto, 'nada'
  elif texto[len(texto)-3:len(texto)] == 'ora' or texto[len(texto)-4:len(texto)] == 'oras':
    if texto[len(texto)-3:len(texto)] == 'ora':
      texto = texto[:len(texto)-1]
      return texto, 'masculino'
    else:
      texto = texto[:len(texto)-2]
      texto = texto + 'es'
      return texto, 'masculino'
  elif texto[len(texto)-2:len(texto)] == 'or' or texto[len(texto)-4:len(texto)] == 'ores':
    if texto[len(texto)-2:len(texto)] == 'or':
      texto = texto + 'a'
      return texto, 'feminina'
    else:
      texto = texto[:len(texto)-2]
      texto = texto + 'as'
      return texto, 'feminina'
  elif texto[len(texto)-2:len(texto)] == 'om' or texto[len(texto)-3:len(texto)] == 'ons':
    if texto[len(texto)-2:len(texto)] == 'om':
      texto = texto[:len(texto)-1]
      texto = texto + "a"
      return texto, "feminina"
    else:
      texto = texto[:len(texto)-2]
      texto = texto + "as"
      return texto, "feminina"
  elif texto[len(texto)-2:len(texto)] == 'oa' or texto[len(texto)-3:len(texto)] == 'oas':
    if texto[len(texto)-2:len(texto)] == 'oa':
      texto = texto[:len(texto)-1]
      texto = texto + "m"
      return texto, "masculino"
    else:
      texto = texto[:len(texto)-2]
      texto = texto + "ns"
      return texto, "masculino"
  elif texto[len(texto)-1] == 'á' or texto[len(texto)-2:len(texto)] == 'ás':
    return texto, 'nada'
  elif texto[len(texto)-1] == 'o' or texto[len(texto)-2:len(texto)] == 'os' or texto[len(texto)-1] == 'O' or texto[len(texto)-2:len(texto)] == 'Os':
    if texto[len(texto)-1] == 'o':
      texto = texto[:len(texto)-1]
      texto = texto + 'a'
      return texto, 'feminina'
    elif texto[len(texto)-1] == 'O':
      texto = texto[:len(texto)-1]
      texto = texto + 'A'
      return texto, 'feminina'
    elif texto[len(texto)-2:len(texto)] == 'Os':
      texto = texto[:len(texto)-2]
      texto = texto + 'As'
      return texto, 'feminina'
    else:
      texto = texto[:len(texto)-2]
      texto = texto + 'as'
      return texto, 'feminina'
  elif texto[len(texto)-1] == 'a' or texto[len(texto)-2:len(texto)] == 'as' or texto[len(texto)-1] == 'A' or texto[len(texto)-2:len(texto)] == 'As':
    if texto[len(texto)-1] == 'a':
      texto = texto[:len(texto)-1]
      texto = texto + 'o'
      return texto, 'masculino'
    elif texto[len(texto)-1] == 'A':
      texto = texto[:len(texto)-1]
      texto = texto + 'O'
      return texto, 'masculino'
    elif texto[len(texto)-2:len(texto)] == 'As':
      texto = texto[:len(texto)-2]
      texto = texto + 'Os'
      return texto, 'masculino'
    else:
      texto = texto[:len(texto)-2]
      texto = texto + 'os'
      return texto, 'masculino'
  return texto, 'nada'


#Modifica frase mudando seu genero
def muda_genero(texto):

  pln = spacy.load('pt')
  stop_words = spacy.lang.pt.stop_words.STOP_WORDS
  text = pln(texto)
  i = 0
  lista = {}
  for token in text:
    lista[str(i)] = token
    i = i+1
  for i in lista:
    try:
      if (lista[i].pos_ == 'NOUN' or lista[i].pos_ == 'PRP') and not lista[i].text in stop_words:
        if not descobre_oposto_substantivo_pronome(lista[i].text)[1] == 'nada':
          lista[i] = descobre_oposto_substantivo_pronome(lista[i].text)[0]
          antecessores = list(text[1].ancestors)
          filhos = list(text[1].children)
          for antecessor in antecessores:
            for j in lista:
              if antecessor.text == str(lista[j]):
                lista_filhos = list(text[int(j)].children)
                for filho in lista_filhos:
                  if lista[i] == descobre_oposto_adjetivo(filho.text)[0] and (antecessor.pos_ == 'ADJ' or antecessor.pos_ == 'DT'):
                    lista[j] = str(descobre_oposto_adjetivo(antecessor.text)[0])
          for filho in filhos:
            for j in lista:
              if filho.text == str(lista[j]):
                lista_antecessores = list(text[int(j)].ancestors)
                for antecessor in lista_antecessores:
                  if lista[i] == descobre_oposto_substantivo_pronome(antecessor.text)[0] and (filho.pos_ == 'ADJ' or filho.pos_ == 'DET'):
                    lista[j] = str(descobre_oposto_substantivo_pronome(filho.text)[0])
    except:
      pass
  texto = ' '.join([str(lista[elemento]) for elemento in lista])
  return texto

#Troca algumas palavras do texto por sinonimos
def muda_sinonimo(texto):
   
  pln = spacy.load('pt')
  stop_words = spacy.lang.pt.stop_words.STOP_WORDS
  text = pln(texto)
  lista = []
  for token in text:
    if (token.pos_ == 'NOUN' or token.pos_ == 'NNP') and not token.text in stop_words:
      lista.append(token)
  random.shuffle(lista)
  tamanho = len(lista)

  for i in range(1, tamanho):

    word = Search(lista[i].text)
    sinonimos_word = word.synonyms()
    if not sinonimos_word == 404:
      max = 0
      sinonimo = ''
      for sinonimos in sinonimos_word:
        palavra = pln(sinonimos)
        if descobre_genero(lista[i].text) == descobre_genero(sinonimos):
          if lista[i].similarity(palavra) > max:
            max = palavra.similarity(palavra)
            sinonimo = sinonimos
        if not sinonimo == '':
          texto = texto.replace(lista[i].text, sinonimo)

  return texto
