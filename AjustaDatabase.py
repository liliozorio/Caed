Import Functions
Import Pandas as pd

#Ajusta Dataset
dataframe = pd.read_csv('/Dataset/training.csv')
dataframe['quantPalavras'] = dataframe['text'].apply(conta_palavras)
dataframe['virgula'] = dataframe['text'].apply(quantidade_virgulas)
dataframe['quantStopWords'] = dataframe['text'].apply(conta_stop_words)
dataframe['canonicas'] = dataframe['text'].apply(canonicidade)
dataframe.to_csv('/Dataset/train_final.csv', index = False)