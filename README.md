# Caed

## O que precisa para rodar 
<p>O programa é em python e utiliza as bibliotecas:</p>

- pandas - `pip install pandas`
- string - `pip install strings`
- random - `pip install random`
- spacy - `pip install spacy`
- collections - `pip install collection`
- sklearn - `pip install scikit-learn`
- pickle - `pip install pickle5`
- seaborn - `pip install seaborn`
- nltk - `pip install nltk`
- pysinonimos.sinonimos - `pip install pysinonimos`
- linguagem português spacy - `python -m spacy download pt`

## Arquivos 

<p>O programa possui 4 arquivos .py, sendo eles:</p>

- AjustaDatabase.py - O qual analisa os textos e adiciona os features dos textos para se utilizar no modelo de predição
- ModelPredicao.py - O qual cria o modelo de predição a ser utilizado, usando o skitlearn
- index.py - O qual recebe um quantidade de frases modifica elas e faz a predicção da classificação dessas novas frases gerando um grafico que mostra o quanto a modificação altera a classificação
- Functions.py - O qual possui diversas funções as quais são chamadas pelos demais arquivos

<p>O programa contem tambem uma pasta com os datasets utilizados para fazer os testes, estes estão dentro da pasta Dataset</p>
<p>Além dos modelos de predição que foram gerados pelos testes, os quais estão na pasta Models</p> 

## Como Rodar

<p>O programa tem 3 arquivos que podem ser executados:</p>

- Caso deseje-se gerar uma nova base deve-se rodar o arquivo AjustaDatabase.py - `python AjustaDatabase.py`
- Caso deseje-se refazer o modelo de predição deve-se rodar o arquivo ModelPredicao.py - `python ModelPredicao.py`
- Caso deseje testar os modelos de predição e de modificação de palavras deve-se rodar o arquivo index.py - `python index.py`
