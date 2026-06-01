from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import spacy
import re
import string

nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_original(text):
    text = text.lower()
    return text.split()


import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess_stemming(text):
    text = text.lower()

    # mantém apenas letras e espaços
    text = re.sub(r'[^a-z\s]', ' ', text)
    #pergunta: número também são importantes?

    tokens = text.split()

    return [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words
        and token.isalpha()
        and len(token) > 2  # remove palavras muito curtas, pois estava retornando X, B, algumas coisas de cabeçalho de email, etc.
    ]


nlp = spacy.load("en_core_web_sm")

def preprocess_lemma(text):

    doc = nlp(text)

    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop      # remove stopwords
        and not token.is_punct    # remove pontuação
        and not token.is_space    # remove espaços
        and token.is_alpha        # mantém apenas palavras   
        and len(token) > 2       # remove palavras muito curtas   
    ]