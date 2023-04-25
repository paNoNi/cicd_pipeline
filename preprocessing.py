import ssl
import string
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import inflect
import nltk
import pandas as pd
from pandas.io.json._json import JsonReader
from tqdm import tqdm



nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


def text_lowercase(text):
    return text.lower()


p = inflect.engine()


def convert_number(text):
    temp_str = text.split()
    new_string = []

    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)

        else:
            new_string.append(word)

    temp_str = ' '.join(new_string)
    return temp_str


def remove_punctuation(text: str):
    for punc in string.punctuation:
        text = text.replace(punc, ' ')
    return text


def remove_whitespace(text):
    return ' '.join(text.split())


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


stemmer = PorterStemmer()


def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return ' '.join(stems)


lemmatizer = WordNetLemmatizer()


def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    return ' '.join(lemmas)


pipeline = [text_lowercase, remove_punctuation, convert_number,
            remove_whitespace, remove_stopwords, stem_words,
            lemmatize_word]


def preproc(text):
    copy_text = text
    for func in pipeline:
        copy_text = func(copy_text)
    return copy_text


