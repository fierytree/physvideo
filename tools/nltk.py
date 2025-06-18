import nltk
nltk.download('words')
from nltk.corpus import words
import re

word_list = words.words('en-basic')

def f(num):
    return word_list[num % len(word_list)]

def h(word):
    return word_list.index(word.lower())