import nltk
# nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# librairie nltk pour tokkenizer, stemmer (raicne) et bag of word
def tokenize(sentence):
    """
    nltk.word_tokenize permet de separer les mots d'une phrase
    et les ajoute en talbeau de mots
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Trouve la racine du mot 
    preparer, preparation -> prepa
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    bag_of_words : liste de mots associee à une liste binaire : 
    -> 0 si le mots n'est pas present dans la phrase, 1 sinon 
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag