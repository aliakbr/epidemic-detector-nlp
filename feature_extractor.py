# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 05:04:26 2017

@author: Ali-pc
"""

from nltk.corpus import stopwords
from collections import Counter
stop_words = set(stopwords.words('english'))
import pandas as pd
import numpy as np

def get_words(filename):
    dataset = pd.read_csv(filename)
    dataset = dataset.dropna(axis=0)
    list_text = list(dataset['tweet'])
    list_label = list(dataset['class'])
    for id, text in enumerate(list_text):
        words = text.lower()
        words = words.split(' ')
        for id1, lab in enumerate(list_label):
            if id1 == id:
                label = lab
                break
        yield(label, words)
        
def make_Dictionary(wordsdata):
    all_words = []

    for _, words in wordsdata:
        all_words += words
            
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary.keys())

    for item in list_to_remove:
        if not item.isalpha():
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
        elif item in stop_words:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)

    return dictionary


def build_fastdict(dictionary):
    res = {}
    for i, d in enumerate(a for a, b in dictionary):
        res[d] = i
    return res


def extract_features(wordsdata, fast_dictionary):
    docID = 0
    features_matrix = np.zeros((33800, 3000))
    train_labels = np.zeros(33800)

    for class_label, all_words in wordsdata:
        for words in all_words:
            if words in fast_dictionary.keys():
                wordID = fast_dictionary[words]
                features_matrix[docID, wordID] += 1

        train_labels[docID] = class_label
        docID = docID + 1
    return features_matrix, train_labels

def create_features_matrix(filename):
    wordsdata = list(get_words(filename))
    dictionary = make_Dictionary(wordsdata)
    fast_dictionary = build_fastdict(dictionary)
    print('dictionary created')

    features_matrix, labels = extract_features(wordsdata, fast_dictionary)
    return features_matrix, labels