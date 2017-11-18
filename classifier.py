#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:43:22 2017

@author: aliakbar
"""

import os
import numpy as np
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pandas as pd

FILE_NAME = 'dataset_self.csv'
    
def get_words(filename):
    dataset = pd.read_csv(FILE_NAME)
    dataset = dataset.dropna(axis=0)
    list_text = list(dataset['tweet'])
    list_label = list(dataset['class'])
    for id, text in enumerate(list_text):
        words = text.split(' ')
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
    print(res)
    return res


def extract_features(wordsdata):
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


# Create a dictionary of words with its frequency
if 'self_features_matrix.npy' not in os.listdir('.'):
    wordsdata = list(get_words(FILE_NAME))
    dictionary = make_Dictionary(wordsdata)
    fast_dictionary = build_fastdict(dictionary)
    print('dictionary created')

    features_matrix, labels = extract_features(wordsdata)
    np.save('self_features_matrix.npy', features_matrix)
    np.save('self_labels.npy', labels)
else:
    features_matrix = np.load('self_features_matrix.npy')
    labels = np.load('self_labels.npy')

print(features_matrix.shape)
print(labels.shape)
print(sum(labels == 0), sum(labels == 1))

X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

# Training models and its variants

models = [
    ('Support Vector Machine', LinearSVC()),
    ('Naive Bayes', MultinomialNB()),
    ('Stochastic Gradient Descent', SGDClassifier()),
    ('Multilayer Perceptron', MLPClassifier(hidden_layer_sizes=(5, 5), solver='adam')),
]

import pickle
for model_name, model in models:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    end = time.perf_counter()
    result = model.predict(X_test)

    print('Time to train {}: {} ms'.format(model_name, int((end - start) * 1000)))

    print('Confusion matrix for {}:'.format(model_name))
    print(confusion_matrix(y_test, result))
    print('Accuracy for {}: {}'.format(model_name, model.score(X_test, y_test)))
    # Save model
    filename = '{}_model.sav'.format(model_name)
    pickle.dump(model, open(filename, 'wb'))
    
    # Load Example
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)
