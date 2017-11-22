#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:43:22 2017

@author: aliakbar
"""

import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from preprocess import preprocess, tokenize
import pickle

FILE_NAME = 'dataset_related.csv'

def clean(s):
    s = preprocess(s)
    tokenize_s = tokenize(s)
    for item in tokenize_s:
        if not item.isalpha():
            tokenize_s.remove(item)
        elif len(item) == 1:
            tokenize_s.remove(item)
    return ' '.join(tokenize_s)

# Extract feature testing with TfIdfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Get dataset
data_file = pd.read_csv(FILE_NAME, engine='python')
data_file = data_file.dropna(axis=0)
data_train = {}
data = []
for tweet in list(data_file['tweet']):
    data.append(clean(tweet))
data_train['data'] = data
data_train['target'] = list(data_file['class'])

# Save vocabulary
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(data_train['data'])
feature_vocab = vectorizer.vocabulary_
feature_vocab_file = open('feature_vocab.pkl', 'wb')

pickle.dump(feature_vocab, feature_vocab_file, pickle.HIGHEST_PROTOCOL)

# Create dataset split
labels = data_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.40)

# Create a dictionary of words with its frequency
#from feature_extractor import create_features_matrix
#if 'self_features_matrix.npy' not in os.listdir('.'):
#    features_matrix, labels = create_features_matrix(FILE_NAME)
#    np.save('self_features_matrix.npy', features_matrix)
#    np.save('self_labels.npy', labels)
#else:
#    features_matrix = np.load('self_features_matrix.npy')
#    labels = np.load('self_labels.npy')
#
#print(features_matrix.shape)
#print(labels.shape)
#print(sum(labels == 0), sum(labels == 1))

#X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

# Training models and its variants

models = [
    ('Support Vector Machine', LinearSVC()),
    ('Naive Bayes', MultinomialNB()),
    ('Stochastic Gradient Descent', SGDClassifier()),
    ('Multilayer Perceptron', MLPClassifier(hidden_layer_sizes=(5, 5), solver='adam')),
]

from sklearn.metrics import f1_score
for model_name, model in models:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    end = time.perf_counter()
    result = model.predict(X_test)

    print('Time to train {}: {} ms'.format(model_name, int((end - start) * 1000)))

    print('Confusion matrix for {}:'.format(model_name))
    print(confusion_matrix(y_test, result))
    print('Accuracy for {}: {}'.format(model_name, model.score(X_test, y_test)))
    print ('F1 Score for {}: {}'.format(model_name, f1_score(y_test, result)))
    # Save model
    filename = '{}_model.pkl'.format(model_name)
    pickle.dump(model, open(filename, 'wb'))
    
    # Load Example
    #loaded_model = pickle.load(open(filename, 'rb'))
    #vocabulary = pickle.load(open('feature_vocab.pkl', 'rb'))
    #vectorizer_test = TfidfVectorizer(stop_words='english', max_features=3000, vocabulary=vocabulary)
    # test_feature = vectorizer_test.fit_transform(['Test'])
    #result = loaded_model.predict(test_feature)
    # print(result)
