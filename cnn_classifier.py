from __future__ import print_function

import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import optimizers

# set parameters:
max_features = 20000
maxlen = 150
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
data_train = "data_related_extracted/data_related_extracted_preprocess.txt"

print('Loading data...')
data_x, data_y = [], []
with open(data_train, 'r', encoding='utf8') as f:
    lines = f.readlines()

for line in lines:
    line = line.split('\t')
    train_text = line[0]
    train_label = int(line[1])
    data_x.append(train_text)
    data_y.append(train_label)

x_train, x_test = data_x[:-400], data_x[-400:]
y_train, y_test = data_y[:-400], data_y[-400:]

tk_train = Tokenizer(num_words=max_features)
tk_train.fit_on_texts(x_train)
train_dict_len = len(tk_train.word_index)
print("Train word dict length = %s" % train_dict_len)

print("Converting x_train words to integers..")
x_train_ohv = tk_train.texts_to_sequences(x_train)
x_test_ohv = tk_train.texts_to_sequences(x_test)

print('Pad sequences (samples x time)')
x_train_ohv = sequence.pad_sequences(x_train_ohv, maxlen=maxlen)
x_test_ohv = sequence.pad_sequences(x_test_ohv, maxlen=maxlen)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.fit(x_train_ohv, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

#Predict
preds = model.predict(x_test_ohv)
preds_res = preds[:]
preds_res[preds_res>=0.5] = 1
preds_res[preds_res<0.5] = 0

#Evaluation
sum_precision = 0.
for i in range(len(preds_res)):
    if (preds_res[i] == y_test[i]):
        sum_precision += 1
        
print("Accuracy test = {}".format(sum_precision/len(y_test)))

print("Saving model..")
    
model.save("cnn-train.hdf5")