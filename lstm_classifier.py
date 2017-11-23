from __future__ import print_function

import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, concatenate
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Model
from keras import optimizers

# set parameters:
max_features = 20000
maxlen1 = 40
maxlen2 = 10
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
data_train = "train-cleansed.tsv"
data_test_file = "testing-solusi.tsv"

print('Loading data...')
data_y = []
data_x = []
with open(data_train, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\t')
        train_label = line[-1]
        data_text = (line[2] + "\t" + line[1]).lower()
        data_x.append(data_text)
        data_y.append(train_label)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=42)

x_train_title = [x.split('\t')[0] for x in x_train]
x_train_text = [x.split('\t')[1] for x in x_train]
x_test_title = [x.split('\t')[0] for x in x_test]
x_test_text = [x.split('\t')[1] for x in x_test]

tk_train = Tokenizer(num_words=max_features)
tk_train.fit_on_texts(x_train)
train_dict_len = len(tk_train.word_index)
print("Train word dict length = %s" % train_dict_len)

print("Converting x_train words to integers..")
x_train_title_ohv = tk_train.texts_to_sequences(x_train_title)
x_test_title_ohv = tk_train.texts_to_sequences(x_test_title)
x_train_text_ohv = tk_train.texts_to_sequences(x_train_text)
x_test_text_ohv = tk_train.texts_to_sequences(x_test_text)

print('Pad sequences (samples x time)')
x_train_title_ohv = sequence.pad_sequences(x_train_title_ohv, maxlen=maxlen2, truncating='post')
x_test_title_ohv = sequence.pad_sequences(x_test_title_ohv, maxlen=maxlen2, truncating='post')
x_train_text_ohv = sequence.pad_sequences(x_train_text_ohv, maxlen=maxlen1, truncating='post')
x_test_text_ohv = sequence.pad_sequences(x_test_text_ohv, maxlen=maxlen1, truncating='post')

tk_word = Tokenizer()
tk_word.fit_on_texts(y_train)
word_dict_len = len(tk_word.word_index)
print("Word dict length = %s" % word_dict_len)

print("Converting y_train to vector of class..")
y_train_v = tk_word.texts_to_matrix(y_train)
y_test_v = tk_word.texts_to_matrix(y_test)

print('Pad sequences (samples y time)')
y_train_v = sequence.pad_sequences(y_train_v, maxlen=word_dict_len)
y_test_v = sequence.pad_sequences(y_test_v, maxlen=word_dict_len)

print('Build model...')
main_input1 = Input(shape=(maxlen1,), dtype='int32', name='main_input1')

x = Embedding(output_dim=512, input_dim=max_features, input_length=maxlen1)(main_input1)

lstm_out1 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(x)

main_input2 = Input(shape=(maxlen2,), dtype='int32', name='main_input2')

y = Embedding(output_dim=64, input_dim=max_features, input_length=maxlen2)(main_input2)

lstm_out2 = LSTM(8, dropout=0.2, recurrent_dropout=0.2)(y)

z = concatenate([lstm_out1, lstm_out2])

z = Dense(64, activation='relu')(z)

main_output = Dense(12, activation='softmax', name='main_output')(z)

model = Model(inputs=[main_input1, main_input2], outputs=[main_output])

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.fit([x_train_text_ohv, x_train_title_ohv], y_train_v,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

#Predict
preds = model.predict([x_test_text_ohv, x_test_title_ohv])

#Evaluation
sum_precision = 0.
for i, pred in enumerate(preds):
    pred_idx = np.argsort(pred)
    if y_test_v[i][pred_idx[-1]]:
        sum_precision += 1
        
print("Accuracy test = {}".format(sum_precision/len(y_test_v)))

print("Saving model..")

model.save("lstm-train.hdf5")

#Testing
data_test = []
data_test_id = []
with open(data_test_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\t')
        test_text = (line[2] + "\t" + line[1]).lower()
        data_test_id.append(line[0])
        data_test.append(test_text)
     
data_test_title = [x.split('\t')[0] for x in data_test]
data_test_text = [x.split('\t')[1] for x in data_test]

data_test_text_ohv = tk_train.texts_to_sequences(data_test_text)
data_test_title_ohv = tk_train.texts_to_sequences(data_test_title)
data_test_text_ohv = sequence.pad_sequences(data_test_text_ohv, maxlen=maxlen1, truncating='post')
data_test_title_ohv = sequence.pad_sequences(data_test_title_ohv, maxlen=maxlen2, truncating='post')

label_index = {}
for x in tk_word.word_index:
	label_index[tk_word.word_index[x]] = x
print(label_index)

preds_test = model.predict([data_test_text_ohv, data_test_title_ohv])

filename = "out-complex.txt"
with open(filename, 'w') as f:
    for id, pred in enumerate(preds_test):
        print(id)
        predsort = np.argsort(pred)
        f.write("{}:{}\n".format(data_test_id[id], label_index[predsort[-1]+1]))