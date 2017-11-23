from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

num_chars = 70
data_train = "data_self_extracted/data_self_extracted_rempunc.txt"
test_file = "test_rempunc.txt"
model_file = "cnn-train-char-self-rempunc.hdf5"

print('Loading data...')
data_x, data_y, data_test = [], [], []
with open(data_train, 'r', encoding='utf8') as f:
    lines = f.readlines()

for line in lines:
    line = line.split('\t')
    train_text = line[0]
    train_label = int(line[1])
    data_x.append(train_text)
    data_y.append(train_label)
    
with open(test_file, 'r', encoding='utf8') as f:
    lines = f.readlines()

for line in lines:
    train_text = line[0]
    data_test.append(train_text)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=42)

# Building char dictionary from x_train
tk_char = Tokenizer(filters='', char_level=True)
tk_char.fit_on_texts(x_train)
char_dict_len = len(tk_char.word_index)
print("Char dict length = %s" % char_dict_len)

print("Converting data_test to one-hot vectors..")
test_ohv = []
test_len = len(data_test)
i = 1
for x in data_test:
	if (i % 1000 == 0) or (i == test_len): print("%s of %s" % (i, test_len))
	i += 1
	test_ohv.append(sequence.pad_sequences(tk_char.texts_to_matrix(x), maxlen=num_chars, padding='post', truncating='post'))
print("Add padding to make 150*char_dict_len matrix..")
test_ohv = sequence.pad_sequences(test_ohv, maxlen=150, padding='post', truncating='post')

model = load_model(model_file)

preds = model.predict(test_ohv)
preds_res = preds[:]
preds_res[preds_res>=0.5] = 1
preds_res[preds_res<0.5] = 0

with open('res_self.txt', 'w') as f:
    for pred in preds_res:
        f.write("{}\n".format(int(pred[0])))