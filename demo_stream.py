import tweepy
import json
from pprint import pprint

from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

num_chars = 70
data_train = "data_related_extracted/data_related_extracted_rempunc.txt"
model_file = "cnn-train-char-related-rempunc.hdf5"

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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=42)

# Building char dictionary from x_train
tk_char = Tokenizer(filters='', char_level=True)
tk_char.fit_on_texts(x_train)
char_dict_len = len(tk_char.word_index)
print("Char dict length = %s" % char_dict_len)

# Authentication details. To  obtain these visit dev.twitter.com
consumer_key = 'qJTkb5wFWiSCLLuLWMx0B0Ghl'
consumer_secret = 'Dgwo6GQRWFH3CIwHaUiesfgJw4AloNkZIKXrZFgMcEBQt9kfBr'
access_token = '65892905-16sc7WovrkWGHXrcb1QPfyywr0qvsqeomk4R55WsR'
access_token_secret = '3u5aO1w1chaEhair50NVbh6BFsBEm68Am1D4RpRA98D4R'

# This is the listener, resposible for receiving data
class StdOutListener(tweepy.StreamListener):
    def __init__(self, handle):
        self.handle = handle

    def on_data(self, data):
        # Twitter returns data in JSON format - we need to decode it first
        decoded = json.loads(data)
        test_ohv = []
        test_ohv.append(sequence.pad_sequences(tk_char.texts_to_matrix(decoded['text']), maxlen=num_chars, padding='post', truncating='post'))
        test_ohv = sequence.pad_sequences(test_ohv, maxlen=150, padding='post', truncating='post')
        
        model = load_model(model_file)
        
        preds = model.predict(test_ohv)
        preds_res = preds[:]
        preds_res[preds_res>=0.5] = 1
        preds_res[preds_res<0.5] = 0
        
        if preds_res[0][0]: print("{} {}: {}\n".format(decoded['created_at'], decoded['user']['name'], decoded['text'])

        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    with open('tweets_en.txt', 'a') as f:
        l = StdOutListener(f)
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        stream = tweepy.Stream(auth, l)
        stream.sample(languages=["en"])
