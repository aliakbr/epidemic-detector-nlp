import tweepy
import json
from pprint import pprint

from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import preprocess

num_chars = 70
data_train = "data_related_extracted/data_related_extracted_rempunc.txt"
model_file = "cnn-train-char-related-rempunc.hdf5"

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("praecantatio-f846b-firebase-adminsdk-cj3ro-b847906b6b.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://praecantatio-f846b.firebaseio.com/'})

import googlemaps

gmcl = googlemaps.Client(key='AIzaSyAZ2xVPyJm_mUMj6Roz2BVn1vfLwrYwToM')

def calc_coords(place):
    lat, lon = 0, 0
    count = 0
    if place != place or not place or not place['bounding_box'] or 'coordinates' not in place['bounding_box']:
        return None
    for lt, ln in place['bounding_box']['coordinates'][0]:
        count += 1
        lat += lt
        lon += ln

    return lat/count, lon/count

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

model = load_model(model_file)

# Authentication details. To  obtain these visit dev.twitter.com
consumer_key = 'qJTkb5wFWiSCLLuLWMx0B0Ghl'
consumer_secret = 'Dgwo6GQRWFH3CIwHaUiesfgJw4AloNkZIKXrZFgMcEBQt9kfBr'
access_token = '65892905-16sc7WovrkWGHXrcb1QPfyywr0qvsqeomk4R55WsR'
access_token_secret = '3u5aO1w1chaEhair50NVbh6BFsBEm68Am1D4RpRA98D4R'

# This is the listener, resposible for receiving data
class StdOutListener(tweepy.StreamListener):
    def __init__(self, handle):
        self.logfile = handle

    def on_data(self, data):
        # Twitter returns data in JSON format - we need to decode it first
        decoded = json.loads(data)
        test_ohv = []
        tweettext = decoded['text']
        tweettext = tweettext.replace('\n', ' ')
        tweettext = preprocess.preprocess(tweettext)
        tweettext = preprocess.remove_punc(tweettext)
        test_ohv.append(sequence.pad_sequences(tk_char.texts_to_matrix(tweettext), maxlen=num_chars, padding='post', truncating='post'))
        test_ohv = sequence.pad_sequences(test_ohv, maxlen=150, padding='post', truncating='post')

        preds = model.predict(test_ohv)
        prediction_rate = preds[0][0]

        sent = False

        if prediction_rate >= 0.5:
            c = None
            cm = ''
            if decoded['coordinates']:
                cm = 'coords'
                c = decoded['coordinates']['coordinates']
            elif decoded['place']:
                cm = 'place'
                c = calc_coords(decoded['place'])
            elif decoded['user']['location']:
                cm = 'ulsearch'
                try:
                    res = gmcl.places(decoded['user']['location'])
                    if res['status'] == 'OK':
                        dc = res['results'][0]['geometry']['location']
                        c = (dc['lng'], dc['lat'])
                except:
                    pass

            if c:
                sent = True

                ref = db.reference('flumap')
                ref.push({
                    'username': decoded['user']['screen_name'],
                    'text': decoded['text'],
                    'lat': c[1],
                    'lng': c[0]
                })

                print("{} {} {}: {}".format(cm, decoded['created_at'], decoded['user']['name'], decoded['text']).encode('ascii', errors='ignore'))
            else:
                print("nc {} {}: {}".format(decoded['created_at'], decoded['user']['name'], decoded['text']).encode('ascii', errors='ignore'))

        self.logfile.write('{} - {} - {} - {}\n'.format(prediction_rate, sent, decoded['id_str'], decoded['text']))

        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    with open('stream_class.log', 'a', encoding='utf-8') as lf:
        l = StdOutListener(lf)
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        stream = tweepy.Stream(auth, l)
        print('Starting listener...')
        stream.filter(languages=["en"], track=['flu', 'influenza'])
