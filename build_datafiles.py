import pandas as pd
import json
from os.path import isfile
from preprocess import preprocess
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

tqdm.pandas()

#%%
tweets_data_fn = 'tweets_raw.pkl'

# Load scraped data
if not isfile(tweets_data_fn):
    with open('tweets_en_all.txt') as rf:
        datacls = pd.DataFrame((json.loads(line) for line in rf),
            columns=['user_id',
                     'id_str',
                     'text',
                     'created_at',
                     'coordinates',
                     'place',
                     'user_location']).set_index('id_str')
        datacls.to_pickle(tweets_data_fn)
else:
    datacls = pd.read_pickle(tweets_data_fn)

#%%

# Load training data

datatrn = pd.read_csv('dataset_related.csv', engine='python', dtype=object)
datatrn = datatrn.drop(datatrn.columns[0], axis=1).rename(columns={
    'id': 'id_str',
    'tweet': 'text'})
datatrn = datatrn[~datatrn['text'].isna() & ~datatrn['class'].isna()]
datatrn = datatrn.set_index('id_str')

#%%

# Expand training data

sampled = datacls.sample(4000).drop(columns=set(datacls.columns) - set(['text']))

finaldatatrn = datatrn.append(sampled)
finaldatatrn['class'].fillna('0', inplace=True)

#%%

finaldatatrn['text'] = finaldatatrn['text'].progress_apply(preprocess)
datacls['text'] = datacls['text'].progress_apply(preprocess)

# # Load model
# model_file_name = 'Support Vector Machine_model.pkl'
# loaded_model = pickle.load(open(model_file_name, 'rb'))
# vocabulary = pickle.load(open('feature_vocab.pkl', 'rb'))
# vectorizer_test = TfidfVectorizer(stop_words='english', max_features=3000, vocabulary=vocabulary)
# string_input = '..................'
# test_input = list(string_input)
# test_feature = vectorizer_test.fit_transform([test_input])
# result = loaded_model.predict(test_feature)

#%%

finaldatatrn.reset_index().to_hdf('tweets_training.h5', 'data')
datacls.reset_index().to_hdf('tweets_to_classify.h5', 'data')
