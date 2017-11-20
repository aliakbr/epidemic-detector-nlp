import pandas as pd
import json
from os.path import isfile
from preprocess import preprocess

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

finaldatatrn['text'] = finaldatatrn['text'].apply(preprocess)
datacls['text'] = datacls['text'].apply(preprocess)

#%%

finaldatatrn.to_pickle('tweets_training.pkl')
datacls.to_pickle('tweets_to_classify.pkl')
