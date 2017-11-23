import preprocess
import pandas as pd

filename = 'tweets_flu.csv'
output = 'test_rempunc.txt'

df = pd.read_csv(filename)
df_extract = df.loc[:, 'text']
df_extract = df_extract.dropna()
df_extract = df_extract.drop_duplicates()

tweets = df_extract.tolist()

# Preprocess
for i in range(len(tweets)):
    tweets[i] = tweets[i].replace('\n', ' ')
    tweets[i] = preprocess.preprocess(tweets[i])
    
s = []
for tweet in tweets:
    s.append(tweet + '\n')

with open(output, 'wb') as f:
    for x in s:
        f.write(x.encode('utf-8'))