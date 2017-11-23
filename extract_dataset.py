import preprocess
import pandas as pd

filename = 'dataset_related.csv'
output = 'data_related_extracted/data_related_extracted_remstop.txt'

df = pd.read_csv(filename)
df_extract = df.loc[:, ['tweet', 'class']]
df_extract = df_extract.dropna()
df_extract = df_extract.drop_duplicates()

tweets = df_extract.tweet.values.tolist()
classes = df_extract['class'].values.tolist()

# Preprocess
for i in range(len(tweets)):
    tweets[i] = tweets[i].replace('\n', ' ')
    tweets[i] = preprocess.preprocess(tweets[i])
    tweets[i] = preprocess.remove_punc(tweets[i])
    tweets[i] = preprocess.lemmatize(tweets[i])
    tweets[i] = preprocess.remove_stopwords(tweets[i])
    
s = []
for tweet,cl in zip(tweets,classes):
    s.append(tweet + '\t' + str(int(cl)) + '\n')

with open(output, 'wb') as f:
    for x in s:
        f.write(x.encode('utf-8'))