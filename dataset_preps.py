# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:28:50 2017

@author: Ali-pc
"""
import os
import sys
import numpy as np
import pandas as pd


def get_tweets(root_dir, keyword):
    all_tweets_id = []
    all_tweets_class = []
    tweets_file = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for d in tweets_file:
        if d.endswith('.txt') and keyword.lower() in d.lower():
            tweets_file = open(d)
            for l in tweets_file:
                value = l.strip('\n').split('\t')
                all_tweets_id.append(value[0])
                all_tweets_class.append(value[1])
    return all_tweets_id, all_tweets_class

# get tweets from file
root_dir = 'flu_annotations'
keyword = sys.argv[1]
all_tweets_id, all_tweets_class = get_tweets(root_dir, keyword)

# get tweets
import tweepy
CONSUMER_KEY = 'YkzeiWTXY8kWpjpKa0kPrTYYd'
CONSUMER_SECRET = 'pp102MEm5nsGni6qJzuSaloAMjWiBMVhC5IbyilffLCkZN0rT9'
ACCESS_TOKEN = '2574387926-bwfPiD9PwR1DLbcZaCnqfCdyRZJbbhPlpEFsax2'
ACCESS_TOKEN_SECRET = 'YpgPlVlEhtokcLRVdm67IqGCOde0YDMqlSqwTLqM6F353'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

all_tweets_status = []
count = 0
limit = 100000
print('Count of tweets:', len(all_tweets_id))
for tweet_id in all_tweets_id:
    try:
        status = api.get_status(tweet_id)
        all_tweets_status.append(status.text)
    except:
        all_tweets_status.append('')
    count += 1
    print(count)
    if count == limit: break

# Save Tweet
filename = 'dataset_'+keyword+'.csv'
df = pd.DataFrame()
df['id'] = all_tweets_id[:count]
df['tweet'] = all_tweets_status[:count]
df['class'] = all_tweets_class[:count]
df.to_csv(filename, encoding='utf-8')
