# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:28:50 2017

@author: Ali-pc
"""
import os
import numpy as np
import pandas as pd


def get_tweets(root_dir):
    all_tweets_id = []
    all_tweets_class = []
    tweets_file = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for d in tweets_file:
        if d.endswith('.txt'):
            tweets_file = open(d)
            for l in tweets_file:
                value = l.strip('\n').split('\t')
                all_tweets_id.append(value[0])
                all_tweets_class.append(value[1])
    return all_tweets_id, all_tweets_class
                
                

# get all tweets
CONSUMER_KEY = 'YkzeiWTXY8kWpjpKa0kPrTYYd'
CONSUMER_SECRET = 'pp102MEm5nsGni6qJzuSaloAMjWiBMVhC5IbyilffLCkZN0rT9'
ACCESS_TOKEN = '2574387926-bwfPiD9PwR1DLbcZaCnqfCdyRZJbbhPlpEFsax2'
ACCESS_TOKEN_SECRET = 'YpgPlVlEhtokcLRVdm67IqGCOde0YDMqlSqwTLqM6F353'

root_dir = 'flu_annotations'
all_tweets_id, all_tweets_class = get_tweets(root_dir)
import tweepy
#api = twitter.Api(consumer_key=[CONSUMER_KEY],
#                  consumer_secret=[CONSUMER_SECRET],
#                  access_token_key=[ACCESS_TOKEN],
#                  access_token_secret=[ACCESS_TOKEN_SECRET])

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

all_tweets_status = []
for tweet_id in all_tweets_id:
    try:
        status = api.get_status(tweet_id)
    except:
        continue
    all_tweets_status.append(status.text)

# Save Tweet
df = pd.DataFrame()
df['id'] = np.asarray(all_tweets_id)
df['tweet'] = np.asarray(all_tweets_status)
df['class'] = np.asarray(all_tweets_class)
df.to_csv('dataset.csv')