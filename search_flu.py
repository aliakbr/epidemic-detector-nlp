import tweepy
import json
from tweepy import OAuthHandler
import sys
import signal

# Checkpoint init
cp = 0
time = ""
max_id = None

# Read since_id for continuing fetching tweets

consumer_key = 'qJTkb5wFWiSCLLuLWMx0B0Ghl'
consumer_secret = 'Dgwo6GQRWFH3CIwHaUiesfgJw4AloNkZIKXrZFgMcEBQt9kfBr'
access_token = '65892905-16sc7WovrkWGHXrcb1QPfyywr0qvsqeomk4R55WsR'
access_secret = '3u5aO1w1chaEhair50NVbh6BFsBEm68Am1D4RpRA98D4R'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

# Authorization & wait if twitter API rate limit exceeds
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Fetch tweets
with open('flu.txt', 'ab') as f:
    for tweet in tweepy.Cursor(api.search, q='flu OR influenza', max_id=max_id, since="2011-09-10", until="2017-11-24", lang="en").items():
        #Process a single status
        tweet = tweet._json
        res = {
        	'user_id' : tweet['user']['id_str'],
        	'id_str': tweet['id_str'],
        	'text': tweet['text'],
        	'created_at': tweet['created_at']
        }
        print(res)

        if tweet['coordinates']: res['coordinates'] = tweet['coordinates']
        if tweet['place']: res['place'] = tweet['place']
        if tweet['user']['location']: res['user_location'] = tweet['user']['location']

        # Also, we convert UTF-8 to ASCII ignoring all bad characters sent by users
        json.dump(res, f)
        f.write('\n')
