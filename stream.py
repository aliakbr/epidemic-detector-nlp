import tweepy
import json
from pprint import pprint


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
        if (not decoded['retweeted']) and ('RT @' not in decoded['text']):
            #print('"%s"\n' % (decoded['text'].encode('ascii', 'ignore').replace(b'\n',b' ')))
            res = {
                'user_id' : decoded['user']['id_str'],
                'id_str': decoded['id_str'],
                'text': decoded['text'],
                'created_at': decoded['created_at'],
            }

            if decoded['coordinates']: res['coordinates'] = decoded['coordinates']
            if decoded['place']: res['place'] = decoded['place']
            if decoded['user']['location']: res['user_location'] = decoded['user']['location']

            # Also, we convert UTF-8 to ASCII ignoring all bad characters sent by users
            json.dump(res, self.handle)
            f.write('\n')

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
