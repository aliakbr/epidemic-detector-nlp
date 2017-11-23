import re, string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)+' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
html_regex = re.compile('<[^>]+>')
mention_regex = re.compile('(?:@[\w_]+)')
url_regex = re.compile('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
hashtag_regex = re.compile("(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s):
    tokens = tokenize(s)
    tokens = (token.lower() for token in tokens)
    tokens = (token for token in tokens if not html_regex.match(token))
    tokens = ('@user' if mention_regex.match(token) else token for token in tokens)
    tokens = ('!url' if url_regex.match(token) else token for token in tokens)
    s = ' '.join(t for t in tokens if t).replace('rt @user : ','')
    s = remove_punc(s)
    s = remove_stopwords(s)
    s = lemmatize(s)
    return s

def remove_punc(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    s = ' '.join(x for x in s.split())
    return s

def lemmatize(s):
    wordnet_lemmatizer = WordNetLemmatizer()
    s = ' '.join(wordnet_lemmatizer.lemmatize(x) for x in s.split())
    s = ' '.join(wordnet_lemmatizer.lemmatize(x, pos='v') for x in s.split())
    return s

def remove_stopwords(s):
    stop = set(stopwords.words('english'))
    s = ' '.join(x for x in s.split() if x not in stop)
    return s
