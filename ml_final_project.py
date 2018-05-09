import tweepy
from tweepy import OAuthHandler
from twython import Twython
import json
# have to install tweet preprocessor: pip install tweet-preprocessor
import preprocessor as p
import re
import nltk
nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
import os
import datetime

import numpy as np
import time

stemmer = LancasterStemmer()

consumer_key = 'efOxpT68s5lujsROCz3Ql6TO9'
consumer_secret = 'eFTnoHPK5mxyhZeSucXzxhfyhZjOa0tigSNR1uCLPOq6nvtrit'
access_token = '955211543825743872-cs2gPusj9t5AYpC3sJdhN5MIj4P2FUb'
access_secret = 'ZXeWRunPFJkyRL0PAKVfDvEOyv8URBo2COT1eE8KmuNvi'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

t = Twython(app_key=consumer_key,
            app_secret=consumer_secret,
            oauth_token=access_token,
            oauth_token_secret=access_secret)

# the above code we need for anything running tweepy

print("")


def tag_to_tweet(topic):

    hashtag = "#" + str(topic)
    # print(hashtag)

    search = t.search(q=hashtag, count=1000)

    results = search['statuses']

    # characters to also be removed from tweet to clean it
    # punc_marks = list('.,?!<>/:;{}[]()-_+=|\@^~`$%&*"')
    # punc_marks.append("'")
    alphabet = list('abcdefghi jklmnopqrstuvwxyz')

    # items to be cleaned from tweet
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)

    # create a list for the tweets
    tweets = list()

    for tweet in results:
        if tweet['text'].find("RT ") == -1:
            tweets.append(p.clean(tweet['text']))

    # list of tweets that have been cleaned
    clean_tweets = list()

    ignored_words = ['time', 'person', 'year', 'way', 'day', 'thing',
                     'man', 'world', 'life', 'hand', 'part', 'child', 'eye',
                     'week','case', 'point', 'number', 'group', 'problem', 'fact', 'to', 'of',
                     'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
                     'over', 'after', 'the', 'and', 'a', 'that', 'i', 'it', 'not', 'he', 'as',
                     'you', 'this', 'but', 'his', 'they', 'her', 'she', 'or', 'an', 'will',
                     'my', 'one', 'all', 'would', 'there', 'their']

    for tweet in tweets:
        tweet_list = list(tweet)
        words = tweet.split(' ')
        for word in words:
            if word in ignored_words:
                tweet.replace(word, '')
        for char in tweet_list:
            char = char.lower()
            if char not in alphabet:
                try:
                    tweet_list[tweet_list.index(char)] = ' '
                except ValueError:
                    print('Invalid character: ' + char)
        clean_tweets.append(''.join(tweet_list))

        # for tweet in clean_tweets:
        # print(tweet + "\n")

    return clean_tweets


training_data = []
# sports
sports_data = tag_to_tweet('sports')
for sentence in sports_data:
    training_data.append({"class":"sports", "sentence":sentence})

sports_data = tag_to_tweet('soccer')
for sentence in sports_data:
    training_data.append({"class":"sports", "sentence":sentence})

sports_data = tag_to_tweet('football')
for sentence in sports_data:
    training_data.append({"class":"sports", "sentence":sentence})

sports_data = tag_to_tweet('baseball')
for sentence in sports_data:
    training_data.append({"class":"sports", "sentence":sentence})

# music
music_data = tag_to_tweet('music')
for sentence in music_data:
    training_data.append({"class":"music", "sentence":sentence})

music_data = tag_to_tweet('studio')
for sentence in music_data:
    training_data.append({"class":"music", "sentence":sentence})

music_data = tag_to_tweet('spotify')
for sentence in music_data:
    training_data.append({"class":"music", "sentence":sentence})

music_data = tag_to_tweet('soundcloud')
for sentence in music_data:
    training_data.append({"class":"music", "sentence":sentence})

# tech
tech_data = tag_to_tweet('tech')
for sentence in tech_data:
    training_data.append({"class":"tech", "sentence":sentence})

tech_data = tag_to_tweet('ios')
for sentence in tech_data:
    training_data.append({"class":"tech", "sentence":sentence})

tech_data = tag_to_tweet('android')
for sentence in tech_data:
    training_data.append({"class":"tech", "sentence":sentence})

tech_data = tag_to_tweet('google')
for sentence in tech_data:
    training_data.append({"class":"tech", "sentence":sentence})

# politics
politics_data = tag_to_tweet('politics')
for sentence in politics_data:
    training_data.append({"class":"politics", "sentence":sentence})

politics_data = tag_to_tweet('congress')
for sentence in politics_data:
    training_data.append({"class":"politics", "sentence":sentence})

politics_data = tag_to_tweet('republican')
for sentence in politics_data:
    training_data.append({"class":"politics", "sentence":sentence})

politics_data = tag_to_tweet('democrat')
for sentence in politics_data:
    training_data.append({"class":"politics", "sentence":sentence})

# television
tv_data = tag_to_tweet('tvaudition')
for sentence in tv_data:
    training_data.append({"class":"tv", "sentence":sentence})

tv_data = tag_to_tweet('television')
for sentence in tv_data:
    training_data.append({"class":"tv", "sentence":sentence})

tv_data = tag_to_tweet('sitcom')
for sentence in tv_data:
    training_data.append({"class":"tv", "sentence":sentence})

tv_data = tag_to_tweet('film')
for sentence in tv_data:
    training_data.append({"class":"tv", "sentence":sentence})

print(training_data)
print("")


words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our dictionary
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for item in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = item[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(item[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
print([stemmer.stem(word.lower()) for word in w])
print(training[i])
print(output[i])


def clean_sentence(sentence):
    # tokenize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_info=False):
    # tokenize the pattern
    sentence_words = clean_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_info:
                    print ("found in bag_of_words: %s" % w)

    return np.array(bag)


def think(sentence, show_info=False):
    x = bow(sentence.lower(), words, show_info)
    if show_info:
        print ("sentence:", sentence, "\n bag_of_words:", x)
    input_ = x
    hid_ = 1/(1+np.exp(-(np.dot(input_, weights))))
    out_ = 1/(1+np.exp(-(np.dot(hid_, bias))))
    return out_


def train(X, y, neuron_count, eta, epochs):

    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    weights = 2*np.random.random((len(X[0]), neuron_count)) - 1
    bias = 2*np.random.random((neuron_count, len(classes))) - 1

    prev_weights_weight_update = np.zeros_like(weights)
    prev_bias_weight_update = np.zeros_like(bias)

    weights_direction_count = np.zeros_like(weights)
    bias_direction_count = np.zeros_like(bias)

    for j in iter(range(epochs+1)):

        # Feed forward through hidden layers 0, 1, and 2
        in_layer = X
        hid_layer = 1/(1+np.exp(-(np.dot(in_layer, weights))))

        out_layer = 1/(1+np.exp(-(np.dot(hid_layer, bias))))

        out_delta = (y - out_layer) * out_layer * (1 - out_layer)

        hid_delta = out_delta.dot(bias.T) * hid_layer * (1 - hid_layer)

        bias_weight_update = (hid_layer.T.dot(out_delta))
        weights_weight_update = (in_layer.T.dot(hid_delta))

        if j > 0:
            weights_direction_count += np.abs(((weights_weight_update > 0)+0) - ((prev_weights_weight_update > 0) + 0))
            bias_direction_count += np.abs(((bias_weight_update > 0)+0) - ((prev_bias_weight_update > 0) + 0))

        bias += eta * bias_weight_update
        weights += eta * weights_weight_update

        prev_weights_weight_update = weights_weight_update
        prev_bias_weight_update = bias_weight_update

    now = datetime.datetime.now()

    # weights and biases dictionary
    weights_and_biases = {'weights': weights.tolist(), 'biases': bias.tolist(),
                          'datetime': now.strftime("%Y-%m-%d %H:%M"),
                          'words': words,
                          'classes': classes
                          }
    weights_and_biases_file = "weights_and_biases.json"

    with open(weights_and_biases_file, 'w') as outfile:
        json.dump(weights_and_biases, outfile, indent=4, sort_keys=True)
    print ("saved weights/biases to:", weights_and_biases_file)


X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, neuron_count=20, eta=0.1, epochs=10)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")


# probability threshold
ERROR_THRESHOLD = 0
# load our calculated weights and biases values
weights_and_biases_file = 'weights_and_biases.json'
with open(weights_and_biases_file) as data_file:
    weights_and_biases = json.load(data_file)
    weights = np.asarray(weights_and_biases['weights'])
    bias = np.asarray(weights_and_biases['biases'])


def classify(sentence, show_info=False):
    results = think(sentence, show_info)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results


def classify_user():
    # twitter user you want to pull data from
    username = input("Enter a username: ")
    category = input("Enter the primary user category: ")

    # number of tweets you want to check
    tweetCount = 50

    results = api.user_timeline(id=username, count=tweetCount, tweet_mode="extended")

    # characters to also be removed from tweet to clean it
    # punc_marks = list('.,?!<>/:;{}[]()-_+=|\@^~`$%&*"')
    # punc_marks.append("'")
    alphabet = list('abcdefghi jklmnopqrstuvwxyz')

    # items to be cleaned from tweet
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)

    # create a list for the tweets
    tweets = list()

    for tweet in results:
        if tweet.full_text.find("RT ") == -1:
            tweets.append(p.clean(tweet.full_text))

    # list of tweets that have been cleaned
    clean_tweets = list()

    ignored_words = ['time', 'person', 'year', 'way', 'day', 'thing',
                     'man', 'world', 'life', 'hand', 'part', 'child', 'eye',
                     'week','case', 'point', 'number', 'group', 'problem', 'fact', 'to', 'of',
                     'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
                     'over', 'after', 'the', 'and', 'a', 'that', 'i', 'it', 'not', 'he', 'as',
                     'you', 'this', 'but', 'his', 'they', 'her', 'she', 'or', 'an', 'will',
                     'my', 'one', 'all', 'would', 'there', 'their']

    for tweet in tweets:
        tweet_list = list(tweet)
        words = tweet.split(' ')
        for word in words:
            if word in ignored_words:
                tweet.replace(word, '')
        for char in tweet_list:
            if char.lower() not in alphabet:
                tweet_list[tweet_list.index(char)] = ' '
        clean_tweets.append(''.join(tweet_list))

    return clean_tweets, category.lower()


print("")
while (True):
    classify_tweets, category = classify_user()
    if category == 'exit':
        break;
    print(category)
    correct = 0
    # for loop that loops through all tweets and provides accuracy
    for tweety in classify_tweets:
        classify(tweety)
        if classify(tweety)[0][0] == category:
            correct += 1
        print('')

    print("Accuracy: " + str(round((correct/30) * 100, 2)) + "%")

