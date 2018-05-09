import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
stemmer = LancasterStemmer()
training_data = []
training_data.append({"class":"technology", "sentence":"us technology companies founded by st and nd generation immigrants apple google amazon facebook oracle ibm uber airbnb yahoo intel emc ebay spacex vmware at amp tesla nvidia qualcomm paypal adp reddit slackhq wework stripe cognizant intuit"})
training_data.append({"class":"technology", "sentence":"hechnologies of next decade ai machine learning iot blockchain printing mobile autonomous cars mobile internet robotics vr ar wireless power nanotechnology voice ui vpas"})
training_data.append({"class":"technology", "sentence":"trademarks when you are thinking about ico it is important to begin with protecting your intellectual property surrounding your technology that is subject matter of your upcoming ico Trademark protection is one of the forms of ip protection"})
training_data.append({"class":"technology", "sentence":"the blockchain is a disruptive and revolutionary technology for most industries in the future like google apple tencent and netease has already gain huge profit from game industry traditional game had monopoly on game industry for many years"})
training_data.append({"class":"technology", "sentence":"we re hiring cnbc is seeking a technology reporter to expand and improve our coverage of facebook twitter snap communications platforms and apps and related subjects including privacy and silicon valley culture"})

training_data.append({"class":"sports", "sentence":"most playoff wins by teammates in nba historytony parker manu ginobilitim duncan tony parkertim duncan manu ginobilikobe bryant derek fisher michael jordan scottie pippen"})
training_data.append({"class":"sports", "sentence":"the pelicans nba win the series and advance to the second round with a victory over trailblazers nop takes it behind huge performances from anthony davis pts reb, blk and jrue holiday pts ast rajon rondo: ast nbaplayoffs"})
training_data.append({"class":"sports", "sentence":"me playing weekend league would really love to see a revamp of this game mode on fifa less games more time more precise matchmaking and improved connectivity actually also really miss some daylight but not shadows"})
training_data.append({"class":"sports", "sentence":"no texas shuts out no baylor this afternoon in waco horns earn a first round bye at this weeks big championships at texas tennis center horns will take a few days off and play the big semifinals on saturday"})
training_data.append({"class":"sports", "sentence":"more sundays until the seasons since no offense in the nfl has produced more tds of plus yards than the oakland raiders raiders steelers chargers chiefs falcons redskins"})

training_data.append({"class":"politics", "sentence":"james comeys memos are classified i did not declassify them they belong to our government therefore he broke the law additionally he totally made up many of the things he said i said and he is already a proven liar and leaker where are memos on clinton lynch others"})
training_data.append({"class":"politics", "sentence":"one of the biggest and most destructive misconceptions in current politics is that the diehard sanders left progressives far from it on the nra guns on immigration on environmental justice on race sanders has glaring weakneses hardly a standard bearer"})
training_data.append({"class":"politics", "sentence":"i know bipartisanship is possible i know we can return to a time where our individual politics didn’t define us to each other a time when we were all americans even when we disagreed i know it’s possible and we will do it again patriots let our past leaders lead the way"})
training_data.append({"class":"politics", "sentence":"while congressional republicans refuse to protect dreamers the trump administration is targeting them for deportation democrats are committed to protecting our country dreamers"})
training_data.append({"class":"politics", "sentence":"chairman rep joe crowley the gop tax scam is a set up so republicans can go after social security medicare medicaid and we’re not going to let them do that because have children we have veterans we have students and we have working families that need a real break"})
print ("%s sentences in training data" % len(training_data))
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
    # add to documents in our corpus
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
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

print ("# words", len(words))
print ("# classes", len(classes))
# sample training/output
i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print (training[i])
print (output[i])
import numpy as np
import time

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

# ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("error after "+str(j)+" epochs:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "weight_and_bias.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=15, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
# probability threshold
ERROR_THRESHOLD = 0
# load our calculated synapse values
synapse_file = 'weight_and_bias.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s\nclassification: %s\n" % (sentence, return_results))
    return return_results

classify("President Emmanuel Macron of France departs on Monday for a state visit to the United States, having hinted to the French public that by giving President Trump the benefit of the doubt, he could get something in return")
classify("Cryptocurrency is far from dead in China. In fact, the restrictive measures there may have inadvertently triggered a wave of innovation that targets some of the problems faced by cryptocurrencies everywhere, not just in China.")
classify("When a great player gets going in this league, you got a problem.The GameTime crew talks Bojan Bogdanovic's hot streak and Pacers' confidence heading into Game 4.")

while(True):
    try: 
        inputted_Tweet = input("Enter a tweet you would like classified: ")
        classify(inputted_Tweet)
    except:
        print("Invalid Entry, try again\n")
