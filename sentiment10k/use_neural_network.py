from parse_data import create_feature_sets_and_labels
from parse_data import create_lexicon
import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100
hm_epochs = 12

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')


def neural_network(data):

    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
                    'biases': tf.Variable(tf.constant(0.1, shape=[n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


lexicon = create_lexicon('pos.txt', 'neg.txt')


def use_neural_network(input_data):
    prediction = neural_network(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
        if result[0] == 0:
            print('Positive:', input_data)
        elif result[0] == 1:
            print('Negative:', input_data)


use_neural_network("He's an idiot and a jerk and i hate his bad personality.")
use_neural_network("This ugly project was so difficult and bad.")
use_neural_network("The movie got so many negative reviews because the plot was so thin and not cool.")
use_neural_network("I found myself growing more and more frustrated during the movie.")
use_neural_network("The plot was a mess from the beginning, only a stupid person would like it.")

use_neural_network("The landscape I saw on that vacation the most beautiful in the country.")
use_neural_network("This was the best store i have ever visited in this beautiful city.")
use_neural_network("This match was the best match Kobe ever played for this great team.")
use_neural_network("If you want to eat something good, got to Pizza Hut, best pizza in town.")
use_neural_network("It was my pleasure to meet such a wonderful person.")