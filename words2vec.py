# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import numpy as np
from six.moves import xrange 
import tensorflow as tf
from data_utils import read_params
import pdb
import cPickle
data_index = 0
#read the whole dataset and related dictionary and mapping parameters
data = read_params('dictionary/data')
dict_word = read_params('dictionary/dict_word')
dict_characters = read_params('dictionary/dict_characters')
dict_components = read_params('dictionary/dict_components')
word2char = read_params('dictionary/word2char')
char2comp = read_params('dictionary/char2comp')
word_size = len(dict_word)
char_size = len(dict_characters)
comp_size = len(dict_components)
data_size = len(data)
#Function to generate a training batch for the skip-gram model.
def generate_skip_batch(batch_size, num_skips, skip_window):
	global data_index
	global word2char
	global char2comp
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window  # target label at the center of the buffer
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	#find the characters and sub components of the target labels
	char_labels = [[]for i in range(batch_size)] #each target word has variable number of characters
	comp_labels = [[] for i in range(batch_size)]
	for i in range(batch_size):
		tw = labels[i][0] #the index of target word
		char_labels[i] += word2char[tw] #get a list of characters associated with target word
		for char_id in char_labels[i]:
			sub_labels[i] += char2comp[char_id]
	return batch, labels,char_labels,comp_labels
#Function to generate a training sample for the CBOW model
def generate_cbow_batch(num_skips, skip_window):
	global data_index
	global word2char
	global char2comp
	assert num_skips <= 2 * skip_window
	word_input = np.ndarray(shape=(num_skips), dtype=np.int32)
	word_label = [[data[skip_window + data_index]]]
	char_input = []
	comp_input = []
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	target = skip_window
	targets_to_avoid = [skip_window]
	for i in range(num_skips):
		while target in targets_to_avoid:
			target = random.randint(0, span - 1)
		targets_to_avoid.append(target)
		word_input[i] = data[data_index + target]
		char_input += word2char[word_input[i]]
	for ch in char_input:
		comp_input += char2comp[ch]
	data_index = (data_index + 1)
	if (data_index >= data_size - span):
		data_index = 0
	return word_input,char_input,comp_input,word_label
def save_word_embeddings(fname,embeddings):
	f = open(fname,'w')
	cPickle.dump(embeddings, f)
	f.close()

#Hyper parameter settings
batch_size = 1      # batch size of SGD
embedding_size = 100  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 5
#test generate_cbow_batch function
print (data[0:10])
for i in range(10):
	word_input,char_input,comp_input,word_label = generate_cbow_batch(num_skips, skip_window)
	print('Generating iteration ',i)
	print('word input', word_input)
	print('char input', char_input)
	print('comp input', comp_input)
	print('word_label', word_label)
#Build the graph for skip gram model
'''
graph1 = tf.Graph()
with graph.as_default():
	# Input data.
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	tword_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	#for a given training example, the length of corresponding characters and components is variable
	tchar_labels = tf.placeholder(tf.int32)
	tcomp_labels = tf.placeholder(tf.int32)
	with tf.device('/cpu:0'):
		embeddings = tf.Variable(
			tf.random_uniform([word_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
		# Construct the variables for the NCE loss
		word_nce_weights = tf.Variable(
        	tf.truncated_normal([word_size, embedding_size],
        		stddev=1.0 / math.sqrt(embedding_size)))
		word_nce_biases = tf.Variable(tf.zeros([word_size]))
		char_nce_weights = tf.Variable(
			tf.truncated_normal([char_size, embedding_size],
				stddev=1.0 / math.sqrt(embedding_size)))
		char_nce_biases = tf.Variable(tf.zeros([char_size]))
		comp_nce_weights = tf.Variable(
			tf.truncated_normal([comp_size, embedding_size],
				stddev=1.0 / math.sqrt(embedding_size)))
		comp_nce_biases = tf.Variable(tf.zeros([comp_size]))
		#compute the loss of predicting the target word and its characters and components
		word_loss = tf.reduce_mean(
      	tf.nn.nce_loss(word_nce_weights, word_nce_biases, embed, tword_labels,
                     num_sampled, word_size))
		char_loss = tf.reduce_mean()
		comp_loss = 
		loss = word_loss + char_loss + comp_loss
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
		init = tf.initialize_all_variables()
		#normalize the final word embeddings
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
'''
#Build the graph for cbow model
graph2 = tf.Graph()
with graph2.as_default():
	#feed the data
	tword_input = tf.placeholder(tf.int32, shape = [num_skips])
	tchar_input = tf.placeholder(tf.int32)	#variable characters
	tcomp_input = tf.placeholder(tf.int32)  #variable component
	tword_label = tf.placeholder(tf.int32)  #one target label
	with tf.device('/cpu:0'):
		word_embeddings = tf.Variable(
			tf.random_uniform([word_size, embedding_size], -1.0, 1.0))
		char_embeddings = tf.Variable(
			tf.random_uniform([char_size, embedding_size],-1.0, 1.0))
		comp_embeddings = tf.Variable(
			tf.random_uniform([comp_size, embedding_size],-1.0, 1.0))
		word_embed = tf.nn.embedding_lookup(word_embeddings, tword_input)
		char_embed = tf.nn.embedding_lookup(char_embeddings,tchar_input)
		comp_embed = tf.nn.embedding_lookup(comp_embeddings,tcomp_input)
		word_context = tf.reduce_mean(word_embed,0)
		word_context = tf.reshape(word_context,[1,embedding_size])
		char_context = tf.reduce_mean(char_embed,0)
		char_context = tf.reshape(char_context,[1, embedding_size])
		comp_context = tf.reduce_mean(comp_embed,0)
		comp_context = tf.reshape(comp_context,[1, embedding_size])
		nce_weights = tf.Variable(
        	tf.truncated_normal([word_size, embedding_size],
        		stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([word_size]))
		loss1 = tf.reduce_mean(
			tf.nn.nce_loss(nce_weights, nce_biases, word_context, tword_label,num_sampled, word_size))
		loss2 = tf.reduce_mean(
			tf.nn.nce_loss(nce_weights, nce_biases, char_context, tword_label,num_sampled, word_size))
		loss3 = tf.reduce_mean(
			tf.nn.nce_loss(nce_weights, nce_biases, comp_context, tword_label,num_sampled, word_size))
		loss = loss1 + loss2 + loss3
		optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
		init = tf.initialize_all_variables()
		#normalize the final word embeddings
		norm = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
		normalized_embeddings = word_embeddings / norm
		saver = tf.train.Saver()
#Train the cbow model
num_steps = data_size * 5
print_steps = 1000
save_steps = 1000000
retrain = False
with tf.Session(graph=graph2) as session:
	init.run()
	print("Initialized")
	print('Num steps: ',num_steps)
	if retrain == True:
		saver.restore(session, 'model_param/model-74000000')
		print('Successfully load the parameters from model-74000000')
	average_loss = 0
	for step in xrange(num_steps):
		word_input,char_input,comp_input,word_label = generate_cbow_batch(num_skips, skip_window)
		feed_dict = {tword_input: word_input, tchar_input: char_input, 
			tcomp_input: comp_input, tword_label: word_label}
		_, loss_val, char_embed_val, char_context_eval, loss2_val, loss1_val = session.run([optimizer, loss, char_embed,char_context,loss2, loss1], feed_dict=feed_dict)
		average_loss += loss_val
		if step % print_steps == 0:
			if step > 0:
				average_loss /= print_steps
			print("Average loss at step ", step, ": ", average_loss)
			if math.isnan(average_loss):
				print('Input ',step, ' ', word_input,char_input,comp_input,word_label)
				print('char_embed ',char_embed_val, 'char_context ', char_context_eval, 'loss2 ', loss2_val, 'loss1 ', loss1_val)
				break
			average_loss = 0
		#save the model parameters
		if step % save_steps == 0:
			if step > 0:
				saver.save(session, 'model_param/model', global_step=step)
	#compute and save the final word embeddings
	word_embeddings = normalized_embeddings.eval()
	save_fname = 'result/word_embeddings.pk'
	save_word_embeddings(save_fname, word_embeddings)