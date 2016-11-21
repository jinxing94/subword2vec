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
import threading
import time
import sys
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
def generate_cbow_batch_thread(num_skips, skip_window,sample_index):
	global word2char
	global char2comp
	assert num_skips <= 2 * skip_window
	word_input = np.ndarray(shape=(num_skips), dtype=np.int32)
	word_label = [[data[skip_window + sample_index]]]
	char_input = []
	comp_input = []
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	target = skip_window
	targets_to_avoid = [skip_window]
	for i in range(num_skips):
		while target in targets_to_avoid:
			target = random.randint(0, span - 1)
		targets_to_avoid.append(target)
		word_input[i] = data[sample_index + target]
		char_input += word2char[word_input[i]]
	for ch in char_input:
		comp_input += char2comp[ch]
	return word_input, char_input,comp_input,word_label
def save_word_embeddings(fname,embeddings):
	f = open(fname,'w')
	cPickle.dump(embeddings, f)
	f.close()
#Hyper parameter settings
batch_size = 1      # batch size of SGD
embedding_size = 100  # Dimension of the embedding vector.
skip_window = 5       # How many words to consider left and right.
num_skips = 10         # How many times to reuse an input to generate a label.
num_sampled = 10
#test generate_cbow_batch function
'''
print (data[0:10])
for i in range(10):
	word_input,char_input,comp_input,word_label = generate_cbow_batch(num_skips, skip_window)
	print('Generating iteration ',i)
	print('word input', word_input)
	print('char input', char_input)
	print('comp input', comp_input)
	print('word_label', word_label)
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
		sum_context = word_context + char_context + comp_context
		nce_weights = tf.Variable(
			tf.truncated_normal([word_size, embedding_size],
				stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([word_size]))
		loss = tf.reduce_mean(
			tf.nn.nce_loss(nce_weights, nce_biases, sum_context, tword_label,num_sampled, word_size))
		global_step = tf.Variable(0, name="global_step",trainable=False)
		optimizer = tf.train.GradientDescentOptimizer(0.025).minimize(loss,global_step=global_step)
		init = tf.initialize_all_variables()
		#normalize the final word embeddings
		norm = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
		normalized_embeddings = word_embeddings / norm
		saver = tf.train.Saver()
#Train the cbow model
epoch_num = 3  #use the whole training dataset three times
num_steps = data_size * epoch_num
print_steps = 10000
save_steps = 1000000
retrain = False
thread_num = 1
session = tf.Session(graph=graph2)
def train_thread(pid, start_index, end_index):
	global session
	for sample_index in xrange(start_index, end_index):
		word_input,char_input,comp_input,word_label = generate_cbow_batch_thread(num_skips, skip_window, sample_index)
		feed_dict = {tword_input: word_input, tchar_input: char_input, 
			tcomp_input: comp_input, tword_label: word_label}
		loss_val,step,_ = session.run([loss,global_step,optimizer], feed_dict=feed_dict)
		#use the thread info to print the log and save the model at proper time
		if step > 0 and step % print_steps == 0:
			localtime = time.asctime(time.localtime(time.time()))
			print('time: ',localtime,' Thread ',pid, ' step ', step, ' loss ',loss_val)
			sys.stdout.flush()
		if step > 0 and step % save_steps == 0:
			saver.save(session, 'model_param/model', global_step=step)
init.run(session=session)
print("Initialized")
print('Num steps: ',num_steps)
if retrain == True:
	saver.restore(session, 'model_param/model-74000000')
	print('Successfully load the parameters from model-74000000')
span = 2 * skip_window + 1  # [ skip_window target skip_window ]
max_index = data_size - span
thread_index = [int(math.floor(max_index / thread_num * i)) for i in xrange(thread_num + 1)]
print('thread_index ',thread_index)
for epoch in range(epoch_num):
	print('Training epoch ',epoch)
	workers = []
	for i in range(thread_num):
		t = threading.Thread(target=train_thread,args=(i,thread_index[i],thread_index[i+1]))
		t.start()
		workers.append(t)
	for t in workers:
		t.join()
#compute and save the final word embeddings
word_embeddings = normalized_embeddings.eval(session=session)
save_fname = 'result/word_embeddings.pk2'
save_word_embeddings(save_fname, word_embeddings)
session.close()