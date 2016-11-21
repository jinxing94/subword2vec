# -*- coding: UTF-8 -*-
'''
Random sample 30 words and show top-8 similar words 
int the embedding space for each of them
'''
from data_utils import read_params
import cPickle
import numpy as np
import random
import pdb
import sys
def show_similar_words(embed_file, dict_word_file):
	f1 = open(embed_file,'r')
	embeddings = cPickle.load(f1)
	f1.close()
	print('load the embeddings cPickle file')
	dict_word = read_params(dict_word_file)
	reverse_dict_word = dict(zip(dict_word.values(), dict_word.keys()))
	sample_num = 30
	top_k = 8
	word_size = embeddings.shape[0]
	for i in range(sample_num):
		index = random.randint(0, word_size - 1)
		sim = embeddings.dot(embeddings[index].T)
		nearest = (-sim).argsort()[1:top_k + 1]
		log_str = "Nearest to %s:" % reverse_dict_word[index]
		for k in range(top_k):
			close_word = reverse_dict_word[nearest[k]]
			log_str = "%s %s," % (log_str, close_word)
		print(log_str)
	while(True):
		w = raw_input('Please input a word: ').decode(sys.stdin.encoding) 
		if w == 'exit':
			break
		if w not in dict_word:
			print ('not in dictionary, input again!')
			pdb.set_trace()
			continue
		else:
			index = dict_word[w]
			sim = embeddings.dot(embeddings[index].T)
			nearest = (-sim).argsort()[1:top_k + 1]
			log_str = "Nearest to %s:" % reverse_dict_word[index]
			for k in range(top_k):
				close_word = reverse_dict_word[nearest[k]]
				log_str = "%s %s," % (log_str, close_word)
			print(log_str)
	while(True):
		w1 = raw_input('Please input a word: ').decode(sys.stdin.encoding)
		w2 = raw_input('Please input a word: ').decode(sys.stdin.encoding)
		if w1 not in dict_word or w2 not in dict_word:
			print('not in dictionary')
			continue
		else:
			if(w1 == 'a' or w2 == 'a'):
				break
			id1 = dict_word[w1]
			id2 = dict_word[w2]
			cosim = embeddings[id1].dot(embeddings[id2].T)
			print(w1,' ', w2,' ', cosim)
if  __name__ == '__main__':	
	embed_file = 'result/word_embeddings.pk'
	dict_word_file = 'dictionary/dict_word'
	show_similar_words(embed_file, dict_word_file)