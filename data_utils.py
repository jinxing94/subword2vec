# -*- coding: UTF-8 -*-
'''
Data_utils
Preprocess the data:
1. Build the dictionaries mapping the words, characters, subcharacter components
to their indices.
2. Find the map between a word to its character, a character to its sub components.
3. Represent the text data, replace the words by their indices.

*Important* Please run data_utils with Python3 because Python2 doesn't support for
chinese text processing very well.
'''
import json
import pdb
import collections
def remove_duplicate(a):
	a = set(a)
	b = [x for x in a]
	return b
def build_dictionary(chinese_subchar):
	dictionary = dict()
	cnt = 0
	for subchar in chinese_subchar:
		dictionary[subchar] = cnt
		cnt += 1
	return dictionary
def char_map_subchar(data, dict_char, dict_subchar, subchar_name):
	char2subchar = [[] for i in range(len(dict_char))]
	for line in data:
		tmp_char = line['char']
		char_id = dict_char[tmp_char]
		tmp_subs = line[subchar_name]
		sub_ids = [dict_subchar[x] for x in tmp_subs]
		char2subchar[char_id] = sub_ids
	return char2subchar
def word_map_char(dict_word, dict_char):
	word2char = [[] for i in range(len(dict_word))]
	cnt1 = 0
	cnt2 = 0
	for word,index in dict_word.items():
		for x in word:
			if x in dict_char:
				word2char[index].append(dict_char[x])
				cnt1 += 1
			else:
				cnt2 += 1
	print('in char dictionary ', cnt1, 'not in char dictionary ', cnt2)
	return word2char
def save_word_notin_char(fname, dict_word, dict_char):
	f = open(fname,'w')
	uchar = []
	uword = []
	for word,index in dict_word.items():
		flag = False
		for x in word:
			if x not in dict_char:
				uchar.append(x)
				flag = True
		if(flag):
			uword.append(word)
	uchar = remove_duplicate(uchar)
	uword = remove_duplicate(uword)
	print('number of characters not in the dictionary ',len(uchar))
	print('number of words containing exceptional characters ',len(uword))
	f.write(' '.join(uchar))
	f.write('\n')
	f.write(' '.join(uword))
	f.close()
def read_subwords(fname):
	lines = []
	chinese_characters = []
	chinese_components = []
	chinese_strokes = []
	chinese_cangjie = []
	chinese_wubi98 = []
	chinese_radical = []
	with open(fname) as f:
		for line in f:
			lines.append(json.loads(line))
	for line in lines:
		chinese_characters.append(line['char'])
		chinese_components += [x for x in line['components']]
		chinese_strokes += [x for x in line['strokes']]
		chinese_cangjie += [x for x in line['cangjie']]
		chinese_wubi98 += [x for x in line['wubi98']]
		chinese_radical.append(line['radical'])
	chinese_characters = remove_duplicate(chinese_characters)
	chinese_components = remove_duplicate(chinese_components)
	chinese_strokes = remove_duplicate(chinese_strokes)
	chinese_cangjie = remove_duplicate(chinese_cangjie)
	chinese_wubi98 = remove_duplicate(chinese_wubi98)
	chinese_radical = remove_duplicate(chinese_radical)
	return lines,chinese_characters,chinese_components,chinese_strokes,chinese_cangjie,chinese_wubi98,chinese_radical
def read_corpus(fname):
	data = []
	with open(fname,'r') as f:
		for line in f:
			data += line.split()
	return data
def build_datasets(words, dict_char):
	count = [['UNK', -1]]
	#sort words frequency for neg sampling in nce 
	count.extend(collections.Counter(words).most_common()) 
	dictionary = dict()
	for word, cnt in count:
		if(cnt < 5 and cnt != -1):   #remove words frequency less than 5
			break
		flag = True             #remove words containing characters not in the dictionary
		for x in word:
			if x not in dict_char:
				flag = False
				break
		if(flag):
			dictionary[word] = len(dictionary)
	data = list()
	for word in words:
		if word in dictionary:
			index = dictionary[word]
			data.append(index)
	#reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, dictionary
def save_params(fname,paras):
	with open(fname,'w') as f:
		json.dump(paras, f)
def read_params(fname):
	f = open(fname,'r')
	data = json.load(f)
	f.close()
	return data
def process_dataset(corpus_file, sub_file):
	lines,chinese_characters,chinese_components,chinese_strokes,chinese_cangjie,chinese_wubi98,\
		chinese_radical = read_subwords(sub_file)
	#build the dictionary
	dict_characters = build_dictionary(chinese_characters)
	dict_components = build_dictionary(chinese_components)
	char2comp = char_map_subchar(lines, dict_characters, dict_components, 'components')
	words = read_corpus(corpus_file)
	print('Data size', len(words))
	data,dict_word = build_datasets(words, dict_characters)
	print ('Dict size',len(dict_word))
	del(words)
	#build the maps
	word2char = word_map_char(dict_word, dict_characters)
	# save the parameters 
	save_params('dictionary/data', data)
	save_params('dictionary/dict_word', dict_word)
	save_params('dictionary/dict_characters', dict_characters)
	save_params('dictionary/dict_components', dict_components)
	save_params('dictionary/word2char', word2char)
	save_params('dictionary/char2comp', char2comp)
	#find the characters in the corpurs that are not contained in the dictionary
	save_word_notin_char('dictionary/char_exception.txt', dict_word, dict_characters)
if  __name__ == '__main__':	
	#process the dataset
	corpus_file = 'datasets/zh_wiki_01_segment_post'
	sub_file = 'datasets/items_httpcn_6.jl'
	process_dataset(corpus_file, sub_file)