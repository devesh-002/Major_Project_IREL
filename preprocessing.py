import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin
import pandas as pd
from sklearn import preprocessing
import torch
# 用fairseq的时候就用multiprocessing
from multiprocessing import Pool
from Stemmer import Stemmer

from others.logging import logger
# from others.tokenization import BertTokenizer
# fairseq环境去掉这句话。
from transformers import BertTokenizer, BasicTokenizer, AutoTokenizer

# from pytorch_transformers import BertTokenizer

from others.utils import clean
from others.rouge_not_a_wrapper import avg_rouge
# from utils import _get_word_ngrams
from indicnlp.tokenize import indic_tokenize
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
stopwords_english=set(stopwords)
stemmer = Stemmer('porter')

df = pd.read_csv(r"data/hin_train.csv", encoding='utf-8')
stem_words = set()
with open('../hindi_stemwords.txt', 'r') as f:
    stem_words = {word.strip() for word in f}
stopword = open('../hindi_stopwords.txt', 'r')
stopword=set(stopword)
suffixes_gujarati = ['નાં','ના','ની','નો','નું','ને','થી','માં','એ','ઓ','ે','તા','તી','વા','મા','વું','વુ','ો','માંથી','શો','ીશ','ીશું','શે',
			'તો','તું','તાં','્યો','યો','યાં','્યું','યું','્યા','યા','્યાં','સ્વી','રે','ં','મ્','મ્','ી','કો']
prefixes = ['અ']

def tokenise_english(data):  # working
        data=processText(data)
        toks = re.split(r"[^A-Za-z0-9]+", data)
        finaal = list()
        for i in toks:
            word = stemmer.stemWord(i)
            if (
                len(word) <= 1 or len(word) > 45 or word in stopwords_english
            ):  # check for word length
                continue
            finaal.append(word)

        return finaal

def processText(text):
    text = text.lower()
    text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)
    text = re.sub('@[^s]+', '', text)
    text = re.sub('[s]+', ' ', text)
    text = re.sub(r'#([^s]+)', r'1', text)
    text = text.strip('""')
    return text

def tokenizer_hindi( data):
        data=processText(data)
        data = re.sub(r'http[s]?\S*[\s | \n]', r' ', data)  # removing urls
        data = re.sub(r'\{.?\}|\[.?\]|\=\=.*?\=\=', ' ', data)
        clean = ''.join(ch if ch.isalnum() else ' ' for ch in data)
        clean = clean.split()

        final = []

        for w in clean:
            if w in stopword or w in stopwords_english or len(w) > 45 or len(w) <= 1:
                continue
            else:
                for stm in stem_words:
                    if(w.endswith(stm)):
                        w = w[:-len(stm)]
                final.append(w)
        return final
def tokenizer_gujarati(data):
    data=data.lower()
    data=processText(data)
    
    