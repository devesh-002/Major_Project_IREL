import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
from string import punctuation
import subprocess
from collections import Counter
from os.path import join as pjoin
import pandas as pd
from sklearn import preprocessing
import torch
# 用fairseq的时候就用multiprocessing
# from multiprocessing import Pool
from Stemmer import Stemmer

from others.logging import logger
# from others.tokenization import BertTokenizer
# fairseq环境去掉这句话。
from transformers import BertTokenizer, BasicTokenizer, AutoTokenizer

# from pytorch_transformers import BertTokenizer

# from others.utils import clean
# from others.rouge_not_a_wrapper import avg_rouge
# from utils import _get_word_ngrams
from indicnlp.tokenize import indic_tokenize
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
stopwords_english = set(stopwords.words('english'))
stemmer = Stemmer('porter')

stem_words = set()
with open('../hindi_stemwords.txt', 'r') as f:
    stem_words = {word.strip() for word in f}
stopword = open('../hindi_stopwords.txt', 'r')
stopword = set(stopword)
suffixes_gujarati = ['નાં', 'ના', 'ની', 'નો', 'નું', 'ને', 'થી', 'માં', 'એ', 'ઓ', 'ે', 'તા', 'તી', 'વા', 'મા', 'વું', 'વુ', 'ો', 'માંથી', 'શો', 'ીશ', 'ીશું', 'શે',
                     'તો', 'તું', 'તાં', '્યો', 'યો', 'યાં', '્યું', 'યું', '્યા', 'યા', '્યાં', 'સ્વી', 'રે', 'ં', 'મ્', 'મ્', 'ી', 'કો']
prefixes_gujarari = ['અ']
gujarati_stopwords = ['હે', 'છે', 'કે', 'જો', 'જી', 'ને', 'નાં', 'નું', 'ની', 'નો', 'તો', 'જો', 'લેતા', 'શા', 'હો', 'હોઈ', 'મા', 'બધું', 'મી', 'એન', 'તું', 'છો', 'છીએ', 'નં', 'એવો', 'હોવા', 'તેથી', 'નું', 'છ', 'એવા', 'એની', 'થતાં', 'જેવી', 'બંને', 'હશે', 'માં', 'ની', 'હતાં', 'તેવી', 'થયો', 'એવી', 'થી', 'થયું', 'ત્યાં', 'છતાં', 'તેઓ',
                      'તેમ', 'ને', 'તેને', 'હું', 'બાદ', 'શકે', 'જો', 'રહી', 'એમ', 'તેના', 'કરે', 'થઇ', 'સુધી', 'કોઈ', 'ના', 'હવે', 'તેની', 'ન', 'જે', 'તા', 'હોય', 'હતું', 'એ', 'કરી', 'તે', 'હતી', 'માટે', 'તો', 'જ', 'પણ', 'કે', 'આ', 'અને', 'અમે', 'તમે', 'રે', 'હે', 'હા', 'છે', 'કે', 'જો', 'લોલ', 'જી', 'ને', 'નાં', 'નું', 'ની', 'નો', 'તો', 'જો', 'વળી']


def tokenise_english(data):  # working
    data = processText(data)
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


def tokenizer_hindi(data):
    data = processText(data)
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
    data = data.lower()
    data = processText(data)
    clean = ''.join(ch if ch.isalnum() else ' ' for ch in data)
    clean = re.split(r'  ',clean)
    final = []
    punctuations = ('.', ',', '!', '?', '"', "'", '%', '#', '@', '&', '…')

    for w in clean:
        if w.endswith(punctuations):
            w = w[:-1]

        if w in gujarati_stopwords:
            continue

        for stm in suffixes_gujarati:
            if(w.endswith(stm)):
                w = w[:-len(stm)]
                break

        for prefix in prefixes_gujarari:
            if w.startswith(prefix):
                w = w.lstrip(prefix)
                break

        final.append(w)
    final=" ".join(final)

    return final

def read_file(file_path,language,train=False):
    data=''
    if(language=="english"):    
        df = pd.read_csv(r"data/eng_train.csv", encoding='utf-8')
        comment_words = ''
    
        for val in df["Summary"]:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
        df = pd.read_csv(r"data/eng_val.csv", encoding='utf-8')
        for val in df["Summary"]:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
      
            comment_words += " ".join(tokens)+" "
        data=tokenise_english(comment_words)

    elif(language=="hindi"):    
        df = pd.read_csv(r"data/hin_train.csv", encoding='utf-8')
        comment_words = ''
        for val in df["Summary"]:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
        df = pd.read_csv(r"data/hin_val.csv", encoding='utf-8')
        for val in df["Summary"]:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
      
            comment_words += " ".join(tokens)+" "
        data=tokenizer_hindi(comment_words)
    
    elif(language=="gujarati"):    
        df = pd.read_csv(r"data/guj_train.csv", encoding='utf-8')
        comment_words = ''
        for val in df["Summary"]:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            comment_words += " ".join(tokens)+" "
        df = pd.read_csv(r"data/guj_val.csv", encoding='utf-8')
        for val in df["Summary"]:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
      
        data=tokenizer_gujarati(comment_words)
    return data
# def convertToJson()

def finalRead(save_path,raw_path):

    dir_save=os.path.abspath(save_path)
    og_dir=os.path.abspath(raw_path)
    if not dir_save:
        os.mkdir(dir_save)
    
    languages=["eng","hin","guj"]
    dataset_spl=["test","train","valid"]

    for data_dp in dataset_spl:
        for lang in languages:
            assert os.path.isdir(os.path.join(og_dir,data_dp,lang))
        for data_sp in dataset_spl:
         for lan_sp in languages:
            stories_dir = os.path.join(og_dir, data_sp, lan_sp)
            tokenized_stories_dir = os.path.join(dir_save, data_sp, lan_sp)
            stories = os.listdir(stories_dir)

            # make IO list file
            print("Making list of files to tokenize...")
            with open("mapping_for_corenlp.txt", "w") as f:
                for s in stories:
                    if (not s.endswith('story') and not s.endswith('chnref')):
                        continue
                    f.write("%s\n" % (os.path.join(stories_dir, s)))

            if lan_sp == 'eng':
                data=read_file(None,"english")
                command=['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'ssplit',
                           '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                           'json', '-outputDirectory', data]
            subprocess.call(command)
            os.remove("mapping_for_corenlp.txt")

           