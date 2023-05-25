# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:08:29 2021
@author: Junyuan Zhao
email: Junyuan.Zhao@mpi.nl
"""

import nltk
import json
import os
import pandas as pd
from collections import Counter
from nltk.util import bigrams


## split corpus into sentences
with open("corpus.txt","r",encoding="utf-8") as file:
    corpus_str = file.read()
    file.close()
split_str = corpus_str.split("\n")


# get bigram distribution
counter_sum = Counter()
i = 1

while i < len(split_str):
    seg = split_str[i:i+1000] # calculate bigram by slice, just to know the progress
    sent = " ".join(seg) # concatenate individual sentences
    bigram = bigrams(sent) # get bigram for current slice
    freq = nltk.FreqDist(bigram) # generate freq count
    counter_curr = Counter(freq) # convert freq to a format that can be summed
    counter_sum = counter_curr + counter_sum # add current slice to total bigram count
    i += 1001
    print(i)

bigram_dict = dict(counter_sum)
with open("bigram_CH.json","w") as output:
    json.dump(bigram_dict,output)

df_freq = pd.DataFrame(bigram_dict.items(), columns=['word', 'frequency'])
df_freq["word_str"] = df_freq['word'].apply(str)
df_freq["word_str"] = df_freq["word_str"].str.slice(start=2,stop=10,step=5)
df_freq.drop('word', axis=1, inplace=True)
df_freq.to_csv("bigram_CH.csv")


# annotate TP for each word


stimlist = pd.read_excel(
    'W:\shared\Junyuan\Thesis\Experiment\Stimuli\pilot_mpi\wordlist_pilot_mpi.xlsx',
    sheet_name="V-AdvMix"
    )
tps = []
c = 0
for word in stimlist["phraseCH"]:  
    print(word)
    loc_word = df_freq["word_str"].str.contains(word,na=False)
    if loc_word.sum() != 0:
        loc_all = df_freq["word_str"].str.contains("^"+word[0],na=False) # regex start with
        freq_word = int(df_freq[loc_word]["frequency"])
        freq_all = df_freq[loc_all]["frequency"].sum()
        tp = freq_word/freq_all        
        tps.append(tp)
        print(tp)
    else:
        tps.append(0)
        print("not found")
    c += 1
    print(c)
df = pd.DataFrame()
df["tp"] = tps
df.to_csv("VAdvAlt.csv")
