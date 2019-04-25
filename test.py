#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:01:28 2019

@author: williamlopez
"""

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix

import pandas as pd
import numpy as np 

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import matplotlib.pyplot as plt

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint

nlp = en_core_web_sm.load()

L = pd.read_csv("lyrics.csv", index_col=0)

RapL = L.loc[L['genre'] == 'Hip-Hop']
RockL = L.loc[L['genre'] == 'Rock']

def textArtist(s):
    lyrics=""
    for ind,val in L.iterrows():
        if val["artist"]==s:
            lyrics = lyrics + str(val["lyrics"])
    return lyrics

lyrics = textArtist('arcade-fire')
lyrics = lyrics.replace('\n',' ')
lyrics = lyrics.lower()




doc = nlp(lyrics)
pprint([(X.text, X.label_) for X in doc.ents])

items = [x.text for x in doc.ents]
Counter(items).most_common(3)

# %%
