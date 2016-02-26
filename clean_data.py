# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:41:20 2016

@author: galashovalexandr
"""


# Import python modules
import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sets import Set
import pandas as pd

#%% Clean data

data = pd.read_csv("train_2011_2012.csv", delimiter= ";")

#%%

clean_data = data[ data["ASS_SOC_MERE"] == "Entity1 France"]

#%%

clean_data.to_csv("clean_data.csv", index=False) 