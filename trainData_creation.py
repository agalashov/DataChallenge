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
#%%
a = pd.read_csv("clear_data.csv", delimiter=",", nrows = 10)
print (a)

#a.sort(["DATE"], ascending=True)
##%% for instant K = 0
#def trainData_creation(data, K, T):
#    
#    Y = data["CSPL_RECEIVED_CALLS"]
#    
#    # 2012-01-03 00:00:00.000
#
#    
#    return train_data, targets;