
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

#%% parameters

train_filename = "train_2011_2012.csv"
meteo_2011_filename = "meteo_2011.csv"
meteo_2012_filename = "meteo_2012.csv"

meteo_data_2011 = []
meteo_data_2012 = []
meteo_data_processed = []
i_date = 0
i_dept_nb = 1
i_city = 2
i_temp_1 = 3
i_temp_2 = 4
i_wind_dir = 5
i_precip = 6
i_pressure_hpa = 7
num_meteo_2011 = 0
num_meteo_2012 = 0


#%% Read Meteo Data 2011
meteo_data_2011 = pd.read_csv(meteo_2011_filename, header=1,names=['Date', 'Dep', 'City', 'Temp1', 'Temp2',  'Wind', 'Precip', 'Press' ])
#%% Read Meteo Data 2012
meteo_data_2012 = pd.read_csv(meteo_2012_filename, header=1,names=['Date', 'Dep', 'City', 'Temp1', 'Temp2',  'Wind', 'Precip', 'Press' ])
#%% Uniaque cities set
cities_1 = meteo_data_2011['City'].unique()
cities_2 = meteo_data_2012['City'].unique()

cities = Set(cities_1)
cities.update(Set(cities_2))
#%% Unique Department Set
departments = Set([ city for city in meteo_data_2011['Dep'].unique()])
departments.update(Set(meteo_data_2012['Dep'].unique() ) )
#%% Wind directions
wind_directions = Set(meteo_data_2011['Wind'].unique())
wind_directions.update(Set(meteo_data_2012['Wind'].unique()))



    
#%% Feature Selection
    
execfile("./feat_select/chiSQ.py")
execfile("./feat_select/infogain.py")


#
#

