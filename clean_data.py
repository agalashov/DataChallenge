#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:41:20 2016

@author: galashovalexandr
"""


# Import python modules
import numpy as np
import pandas as pd
from drop_variables import drop_variables

train_data = drop_variables("train_2011_2012.csv", "clear_data.csv")


#train_data = pd.read_csv("clear_data.csv", delimiter= ";")

#%% Binarisation of Categories

def toCategorical(data, catName):
    dummies = pd.get_dummies(data[catName]).rename(columns=lambda x: str(catName)+"_" + str(x))
    data = pd.concat( [data, dummies], axis = 1)
    data.drop([catName], axis=1)
    return data

#%% Categorical Features
categorical_features = [ "TPER_TEAM", "ASS_ASSIGNMENT"]
for s in categorical_features :
    train_data = toCategorical(train_data, s)
    train_data = train_data.drop(s, axis=1)
#%%

train_data = train_data.drop(train_data.columns[9 : 72], axis=1);

train_data = train_data.drop(train_data.columns[10:14],axis=1);


train_data = train_data.drop(["TPER_TEAM"], axis=1);

train_data = train_data.sort(["DATE"], ascending=True)

train_data.to_csv("clear_data.csv", sep=";", index=False)

#%%

def new_features(data) :
    
    to_mean_columns = ["SPLIT_COD", "ASS_DIRECTORSHIP", "ASS_PARTNER", "ASS_POLE"]
    data = data.drop(to_mean_columns,axis=1)

    def allbut(*names):
        names = set(names)
        return [item for item in data.columns if item not in names]
    
    
    data = data.groupby(by=allbut("CSPL_RECEIVED_CALLS"),as_index=False).sum();
    
    return data;
    
temp_data = new_features(train_data[:100000])


#%% Creation of the model

initial_date = "2011-01-01 00:00:00.000"
initial_date = pd.to_datetime(initial_date)
ok = initial_date + pd.offsets.Day(1) 
ok


initial_date = "2011-01-01 00:00:00.000"
initial_date = pd.to_datetime(initial_date)

def create_model(data, K, first_date, last_date):
    #first_date = pd.to_datetime(last_moment)
    #last_date = pd.to_datetime(first_date)
    first_possible_date = initial_date + pd.offsets.Day(K)
    if (first_date < first_possible_date) :
        print("error")
        return;
    
    
    past_data = first_date - pd.offsets.Day(K);
    ## previous data from first_date - K till first_date
    prev_data = data[ data["DATE"] < str(past_data)].as_matrix()[:,len(data.columns)-1]

    temp_data = data[ data["DATE"] < str(last_date)]

    temp_data = temp_data[ temp_data["DATE"] > str(first_date)]
    
    labels = temp_data["CSPL_RECEIVED_CALLS"].as_matrix()
    
    temp_data = temp_data.drop(["CSPL_RECEIVED_CALLS"],axis=1)
    temp_data = temp_data.drop(["DATE"], axis=1)
    
    temp_data = temp_data.as_matrix()

    print(last_date)
    print(first_date)
    print((last_date - first_date).total_seconds())

    n = int ( (last_date - first_date).total_seconds() /60 / 30)
    
    print( "n ",n)

    new_attrib_num = int ( float(K * 24 * 60)  / 30)
    

    ## 2 because 1 it is the date, and 1 is the number of received calls which goes to labels
    m = len(data.columns) - 2 + new_attrib_num;
    print("axyel")
    print(prev_data.shape)
    print(new_attrib_num)
    
    X = np.zeros((n,m))

    if new_attrib_num != 0 :
        additional_data = np.zeros((n,new_attrib_num))
        additional_data[0,:] = prev_data[:] 
        
        X = np.concatenate(temp_data, additional_data)
    
    
    m1 = m - new_attrib_num
    print(m)
    print(m1)
    if m1 != m :
        X[1:n,m1:m] = np.concatenate(X[1:n,(m1+1):(m)], labels[0:n-1])
    
    return X, labels;

td = (initial_date + pd.offsets.Day(5) - initial_date)
sec = td.total_seconds() / 60 / 30
int(sec)

first_date = initial_date + pd.offsets.Day(2)
last_date = first_date + pd.offsets.Day(5)

## For instant doesn't work with K different from 0
X,labels = create_model(temp_data, 0, first_date, last_date);