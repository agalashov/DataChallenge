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
#categorical_features = [ "TPER_TEAM", "ASS_ASSIGNMENT"]
categorical_features = [ "TPER_TEAM"]
for s in categorical_features :
    train_data = toCategorical(train_data, s)
    train_data = train_data.drop(s, axis=1)
#%%

train_data = train_data.drop(train_data.columns[8:71],axis=1);
train_data = train_data.drop(train_data.columns[9:13],axis=1);



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
#%%

train_data = new_features(train_data)
train_data.to_csv("clear_data.csv", sep=";", index=False)
#%%
temp_data = train_data[:100011]

#%%%  TESTING

initial_date = "2011-01-01 00:00:00.000"
initial_date = pd.to_datetime(initial_date)

K = 24

first_possible_date = initial_date + pd.offsets.Hour(K)
first_date = initial_date + pd.offsets.Hour(5)
last_date = first_date + pd.offsets.Hour(9)

past_date = first_date - pd.offsets.Hour(K);

lok = int ( (first_date - past_date).total_seconds() / 60 / 30 )
    
prev_data = temp_data[ temp_data["DATE"] >= str(past_date)]
    
prev_data = prev_data[ prev_data["DATE"] < str(first_date)]

new_data = temp_data[ temp_data["DATE"] < str(last_date)]

new_data = new_data[ new_data["DATE"] > str(first_date)]

names = [ ("past_"+str(i+1)) for i in range(lok)]

for name in names:
    new_data[name] = 0
 

row = next(new_data.iterrows())[1]
first_date = row["DATE"]
prev_data = prev_data.drop(["DAY_OFF"],axis=1)
prev_data = prev_data.drop(["WEEK_END"],axis=1)
prev_data = prev_data.drop(["TPER_TEAM_Jours"],axis=1)
prev_data = prev_data.drop(["TPER_TEAM_Nuit"],axis=1)
dates = prev_data["DATE"].unique()

assignments = temp_data ["ASS_ASSIGNMENT"].unique()

for i in range(len(dates)):
    for ass in assignments : 
        prev_data[ (prev_data["DATE"]==dates[0]) & (prev_data["ASS_ASSIGNMENT"]==ass)  ]["CSPL_RECEIVED_CALLS"].values

        

#%% 

initial_date = "2011-01-01 00:00:00.000"
initial_date = pd.to_datetime(initial_date)

def create_model(data, K, first_date, last_date):
    #first_date = pd.to_datetime(last_moment)
    #last_date = pd.to_datetime(first_date)
    first_possible_date = initial_date + pd.offsets.Day(K)
    if (first_date < first_possible_date) :
        print("error")
        return;
    
    
    past_date = first_date - pd.offsets.Day(K);
    
    
    
    
    prev_data = data[ data["DATE"] > str(past_date)]
    
    prev_data = prev_data[ prev_data["DATE"] < str(first_date)]
    prev_data = prev_data.as_matrix()[:,len(data.columns)-1]
    

    new_attrib_num = prev_data.shape[0]
    print("dada")
    print(new_attrib_num)
    print(prev_data.shape)

    temp_data = data[ data["DATE"] < str(last_date)]

    temp_data = temp_data[ temp_data["DATE"] > str(first_date)]

    labels = temp_data["CSPL_RECEIVED_CALLS"].as_matrix()
    
    temp_data = temp_data.drop(["CSPL_RECEIVED_CALLS"],axis=1)
    temp_data = temp_data.drop(["DATE"], axis = 1)
    
    temp_data = toCategorical(temp_data, "ASS_ASSIGNMENT" )
    temp_data = temp_data.drop("ASS_ASSIGNMENT", axis=1)
    temp_data = temp_data.as_matrix()
    
    
    
    print(past_date)
    print(last_date)
    print(first_date)
    print((last_date - first_date).total_seconds())

    n = temp_data.shape[0]
    #n = int ( (last_date - first_date).total_seconds() /60 / 30)
    
    print( "n ",n)

    # new_attrib_num = int ( float(K * 24 * 60)  / 30)
    

    ## 2 because 1 it is the date, and 1 is the number of received calls which goes to labels
    m = len(data.columns) - 2 + new_attrib_num;
    print("axyel")
    print(prev_data.shape)
    print(new_attrib_num)
    
    X = np.zeros((n,m))
    
    print("concat")
    print(temp_data.shape)
    print(n)

    if new_attrib_num != 0 :
        additional_data = np.zeros((n,new_attrib_num))
        additional_data[0,:] = prev_data[:] 
        
        print("OK")
        
        X = np.hstack((temp_data, additional_data))
    else : 
        X = temp_data
    
    print("we are here")
    m1 = m - new_attrib_num
    print(m)
    print(m1)
    print(X.shape)
    
    if m1 != m :
#        X[1:n,m1:m] = np.hstack((X[1:n,(m1+1):(m)], labels[0:n-1]))
        X[1:n,m1:m] = X[0:n-1, (m1+1):m]
    
    return X, labels;

#%% Test of this function

td = (initial_date + pd.offsets.Day(5) - initial_date)
sec = td.total_seconds() / 60 / 30
int(sec)

first_date = initial_date + pd.offsets.Day(2)
last_date = first_date + pd.offsets.Day(5)

#%% Works with K = 0, but not with others
X,labels = create_model(temp_data, 0, first_date, last_date);