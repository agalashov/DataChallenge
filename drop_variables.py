# Import python modules
import numpy as np
import pandas as pd

def delUnUsefulFeatures(data, feat_set):
    for s in feat_set : 
        data.drop(s, axis=1, inplace=True);
    return data

def drop_variables(f_in_name, f_out_name):
    

    train_data = pd.read_csv(f_in_name, delimiter=";")
    

    
    
    train_data = train_data[ train_data["ASS_SOC_MERE"] == "Entity1 France"]

    unuseful_features_1 = [ "ASS_BEGIN", "ASS_END", "ASS_COMENT"]
    unuseful_features_2 = [ "DAY_DS", "ACD_COD", "ACD_LIB", "ASS_SOC_MERE"]
    

    
    train_data = delUnUsefulFeatures(train_data, unuseful_features_1)
    train_data = delUnUsefulFeatures(train_data, unuseful_features_2)
    

    train_data.sort(["DATE"], ascending=True)
    temp_unuseful_features = ["TPER_HOUR", "DAY_WE_DS"]
    train_data = delUnUsefulFeatures(train_data, temp_unuseful_features)
    
    #train_data.to_csv(f_out_name, sep=";", index=False)
    
    return train_data