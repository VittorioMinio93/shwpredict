# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:36:57 2022

@author: Utente
"""
#Importing packages 
import swhpredict.learning_model.core as lm
import swhpredict.data_processing .core as dp


#Insert the name of configuration file 
filename='configfile.ini'


#Processed RMS and sea wave time series from raw data
# df_seism, df_sea, org_shape=dp.PrepareLocalDataset(filename,save_data=True)


# Prepares the dataset for Machine learning process 
clean_df_seism, clean_df_sea, colmask, org_shape, colname, idx_sks, lmbda, alpha=lm.PrepareDataset(filename,df_seism, df_sea, org_shape, earth_check=True,save_data=True)


# Learns predictive model 
#model,scaler_x,scaler_y=lm.LearningModel(filename,clean_df_seism, clean_df_sea, colmask, org_shape,colname,CrossVal=True,save_model=True)
