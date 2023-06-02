# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:55:59 2022

@author: Utente
"""
#Importing main packages 

import logging
import numpy as np
import pickle
import configparser
import ast
import pandas as pd
from scipy import stats
from sklearn.preprocessing import  MinMaxScaler
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score
from obspy.core import UTCDateTime

   
def ReadConfig(filename):
    '''
    Reads each input paramaters stored in the configuration file

            Parameters:
                    filename (str): name of the configuration file 
            
            Returns:
                network (str): list of the network codes
                list_sta (list): list of station codes 
                list_chan (list): list of channel codes
                lat1, lat2, lon1, lon2 (float): area of interest used to learn the model 
                rms_thr (float): threshold of ambiguous RMS values
                row_thr (int):  threshold of the number of null values contained in the seismic dataset
                skew_limit (float):skewness values used to evaluate and correct the symmetry of the dataset
                percentage (float): percentage of the dataset used only for the training step (betweeen 0 and 1)
                nchunks (int): number of chunks used to split the time series 
                kfolds (int): number of datasets for Cross-Validation
                RF_max_depth (int): maximum depth of the tree 
                RF_n_estimator (int): number of trees in the forest
                RF_max_features (int): number of features to consider when looking for the best split
                KNN_n_neighbors (int): number of neighbors to use by default for kneighbors queries
                KNN_weights (str): weight function used in prediction.
                LGB_n_estimators (int): number of boosted trees to fit
                LGB_learning_rates (float): boosting learning rate
                LGB_max_depth (float): maximum tree depth for base learners     
                LGB_num_leaves (int): maximum tree leaves for base learners
                model_flag (str): algorithm used to learn the model 
                file_out (str): name of the .pickle file used to store the pre-processed seismic/sea dataset
                file_out_err (str): name of the .pickle file used to store the results of validation model
                folder_save (str): directory in which it saves the output
                filemodel (str): name of the file containing the predictive model
                eqCatalog (str): link to real-time earthquakes 
                MagMed (float): magnitude threshold for regional earthquakes
                MagWorld (float): magnitude threshold for worldwide earthquakes
                mLat, MLat, mLon, MLon (float): area for regional earthquakes in decimaal degrees 
                starttime, endtime (obj): temporal limits in UTC (default value is “1970-01-01T00:00:00.0Z”) for earthquakes extracting 
                hours_del (float): semi-interval in hours used to delete data affected by earthquakes 
    '''
    try:
        #Reading the configuration file
        config_obj = configparser.ConfigParser()
        config_obj.read(filename)
        
        #Reading model and network sections from configuration file:
        net_info = config_obj["Network"]
        model_info = config_obj["Model"]
        
        #Reading infomation about the seismic network used
        network=ast.literal_eval(net_info['network'])
        list_sta=ast.literal_eval(net_info['stations'])
        list_chan=ast.literal_eval(net_info['channels'])        
        starttime=UTCDateTime(net_info['starttime'])
        endtime=UTCDateTime(net_info['endtime'])
 
        #Reading all paramaters for learning the model 
        lat1=float(model_info['lat1'])
        lat2=float(model_info['lat2'])
        lon1=float(model_info['lon1'])
        lon2=float(model_info['lon2'])
        rms_thr=float(model_info['rms_thr'])
        row_thr=int(model_info['row_thr'])
        skew_limit = float(model_info['skew_limit'])
        percentage =float(model_info['percentage'])
        nchunks=int(model_info['nchunks'])
        kfolds=int(model_info['kfolds'])
        RF_max_depth=int(model_info['RF_max_depth'])
        RF_n_estimator=int(model_info['RF_n_estimators'])
        RF_max_features=int(model_info['RF_max_features'])
        KNN_n_neighbors=int(model_info['KNN_n_neighbors'])
        KNN_weights= model_info['KNN_weights']
        LGB_n_estimators=int(model_info['LGB_n_estimators'])
        LGB_learning_rates=float(model_info['LGB_learning_rate'])
        LGB_max_depth =float(model_info['LGB_max_depth'])       
        LGB_num_leaves=int(model_info['LGB_num_leaves'])
        model_flag=model_info['model_flag']
        file_out=model_info['file_out']
        file_out_err=model_info['file_out_err']
        folder_save=model_info['folder save']
        filemodel=model_info['filename']        
        eqCatalog=model_info['eqCatalog']
        MagMed=float(model_info['MagMed'])
        MagWorld=float(model_info['MagWorld'])
        mLat=float(model_info['eLat1'])
        MLat=float(model_info['eLat2'])
        mLon=float(model_info['eLon1'])
        MLon=float(model_info['eLon2'])
        hours_del = float(model_info['hours_del'])
        
        #Update log file about the completing of the operation 
        logging.info("OK!Configuration file was read.")
        return network, list_sta, list_chan, lat1, lat2,lon1, lon2, rms_thr, row_thr, skew_limit, percentage, nchunks, kfolds, RF_max_depth, RF_n_estimator, RF_max_features, KNN_n_neighbors, KNN_weights, LGB_n_estimators, LGB_learning_rates, LGB_max_depth, LGB_num_leaves, model_flag, file_out, file_out_err, folder_save, filemodel, eqCatalog, MagMed, MagWorld, mLat,MLat, mLon, MLon, starttime, endtime, hours_del
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Configuration file was not read.')

# ###############################################################################
# def ReadRmsData(seism_data, network, list_sta, list_chan):
#     '''
#     Reads RMS data from a local repository.
#     REMEMBER: this part of the code is temporary. It needs to adapt to the other package

#             Parameters:
#                 seism_data (str): directory containing RMS data in .mat files
#                 network (str): list of the network codes
#                 list_sta (list): list of station codes 
#                 list_chan (list): list of channel codes
            

#             Returns:
#                 df_seism (obj): dataframe containing RMS times series for each frequency band, station and channel 
                     
#     '''
    
#     try:
        
#         #Sorts stations and channels codes 
#         list_sta=sorted(list_sta)
#         list_chan=sorted(list_chan)
        
#         #Create a list of folder
#         list_fold=glob(seism_data+'/*[0-9]')
        
#         #Initialize an empty variable to storage all RMS data through rows
#         data_seism=None
        
#         for fld in list_fold:
#             #Create a list of matlab files containing RMS data 
#             list_rms=[glob(fld+'/**/*'+staz+ch[-1]+'*.mat',recursive=True) for staz in list_sta for ch in list_chan]
#             list_rms = list(np.concatenate(list_rms).flat)  
#             # list_rms=glob(seism_data+'/**/*.mat',recursive=True)
            
#             #Initialize an empty variable to storage all RMS data through columns
#             data_col=None
            
#             #Loop through the matlab files 
#             for file in list_rms:
                
#                 #Try for each file.Eventually it skips to the next iteration
#                 try:
              
#                     #Read a matlab file for a specific channel, station and for all frquency bands 
#                     data=loadmat(file,struct_as_record=False, squeeze_me=True)
                    
#                     #Extract RMS values and associated data 
#                     station = station = data['info_structure'].station #station name
#                     comp = data['info_structure'].component #channel name
#                     f_bands=data['info_structure'].f_bands #frequency bands used for RMS calculation 
#                     time_vector=data['time_vector'] #timing of RMS time series 
#                     RMS_main=data['RMS_main'] #matrix containing RMS for all frequency bands (samples X number of frequency bands)
                    
#                     #Create an empty dataframe with the correct header
#                     #In this case, the dataframe must have three levels (station,channel,frequency band)
#                     data_header = pd.DataFrame( [[station, comp, ''.join(map(str,np.round(f, 2)))] for f in f_bands],
#                                            columns=['station', 'component', 'freq'])
                    
#                     #Shift to approximate to the end of the hour
#                     #In this case, the RMS times series were hourly sampled 
#                     #but the time vector contains the middle points of the hourly
#                     #time windows (30 min). So, you needs to approximate to the end of the hourapproximate to the end of the hour
#                     fiveminutes = 5*60/86400
#                     times = pd.to_datetime(pd.Series(time_vector.T-719529+fiveminutes), unit='D', utc=True).dt.round('H')
                    
#                     #Inserts the RMS data in the empty dataframe 
#                     df = pd.DataFrame(data=RMS_main, columns = pd.MultiIndex.from_frame(data_header), index=times)
                    
#                     #Storage the results with the correct order through the columns 
#                     data_col = df if data_col is None else pd.concat([data_col, df], axis =1)
#                 except:
#                     continue
#             data_seism =  data_col if data_seism is None else pd.concat([ data_seism, data_col], axis=0)
#         #Update log file about the completing of the operation 
#         logging.info("OK!RMS data were read.")
#         return data_seism
        
#     except Exception:
#         #Update log file about the fall of the operation 
#         logging.warning('Warning!RMS data were not read.')

###############################################################################

# def ReadSeaData(sea_data,lat1, lat2,lon1, lon2):
#     '''
#     Reads sea wave data from a local repository.
#     REMEMBER: this part of the code is temporary. It needs to adapt to the other package

#             Parameters:
#                 sea_data (str): directory containing sea wave data in .mat files
#                 lat1, lat2, lon1, lon2 (float): area of interest used to learn the model                      

#             Returns:
#                 df_sea (obj): dataframe containing sea times series for each position of the area of interest  
#                 org_shape (tuples): shape of sea wave data matrix  
#     '''
    
#     try:
        
#         #Create a list of folder
#         list_sea=glob(sea_data+'/**/*.mat',recursive=True)
        
#         #Initialize an empty dictionary to storage all sea data through rows
#         data_sea = {'Y': None, 'time': None, 'lat':[], 'lon':[]}
        
#         #Iterates through the .mat files and concatenates the results 
#         #in the dataframes
#         for f in list_sea:
            
#             #Try for each .mat files. Eventually, it skips to the next iteration
#             try:   
#                 #Reads the sea state file (.mat file)
#                 f = h5py.File(f, 'r')
                
#                 if data_sea['Y'] is None:
#                     data_sea['Y'] = f['storage'][()]
#                 else:
#                     data_sea['Y'] = np.append(data_sea['Y'], f['storage'][()] , axis=0)
#                 if data_sea['time'] is None:
#                     data_sea['time'] = f['time_storage'][()]
#                 else:
#                     data_sea['time'] = np.append(data_sea['time'], (f['time_storage'][()]), axis =1 )
#                 # data_sea['bathym'] = f['bathym_xtr'][()]
#                 data_sea['lat'] = f['latitude_grid'][()]
#                 data_sea['lon'] = f['longitude_grid'][()]
#             except: 
#                 continue             
#         #Convert the time format into a user-friendly format 
#         #and rotates the sea state data matrixes (bathymetry, sea wave height, 
#         #latitude and longitude) of about 90°
#         data_sea['Y'] = np.array([np.rot90(m) for m in data_sea['Y']])
#         data_sea['time'] =  pd.to_datetime(pd.Series(data_sea['time'][0]-719529), unit='D', utc=True).dt.round('1min')
#         # data_sea['bathym'] = np.rot90(data_sea['bathym'])
#         data_sea['lat'] = np.rot90(data_sea['lat'])
#         data_sea['lon'] = np.rot90(data_sea['lon'])
        
        
#         #Cuts the area of interest in terms of latitude and longitude
#         LatRange = [ lat1,  lat2]
#         LonRange = [ lon1,  lon2]
        
#         #Search all rows and columns referring to the area of study 
#         #in terms of latitude and longitude 
#         rows, cols = np.where( (data_sea['lat']>=LatRange[0]) & (data_sea['lat']<=LatRange[1]) & (data_sea['lon']>=LonRange[0]) & (data_sea['lon']<=LonRange[1]))

#         mr = min(rows)
#         Mr = max(rows)
#         mc = min(cols)
#         Mc = max(cols)

#         #Filtering of the data 
#         data_sea['Y'] = data_sea['Y'][:, mr:Mr, mc:Mc]
#         data_sea['Y'][:, 0:2, 17:38] = np.nan # In this specific case there are some ambigous data, because they are in the land (They should be NaN)
#         data_sea['lat'] = data_sea['lat'][mr:Mr, mc:Mc]
#         data_sea['lon'] = data_sea['lon'][mr:Mr, mc:Mc]
#         # data_sea['bathym'] = data_sea['bathym'][mr:Mr, mc:Mc]
    
#         #Extract the shape of sea data  matrix.
#         #It's very important to plot the results.
#         org_shape=data_sea['Y'].shape
        
#         #Update log file about the completing of the operation 
#         logging.info("OK!Sea data were read.")
#         return data_sea, org_shape
        
#     except Exception:
#         #Update log file about the fall of the operation 
#         logging.warning('Warning!Sea data were not read.')

###############################################################################
def CleanSeaData(df_sea):
    '''
    Clean sea wave data to apply machine learning algorithm.

            Parameters:
               df_sea (obj): dataframe containing sea times series for each position of the area of interest  
               
            Returns:
                clean_df_sea (obj): cleaned dataframe containing sea times series for each position of the area of interest  
                colmask (obj): mask containing the null values refering to land (There are not sea wave date in the land)                          
    '''
    
    try:
        # PRE-PROCESSING TARGET DATA (sea state data)
        # Changes the shape of sea data matrix from 3D to 2D
        sea = np.reshape(df_sea['Y'], (df_sea['Y'].shape[0], df_sea['Y'].shape[1]*df_sea['Y'].shape[2]))
        
        #Create a mask of boolean values to delete null columns 
        colmask = ~np.isnan(sea).all(axis=0)
        
        #Use the column mask to filter data and 
        #to exclude columns contaning  NaN values 
        sea = sea[:, colmask]

        #Create another mask of boolean values to delete null rows
        rowmask = ~np.isnan(sea).any(axis=1)
        
        #Use the row mask to filter data and 
        #to exclude rows contaning  NaN values 
        sea = sea[rowmask, :]

        #Use the row mask to filter the time vector and 
        #to exclude rows contaning  NaN values
        seatime = df_sea['time'][rowmask]

        #Create a dataframe with cleaned sea data    
        clean_df_sea = pd.DataFrame(data=sea, index=seatime)

       
            
        #Update log file about the completing of the operation 
        logging.info("OK!Sea data were cleaned.")    
        return clean_df_sea, colmask    
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Sea data were not cleaned.')
        
###############################################################################
def CleanRmsData(df_seism,rms_thr,row_thr):
    '''
    Clean rms data to apply machine learning algorithm.

            Parameters:
                df_seism (obj): dataframe containing RMS times series for each frequency band, station and channel   
                rms_thr (float): threshold of ambiguous RMS values
                
            Returns:
                clean_df_seism (obj): cleaned dataframe containing RMS times series for each frequency band, station and channel   
                                          
    '''
    
    try:
        # PRE-PROCESSING FEATURES DATA (rms data):
        # delete wrong data
        df_seism.values[df_seism.values<rms_thr] = np.nan

        #Deletes all columns showing many null values. 
        #It uses a threshold to fix the mininum number of rows  that could
        #could be considered reliable
        df_seism.drop(columns=df_seism.columns[df_seism.isnull().sum(axis=0)>row_thr], inplace=True)

        #Interpolates the remainder null values 
        df_seism.interpolate(limit_direction = 'both', inplace=True)

        ##Create a dataframe with cleaned RMS data and the correct index  
        clean_df_seism = df_seism.copy()        
        clean_df_seism = clean_df_seism.T.reset_index(drop=True).T 
        
        colname= df_seism.columns
                    
        #Update log file about the completing of the operation 
        logging.info("OK!RMS data were cleaned.")    
        return clean_df_seism, colname   
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!RMS data were not cleaned.')
        

###############################################################################
def DeleteDuplicates(df_seism,df_sea):
    '''
    Deletes duplicates from sea and rms data 

            Parameters:
                df_seism (obj): dataframe containing RMS times series for each frequency band, station and channel   
                df_sea (obj): dataframe containing sea times series for each position of the area of interest
                
            Returns:
                clean_df_seism (obj): cleaned dataframe containing RMS times series for each frequency band, station and channel   
                clean_df_sea (obj): cleaned dataframe containing sea times series for each position of the area of interest                          
    '''
    
    try:
        #Checks if some duplicates are in the sea and RMS data dataframes.
        #If true, it deletes any duplicates 
        df_seism = df_seism.loc[~df_seism.index.duplicated(keep='first')]
        df_sea = df_sea.loc[~df_sea.index.duplicated(keep='first')]

        #Merges the sea and RMS dataframes and it deletes further null values
        alldata = pd.concat([df_seism, df_sea], axis=1)
        alldata.dropna(axis=0, inplace=True)
                            
        #Re-split the two different datasets 
        clean_df_seism = alldata.iloc[:, 0:df_seism.shape[1]].copy() 
        clean_df_sea = alldata.iloc[:, df_seism.shape[1]:].copy()
                            
        #Update log file about the completing of the operation 
        logging.info("OK!Duplicates from sea wave and RMS data were removed.")    
        return clean_df_seism, clean_df_sea   
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Duplicates from sea wave and RMS data were not removed.')

###############################################################################
def SkewnessCorrection(df_seism,skew_limit):
    '''
    Evaluates the symmetry/asymmetry of the data distrution for each column and
    applies the box-cox transformation if the values overcome a fixed threshold.
    This transformation allows to obtain a symmetryc distribution (normal distribution).
    
    The Box-Cox transform is given by:

    y = (x**lmbda - 1) / lmbda,  for lmbda != 0
    log(x),                       for lmbda = 0

            Parameters:
                df_seism (obj): dataframe containing RMS times series for each frequency band, station and channel   
                skew_limit (float):skewness values used to evaluate and correct the symmetry of the dataset
                
            Returns:
                clean_df_seism (obj): cleaned dataframe containing RMS times series for each frequency band, station and channel 
                idx_sks (ndarray): column indexes where the box-cox transformation is applied 
                lmbda (ndarray): values of lambda that maximizes the log-likelihood function
                alpha (ndarray): the 100 * (1-alpha)% confidence interval for each lambda 
                
    '''
    
    try:
        
        #Creates a copy of the RMS dataset 
        clean_df_seism=df_seism.copy()
        
        #Initilize empyt arrays/list
        idx_sks=np.array([])
        lmbda=np.array([])
        alpha=[]
        #Checks for each column the symmetry/asymmetry of the distribution values
        for icol, col in enumerate(df_seism.columns): # enumerate() adds counter to an iterable and returns it.
            
            #Applies the skewness function to verify 
            #if the values are normally distributed
            #The skewness value provides the symmetric degree
            skewness = df_seism[col].skew()
            
            #If thee skwness values overcome a certain threshold,
            #then the distribution is asymmetric
            
            if abs(skewness) > skew_limit:
                
               #Applies the box-cox transformation to obtain a symmetric distribution 
               bxcx = stats.boxcox(df_seism.iloc[:,icol],alpha=True)
               clean_df_seism.iloc[:,icol]=bxcx[0]
               
               #Saves information about boc-cox transformation for each column 
               idx_sks=np.append(idx_sks,icol)
               lmbda=np.append(lmbda,bxcx[1])
               alpha.append(bxcx[2])
        
                            
        #Update log file about the completing of the operation 
        logging.info("OK!Skewness correction was performed.")    
        return clean_df_seism, idx_sks, lmbda, alpha   
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Skewness correction was not performed.')
###############################################################################
def EarthQuakeCatalog(clean_df_seism, clean_df_sea,starttime,endtime,eqCatalog,MagMed,MagWorld,lat1, lat2, lon1, lon2, hours_del):
    '''
    Checks if significant global or regional earthquakes (in terms of magnitude) may affect the whole learning process,
    Eventually, it deletes the rows (hours) from the data affected by quakes.

            Parameters:
                    clean_df_seism (obj): cleaned dataframe containing RMS times series for each frequency band, station and channel   
                    clean_df_sea (obj): cleaned dataframe containing sea times series for each position of the area of interest 
                    starttime (obj): Date and time of the first data sample given in UTC (default value is “1970-01-01T00:00:00.0Z”).
                    endtime (obj): Date and time of the last data sample given in UTC (default value is “1970-01-01T00:00:00.0Z”).
                    eqCatalog (str): link to real-time earthquakes
                    MagMed (float): magnitude threshold for regional earthquakes
                    MagWorld (float): magnitude threshold for worldwide earthquakes
                    lat1, lat2, lon1, lon2 (float): area for regional earthquakes in decimaal degrees 
                    hours_del (float): semi-interval in hours used to delete data affected by earthquakes 

            Returns:
                    clean_df_seism (obj): cleaned dataframe containing RMS times series for each frequency band, station and channel   
                    clean_df_sea (obj): cleaned dataframe containing sea times series for each position of the area of interest 
    '''
    
    try:
        #Builds the query to retrieve earthquake catalog
        start_term1=starttime.strftime("%Y-%m-%d")
        start_term2=starttime.strftime("%H:%M:%S")
        end_term1=endtime.strftime("%Y-%m-%d")
        end_term2=endtime.strftime("%H:%M:%S")
        query_term='?starttime='+start_term1+'%20'+start_term2+'&endtime='+end_term1+'%20'+ end_term2+'&minmagnitude='+str(MagMed)+'&orderby=time'
               
        #Reading earthquakes catalog 
        eqs = pd.read_csv(filepath_or_buffer=eqCatalog+query_term)
        
        #Creating two copies of the original dataframes 
        eqs_wolrd= eqs.copy()
        eqs_region= eqs.copy()
        
        #Selecting global earthquakes overcoming a magnitude threshold,
        #except for Mediterreanean earthquakes (regional)
        eqs_wolrd=eqs_wolrd.loc[((eqs_wolrd['time']>=starttime) 
                                      & (eqs_wolrd['time']<= endtime)) 
                                      & (eqs_wolrd['mag']>MagWorld) 
                                      & ((eqs_wolrd['latitude']<lat1) 
                                      | (eqs_wolrd['latitude']>lat2)) 
                                      & ((eqs_wolrd['longitude']<lon1) 
                                      | (eqs_wolrd['longitude']>lon2))]
        
        #Selecting Mediterreanean earthquakes overcoming a magnitude threshold,
        eqs_region=eqs_region.loc[(eqs_region['time']>=starttime) 
                                      & (eqs_region['time']<= endtime) 
                                      & (eqs_region['mag']>MagMed) 
                                      & (eqs_region['latitude']>=lat1) 
                                      & (eqs_region['latitude']<=lat2) 
                                      & (eqs_region['longitude']>=lon1) 
                                      & (eqs_region['longitude']<=lon2)]
        
        #If regional or global earthquakes are non-empty,
        #the code deletes the a time interval from RMS and sea wave data 
        #for each eartquakes.
        if (len(eqs_wolrd)!=0) | (len(eqs_region)!=0):
            
            #Extracts the time from earthquakes datasets
            time_region= eqs_region.time
            time_world=eqs_wolrd.time
            time=pd.concat([time_region,time_world],axis=0)
            
            #Deletes rows for each earthquake
            for i in time:
                
                #Calculates time interval
                time1= pd.to_datetime(i, utc=True)
                time2=time1+ pd.to_timedelta(hours_del, unit='h')
                
                #Selects rows without eartquakes  
                clean_df_sea=clean_df_sea.loc[(clean_df_sea.index < time1) | (clean_df_sea.index > time2)]
                clean_df_seism=clean_df_seism.loc[(clean_df_seism.index < time1) | (clean_df_seism.index > time2)]
                      
            #Update log file about the completing of the operation 
            logging.info("OK! Earthquakes were found. Data were cleaned.")
        else:
            #Update log file about the completing of the operation 
            logging.info("OK! Earthquakes were not found.")           
        return clean_df_seism, clean_df_sea
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning! RMS and Sea data were not cleaned from earthquakes.')        
###############################################################################
def SaveData(folder_save, file_out, df_seism,df_sea, colmask, org_shape, colname, idx_sks, lmbda, alpha):
    '''
    Saves sea wave and rms data in a local repository.

            Parameters:
               folder_save (str): directory in which you can save the data
               file_out (str): name of the output file containing pre-processed data
               df_sea (obj): dataframe containing sea times series for each position of the area of interest  
               df_seism (obj): dataframe containing RMS times series for each frequency band, station and channel
               colmask (ndarray): indexes of non-NaN values of the sea wave height data used during the learning of the model 
               org_shape (ndarray): vector containing the original shape of the sea wave height matrix used during the learning of the model 
               colname (obj): header of  RMS data 
               idx_sks (ndarray): column indexes where the box-cox transformation is applied 
               lmbda (ndarray): values of lambda that maximizes the log-likelihood function
               alpha (ndarray): the 100 * (1-alpha)% confidence interval for each lambda 
            
            Returns:
                save the results into .pickle format
                             
    '''
    
    try:
        # Store data (serialize)
        with open(folder_save+'/'+ file_out, 'wb') as handle:
            pickle.dump(df_seism, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(df_sea, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(colmask, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(org_shape, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(colname, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(idx_sks, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(lmbda, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(alpha, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        #Update log file about the completing of the operation 
        logging.info("OK!Sea and RMS data were saved.")    
            
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Sea and RMS data were not saved.')
        
###############################################################################

def LoadData(filename):
    '''
    Load pre-processed data sea wave and rms data from a local repository.

            Parameters:
               filename (str): directory containing pre-processed data 

            Returns:
                df_sea (obj): dataframe containing sea times series for each position of the area of interest  
                df_seism (obj): dataframe containing RMS times series for each frequency band, station and channel
                folder_save (str): directory in which you can save the data
                colmask (ndarray): indexes of non-NaN values of the sea wave height data used during the learning of the model 
                org_shape (ndarray): vector containing the original shape of the sea wave height matrix used during the learning of the model 
                colname (obj): header of  RMS data 
                idx_sks (ndarray): column indexes where the box-cox transformation is applied 
                lmbda (ndarray): values of lambda that maximizes the log-likelihood function
                alpha (ndarray): the 100 * (1-alpha)% confidence interval for each lambda 
                             
    '''
    
    try:
        # Store data (serialize)
        with open(filename, 'rb') as handle:
            df_seism=pickle.load(handle)
            df_sea=pickle.load(handle)
            colmask=pickle.load(handle)
            org_shape=pickle.load(handle)
            colname=pickle.load(handle)
            idx_sks=pickle.load(handle)
            lmbda=pickle.load(handle)
            alpha=pickle.load(handle)
            
        #Update log file about the completing of the operation 
        logging.info("OK!Pre-processed sea and RMS data were loaded.")    
        return df_seism,df_sea, colmask, org_shape, colname, idx_sks, lmbda, alpha   
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Pre-processed sea and RMS data were not loaded.')
        
###############################################################################

def SplitData(clean_df_seism, clean_df_sea, percentage,nchunks, k):
    '''
    Splits the data into training and test k-th datasets.

            Parameters:
               clean_df_sea (obj): cleaned dataframe containing sea times series for each position of the area of interest  
               clean_df_seism (obj): cleaned dataframe containing RMS times series for each frequency band, station and channel 
               percentage (float): splitting proportion training/test
               nchunks (int): number of chuncks used to select differnt temporal samples
               k (int): number of training/test datasets used for cross-validation
            Returns:
                list_of_dataset (list): list of training/test datasets 
                T (obj): timing vector of the sea wave data
                scaler_x, scaler_y (obj): scaling object containing the direct/inverse transformations
                
                 
                             
    '''
    
    try:
        #Separetes features (RMS data) and target (Sea data) variables.
        #In addition, it saves the timing vector of the sea wave data
        X = clean_df_seism.values 
        Y = clean_df_sea.values
        T = clean_df_sea.index.tolist()
        
        #Scaling RMS and sea wave data data between 0 ans 1.
        #Defines the function used for scaling data
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        #Calucaltes the minimum and maximum values of the data
        #in order to use them in the scaling
        scaler_x.fit(X) 
        scaler_y.fit(Y)
        
        #Applies the scaling transformation
        Xscale = scaler_x.transform(X) 
        Yscale = scaler_y.transform(Y)
        
        #Calculates the training size and the length of the chunks 
        training_size =  (clean_df_sea.shape[0]*percentage)//100
        lenchunk = round(training_size/nchunks)
        
        #Generates k-th datasets for Cross Validation 
        list_of_dataset = [{'idx': [], 'XscaleTrain': [], 'YscaleTrain': [], 'XscaleTest': [], 'YscaleTest': []} for number in range(k)]

        for i, dic in enumerate(list_of_dataset):

            #Creates the training dataset 
            list_of_dataset[i]['idx'] = sampleIdxTrainSet(Xscale.shape[0], lenchunk, nchunks)
            list_of_dataset[i]['XscaleTrain'] = Xscale[list_of_dataset[i]['idx'],:] 
            list_of_dataset[i]['YscaleTrain'] = Yscale[list_of_dataset[i]['idx'],:] 
            
            #Creates the tests dataset 
            list_of_dataset[i]['XscaleTest'] = np.delete(Xscale, list_of_dataset[i]['idx'], axis=0) # test set con i dati del microsisma.
            list_of_dataset[i]['YscaleTest'] = np.delete(Yscale, list_of_dataset[i]['idx'], axis=0) # Test set con i dati del mare.
   
            
        #Update log file about the completing of the operation 
        logging.info("OK!Training and Test data were correctly splitted.")    
        return list_of_dataset, T, scaler_x, scaler_y
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Training and Test data were not correctly splitted.')

###############################################################################
def LearnPredictData(list_of_dataset,k, scaler_x, scaler_y,RF_max_depth, RF_n_estimator, RF_max_features, KNN_n_neighbors, KNN_weights, LGB_n_estimators, LGB_learning_rates, LGB_max_depth, LGB_num_leaves, model_flag):
    '''
    Learns the model by using the cross-validation and predicts
    the sea wave data for the testing RMS data

            Parameters:
                list_of_dataset (list): list of training/test datasets
                scaler_x, scaler_y (obj): scaling object containing the direct/inverse transformations
                kfolds (int): number of datasets for Cross-Validation
                RF_max_depth (int): maximum depth of the tree 
                RF_n_estimator (int): number of trees in the forest
                RF_max_features (int): number of features to consider when looking for the best split
                KNN_n_neighbors (int): number of neighbors to use by default for kneighbors queries
                KNN_weights (str): weight function used in prediction.
                LGB_n_estimators (int): number of boosted trees to fit
                LGB_learning_rates (float): boosting learning rate
                LGB_max_depth (float): maximum tree depth for base learners     
                LGB_num_leaves (int): maximum tree leaves for base learners
                model_flag (str): algorithm used to learn the model
               
            Returns:
                Ypred (ndarray): array containing the predicted sea wave data
                YTest (ndarray): array containing the original/testing sea wave data
    '''
    
    try:
        #Chooses and sets the machine learning algorithm.
        #The variable model_flag sets the algorithm to use
        match model_flag:
            
            #k-Nearest Neighbors
            case 'KNN':
                from sklearn.neighbors import KNeighborsRegressor
                model = KNeighborsRegressor(n_neighbors=KNN_n_neighbors,weights=KNN_weights) 
            
            #Random forest
            case 'RF':
                from sklearn.ensemble  import RandomForestRegressor
                model = RandomForestRegressor(bootstrap = True, max_depth=RF_max_depth, n_estimators=RF_n_estimator, max_features= RF_max_features)
        
        #Initiliaze the a empty variables
        Ypred = None
        YTest = None
        #Applies the learning/prediction to the k-th datasets  
        for index, d in enumerate(list_of_dataset):    
            
            #Fits the model to training data 
            model.fit(d['XscaleTrain'], d['YscaleTrain'])
            
            #Predicts the sea wave data by using the testing data 
            ypred = model.predict(d['XscaleTest'])
            
            #Inverts the scaling of the data and stores the results in the predicted/testing arrays
            Ypred = scaler_y.inverse_transform(ypred)if Ypred is None else np.concatenate([Ypred, scaler_y.inverse_transform(ypred)], axis =0) 
            YTest = scaler_y.inverse_transform(d['YscaleTest']) if YTest is None else np.concatenate([YTest, scaler_y.inverse_transform(d['YscaleTest'])], axis =0)
                    
        #Update log file about the completing of the operation 
        logging.info("OK!Cross-validation learning and prediction were completed.")    
        return Ypred,YTest   
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Cross-validation learning and prediction were not completed.')
        
###############################################################################
def EvaluatePerformance(Ypred,YTest):
    '''
    Calculates the differences bewtween the real and the predicted values 

            Parameters:
                Ypred (ndarray): array containing the predicted sea wave data
                YTest (ndarray): array containing the original/testing sea wave data
                
            Returns:
                Sea (ndarray): sea wave data in normal conditions 
                SeaExtr (ndarray): sea wave data in extreme conditions
                ErrN (ndarray): absolute errors for normal sea state 
                ErrMeanN (ndarray): average absolute errors for normal sea state
                ErrRelN (ndarray): relative errors for normal sea state
                ErrRelMeanN (ndarray): average relative errors for normal sea state
                ErrExtr (ndarray): absolute errors for extreme sea state 
                ErrExtrMean (ndarray): average absolute errors for extreme sea state
                ErrRelExtr (ndarray): relative errors for extreme sea state
                ErrRelExtrMean (ndarray): average relative errors for extreme sea state
                R2, R2_N, R2_E (float): R-squared values calculated between real and predicted data for general, normal and extreme sea state condition 
    '''
    
    try:
        #Calculates the difference between the predicted and real values 
        yerr = np.abs(Ypred-YTest) 
               
        #Evaluates normal sea state
        Sea = np.nanpercentile(YTest, 99, axis=0) 
        
        #Selects only values when the sea is <= 99 percentile
        #It creates a mask for the remainder values 
        inoN = YTest>Sea
        maskN = np.zeros(YTest.shape) 
        maskN[ inoN ] = np.nan 
        yerrN = yerr + maskN 
        
        #Computes the 99 percentile and the average of the absolute errors 
        ErrN = np.nanpercentile(yerrN, 99, axis=0)
        ErrMeanN = np.nanmean(yerrN, axis=0) # np.nanmean computes the arithmetic mean along the specified axis, ignoring NaNs.
   
        #Calculates the relative errors 
        yerrRel = yerrN/YTest
         
        #Computes the 99 percentile and the average of the relative errors 
        ErrRelN = np.nanpercentile(yerrRel, 99, axis=0)
        ErrRelMeanN = np.nanmean(yerrRel, axis=0)

        #Select only values greater than 99 percentile
        inoE = YTest<=Sea
        
        #Selects only values when the sea is > 99 percentile
        #It creates a mask for the remainder values 
        maskE = np.zeros(YTest.shape)
        maskE [ inoE ] = np.nan
        yerrE = yerr + maskE

        #Evaluates normal sea state
        #It selects the 99.99 percentile excluding possible outlier (0.01 percentile)
        SeaExtr = np.nanpercentile(YTest, 99.99, axis=0)

        #Selects only values when the sea extreme is <= 99.99 percentile
        #For the 0.01 percentile, it assigns NaN values 
        ino = YTest>SeaExtr
        mask = np.zeros(YTest.shape)
        mask[ ino ] = np.nan
        yerrEFinal = yerrE + mask

        #Computes the 99 percentile and the average of the absolute errors 
        ErrExtr = np.nanpercentile(yerrEFinal, 99, axis=0)
        ErrExtrMean = np.nanmean(yerrEFinal, axis=0)
        
        #Calculates the relative errors 
        yerrRelExt = yerrEFinal/YTest

        #Computes the 99 percentile and the average of the relative errors
        ErrRelExtr = np.nanpercentile(yerrRelExt, 99, axis=0)
        ErrRelExtrMean = np.nanmean(yerrRelExt, axis=0)
        
        #Calculates R-squared score for all data
        #Flating real/testing and predicted data
        YTest_flat = YTest.flatten(order='C').copy() 
        Ypred_flat = Ypred.flatten(order='C').copy() 
        
        #Computes the total R-squared score
        R2 = r2_score(YTest_flat, Ypred_flat) 
        
        #Computes the R-squared score for normal sea state
        #In this case, it uses the previous mask (for values <= 99 percentile) to set NaN to zeros 
        YTestN = YTest + maskN 
        YTestN = np.nan_to_num(YTestN)
        YpredN = Ypred + maskN
        YpredN = np.nan_to_num(YpredN)
        YTestN_flat = YTestN.flatten(order='C').copy() 
        YpredN_flat = YpredN.flatten(order='C').copy()
        R2_N = r2_score(YTestN_flat, YpredN_flat)
        
        #Computes the R-squared score for extreme sea state
        #In this case, it uses the previous mask (for values>99 percentile) to set NaN to zeros 
        YTestE = YTest + maskE 
        YTestE = np.nan_to_num(YTestE) 
        YpredE = Ypred + maskE
        YpredE = np.nan_to_num(YpredE)
        YTestE_flat = YTestE.flatten(order='C').copy() 
        YpredE_flat = YpredE.flatten(order='C').copy() 

        R2_E = r2_score(YTestE_flat, YpredE_flat) #Compute the total r2 score
        
        
        #Update log file about the completing of the operation 
        logging.info("OK!Model errors were calculated.")    
        return Sea, SeaExtr,ErrN, ErrMeanN, ErrRelN, ErrRelMeanN, ErrExtr, ErrExtrMean, ErrRelExtr, ErrRelExtrMean, R2, R2_N, R2_E   
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Model errors were not calculated.')

###############################################################################
def SaveResultsPerformance(folder_save,file_out_err, Sea, SeaExtr, ErrN, ErrMeanN, ErrRelN, ErrRelMeanN, ErrExtr, ErrExtrMean, ErrRelExtr, ErrRelExtrMean,R2, R2_N, R2_E):
    '''
    Calculates the differences bewtween the real and the predicted values 

            Parameters:
               folder_save (str): directory in which it saves the output 
               file_out_err (str): name of the .pickle file used to store the results of validation model 
               Sea (ndarray): sea wave data in normal conditions 
               SeaExtr (ndarray): sea wave data in extreme conditions               
               ErrN (ndarray): absolute errors for normal sea state 
               ErrMeanN (ndarray): average absolute errors for normal sea state
               ErrRelN (ndarray): relative errors for normal sea state
               ErrRelMeanN (ndarray): average relative errors for normal sea state
               ErrExtr (ndarray): absolute errors for extreme sea state 
               ErrExtrMean (ndarray): average absolute errors for extreme sea state
               ErrRelExtr (ndarray): relative errors for extreme sea state
               ErrRelExtrMean (ndarray): average relative errors for extreme sea state
               R2, R2_N, R2_E (float): R-squared values calculated between real and predicted data for general, normal and extreme sea state condition 
               
            Returns:
                save the errors into .pickle format
                              
    '''
    
    try:
        # Store data (serialize)
        with open(folder_save+'/'+ file_out_err, 'wb') as handle:
            pickle.dump(Sea, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(SeaExtr, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(ErrN, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(ErrMeanN, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(ErrRelN, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(ErrRelMeanN, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(ErrExtr, handle, protocol=pickle.HIGHEST_PROTOCOL)          
            pickle.dump(ErrExtrMean, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(ErrRelExtr, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(ErrRelExtrMean, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(R2, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(R2_N, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(R2_E, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #Update log file about the completing of the operation 
        logging.info("OK!Model errors were saved.")    
        return Sea, SeaExtr, ErrN, ErrMeanN, ErrRelN, ErrRelMeanN, ErrExtr, ErrExtrMean, ErrRelExtr, ErrRelExtrMean   
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Model errors were not saved.')
        
###############################################################################
def PlotResultsPerformance(folder_save,model_flag,org_shape,colmask, Sea, SeaExtr, ErrN, ErrMeanN, ErrRelN, ErrRelMeanN, ErrExtr, ErrExtrMean, ErrRelExtr, ErrRelExtrMean,R2, R2_N, R2_E, mlat, Mlat,mlon, Mlon):
    '''
    Plots and saves diagramas concerning the performance of the model 

            Parameters:
               folder_save (str): directory in which it saves the output
               model_flag (str): algorithm used to learn the model
               colmask (ndarray): indexes of non-NaN values of the sea wave height data used during the learning of the model 
               org_shape (ndarray): vector containing the original shape of the sea wave height matrix used during the learning of the model
               Sea (ndarray): sea wave data in normal conditions 
               SeaExtr (ndarray): sea wave data in extreme conditions
               ErrN (ndarray): absolute errors for normal sea state 
               ErrMeanN (ndarray): average absolute errors for normal sea state
               ErrRelN (ndarray): relative errors for normal sea state
               ErrRelMeanN (ndarray): average relative errors for normal sea state
               ErrExtr (ndarray): absolute errors for extreme sea state 
               ErrExtrMean (ndarray): average absolute errors for extreme sea state
               ErrRelExtr (ndarray): relative errors for extreme sea state
               ErrRelExtrMean (ndarray): average relative errors for extreme sea state
               R2, R2_N, R2_E (float): R-squared values calculated between real and predicted data for general, normal and extreme sea state condition
               lat1, lat2, lon1, lon2 (float): area of interest used to learn the model 
                
            Returns:
                save the errors into .jpg format
                              
    '''
    
    try:
        
        #Reshape all sea wave and errors vectors into matrixes 
        MatSea = vec2mat(Sea, org_shape, colmask) 
        MatSeaExtr = vec2mat(SeaExtr, org_shape, colmask)
        MatErrN = vec2mat(ErrN, org_shape, colmask)
        MatErrMeanN = vec2mat(ErrMeanN, org_shape, colmask)        
        MatErrRelN = vec2mat(ErrRelN, org_shape, colmask)
        MatErrRelMeanN = vec2mat(ErrRelMeanN, org_shape, colmask)                
        MatErrExtr = vec2mat(ErrExtr, org_shape, colmask)
        MatErrExtrMean = vec2mat(ErrExtrMean, org_shape, colmask)       
        MatErrRelExtr = vec2mat(ErrRelExtr, org_shape, colmask)
        MatErrRelExtrMean = vec2mat(ErrRelExtrMean, org_shape, colmask)
        
        #Disable the visualization of the figure
        plt.ioff()
        
        #Creates a subplot 4X3
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 9))
        fig.suptitle(f'Performance of the {model_flag} model. R\u00b2 ={round(R2,2)}. Normal sea R\u00b2 ={round(R2_N,2)}. Extreme sea R\u00b2 ={round(R2_E,2)} ')
        
        #Get the position of each subplot in the figure grid 
        gs = ax[0, 0].get_gridspec()
        
        
        #Remove the underlying axes
        for iax in ax[:,0]:
            iax.remove()
        
        #Adds new suplots in the figure 
        ax[0][0] = fig.add_subplot(gs[0:2, 0])
        ax[2][0] = fig.add_subplot(gs[2:5, 0])

        #Displays normal condition of sea wave data 
        img1 = ax[0][0].imshow(MatSea, cmap=plt.get_cmap('Blues'), extent=(mlon, Mlon, mlat, Mlat))
        ax[0][0].set_title('<99% of wave height')
        cbar1 = colorbar(img1)
        cbar1.set_label('m')
        ax[0][0].set_ylabel('normal sea')
        
        #Displays absolute errors 
        img2 = ax[0][1].imshow(MatErrN, cmap=plt.get_cmap('Oranges'), extent=(mlon, Mlon, mlat, Mlat))
        ax[0][1].set_title('<99% of |errs|')
        cbar2 = colorbar(img2)
        cbar2.set_label('m')
        
        #Displays relative errors and their average value 
        img3 = ax[0][2].imshow(MatErrRelN, cmap=plt.get_cmap('Greys'), extent=(mlon, Mlon, mlat, Mlat))
        ax[0][2].set_title('<99% of relative err')
        cbar3 = colorbar(img3)
        cbar3.set_label(f'mean = {np.nanmean(MatErrRelN):0.2f}')
        
        #Deletes the first empty subplot(it's not used)
        ax[1][0].axis('off')
        
        #Displays average absolute errors 
        img4 = ax[1][1].imshow(MatErrMeanN, cmap=plt.get_cmap('Greens'), extent=(mlon, Mlon, mlat, Mlat))
        ax[1][1].set_title('Mean of |errs|')
        cbar4 = colorbar(img4)
        cbar4.set_label('m')
        
        #Displays average relative errors and their average value 
        img5 = ax[1][2].imshow(MatErrRelMeanN, cmap=plt.get_cmap('Purples'), extent=(mlon, Mlon, mlat, Mlat))
        ax[1][2].set_title('Mean of relative err')
        cbar5 =colorbar(img5)
        cbar5.set_label(f'mean = {np.nanmean(MatErrRelMeanN):0.2f}')
        
        #Displays extreme condition of sea wave data
        img6 = ax[2][0].imshow(MatSeaExtr, cmap=plt.get_cmap('Blues'), extent=(mlon, Mlon, mlat, Mlat))
        ax[2][0].set_title('99% of 1% of extreme wave height')
        cbar6 = colorbar(img6)
        cbar6.set_label('m')
        ax[2][0].set_ylabel('extreme sea')

        #Displays absolute errors
        img7 = ax[2][1].imshow(MatErrExtr, cmap=plt.get_cmap('Oranges'), extent=(mlon, Mlon, mlat, Mlat))
        ax[2][1].set_title('<99% of |errs|')
        cbar7 = colorbar(img7)
        cbar7.set_label('m')
        
        #Displays relative errors and their average value 
        img8 = ax[2][2].imshow(MatErrRelExtr, cmap=plt.get_cmap('Greys'), extent=(mlon, Mlon, mlat, Mlat))
        ax[2][2].set_title('<99% of relative err')
        cbar8 = colorbar(img8)
        cbar8.set_label(f'mean = {np.nanmean(MatErrRelExtr):0.2f}')
        
        #Deletes the first empty subplot(it's not used)
        ax[3][0].axis('off')
        
        #Displays average absolute errors
        img9 = ax[3][1].imshow(MatErrExtrMean, cmap=plt.get_cmap('Greens'), extent=(mlon, Mlon, mlat, Mlat))
        ax[3][1].set_title('Mean of |errs|')
        cbar9 = colorbar(img9)
        cbar9.set_label('m')
        
        ##Displays average relative errors and their value 
        img10 = ax[3][2].imshow(MatErrRelExtrMean, cmap=plt.get_cmap('Purples'), extent=(mlon, Mlon, mlat, Mlat))
        ax[3][2].set_title('Mean of relative err')
        cbar10 = colorbar(img10)
        cbar10.set_label(f'mean = {np.nanmean(MatErrRelExtrMean):0.2f}')
        
        #Separates the upper subplots from the lower ones 
        line = plt.Line2D((.0,1.),(.485,.485), color="k", linewidth=2)
        fig.add_artist(line)
        plt.tight_layout(h_pad=1)
        
        #Save the diagram inyo .jpg file in the desidered directory
        plt.savefig(folder_save + '/'+ model_flag+'.jpg', dpi=500)
        plt.close() 
        
        #Update log file about the completing of the operation 
        logging.info("OK!Model errors were displayed and saved.")    
        return ErrN, ErrMeanN, ErrRelN, ErrRelMeanN, ErrExtr, ErrExtrMean, ErrRelExtr, ErrRelExtrMean   
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Model errors were not displayed and saved.')

###############################################################################
def LearnFinalModel(clean_df_seism, clean_df_sea, RF_max_depth, RF_n_estimator, RF_max_features, KNN_n_neighbors, KNN_weights, LGB_n_estimators, LGB_learning_rates, LGB_max_depth, LGB_num_leaves, model_flag):
    '''
    Learns machine learning model for the prediction of sea wave data
    by using all dataset.

            Parameters:
                clean_df_sea (obj): cleaned dataframe containing sea times series for each position of the area of interest  
                clean_df_seism (obj): cleaned dataframe containing RMS times series for each frequency band, station and channel
                RF_max_depth (int): maximum depth of the tree 
                RF_n_estimator (int): number of trees in the forest
                RF_max_features (int): number of features to consider when looking for the best split
                KNN_n_neighbors (int): number of neighbors to use by default for kneighbors queries
                KNN_weights (str): weight function used in prediction.
                LGB_n_estimators (int): number of boosted trees to fit
                LGB_learning_rates (float): boosting learning rate
                LGB_max_depth (float): maximum tree depth for base learners     
                LGB_num_leaves (int): maximum tree leaves for base learners
                model_flag (str): algorithm used to learn the model
               
            Returns:
                model (obj): predictive model
                scaler_x, scaler_y (obj): scaling object containing the direct/inverse transformations
                                             
    '''
    
    try:
        #Separetes features (RMS data) and target (Sea data) variables.
        X = clean_df_seism.values 
        Y = clean_df_sea.values
        
        #Scaling RMS and sea wave data data between 0 ans 1.
        #Defines the function used for scaling data
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        #Calucaltes the minimum and maximum values of the data
        #in order to use them in the scaling
        scaler_x.fit(X) 
        scaler_y.fit(Y)
        
        #Applies the scaling transformation
        Xscale = scaler_x.transform(X) 
        Yscale = scaler_y.transform(Y)
        
        #Chooses and sets the machine learning algorithm.
        #The variable model_flag sets the algorithm to use
        match model_flag:
            
            #k-Nearest Neighbors
            case 'KNN':
                from sklearn.neighbors import KNeighborsRegressor
                model = KNeighborsRegressor(n_neighbors=KNN_n_neighbors,weights=KNN_weights) 
            
            #Random forest
            case 'RF':
                from sklearn.ensemble  import RandomForestRegressor
                model = RandomForestRegressor(bootstrap = True, max_depth=RF_max_depth, n_estimators=RF_n_estimator, max_features= RF_max_features)
       
        #Fits the chosen model
        model.fit(Xscale, Yscale)
        
        #Update log file about the completing of the operation 
        logging.info("OK!Predictive model was retrieved.")    
        return model,scaler_x, scaler_y  
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Predictive model was not retrieved.') 

###############################################################################
def SaveFinalModel(folder_save, filemodel, model,scaler_x,scaler_y, model_flag):
    '''
    Saves machine learning model for the prediction of sea wave data
    by using all dataset.

            Parameters:
                folder_save (str): directory in which it saves the output
                filemodel (str): name of the file containing the predictive model
                model (obj): predictive model
                scaler_x, scaler_y (obj): scaling object containing the direct/inverse transformations
                model_flag (str): algorithm used to learn the model
                
            Returns:
                saves the predictive model and co-paramterers into .sav file
               
                                             
    '''
    
    try:
        #Saves the model 
        pickle.dump([model,scaler_x,scaler_y], open(folder_save+'/'+filemodel+'_'+model_flag+'.sav', 'wb'))
        
        #Update log file about the completing of the operation 
        logging.info("OK!Predictive model was saved.")    
        return model,scaler_x, scaler_y  
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Predictive model was not saved.')
        
###############################################################################
def LoadFinalModel(folder_save, filemodel,model_flag):
    '''
    Load machine learning model for the prediction of sea wave data
    by using all dataset.

            Parameters:
                folder_save (str): directory in which it saves the output
                filemodel (str): name of the file containing the predictive model
                model_flag (str): algorithm used to learn the model
               
            Returns:
                model (obj): predictive model
                scaler_x, scaler_y (obj): scaling object containing the direct/inverse transformations
            
                                             
    '''
    
    try:
        #Loads the model 
        model,scaler_x,scaler_y= pickle.load(open(folder_save+'/'+filemodel+'_'+model_flag+'.sav', 'rb'))
        
        #Update log file about the completing of the operation 
        logging.info("OK!Predictive model was loaded.")    
        return model,scaler_x,scaler_y
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Predictive model was not loaded.')
###############################################################################
#OTHER SUBFUNCTIONS
###############################################################################
def sampleIdxTrainSet(length, lenchunk, nchunks):
    '''
    Selects a reliable sample of the data avoiding 
    to sample close data that will be in the test

            Parameters:
               length (int): length of the data   
               lenchunk (int): length of each chunk in samples 
               nchunks (int): number of chuncks
               
            Returns:
                idx (list): new indexes of the rows of the data 
                                                          
    '''
    try:
        #Initizializes the list of the new indexes 
        idx = None
        
        #Fixes the maximum starting position of the chunk  
        istop = length - lenchunk  
        
        #Sets randomic starting position of a chunk 
        istart = np.random.randint(lenchunk)
        
        #Defines the position of the chunks  
        ndxs = list(range(istart, istop, lenchunk)) 
        
        #Selects a randomic number of chunks 
        #and sorts them
        isx = sorted(random.sample(ndxs, nchunks))
        
        #Fixed the starting index position of the chunks 
        idx = []
        for j in isx:
            idx += list(range(j, j+lenchunk))   
                
        return idx
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning! Impossible to find a proper training set.')

###############################################################################
def vec2mat(x, origshape, colmask):
    '''
    Reshapes the errors vector into a matrix on the basis of the shape
    of the original sea wave data 

            Parameters:
               x(ndarray): errors vector 
               colmask (ndarray): indexes of non-NaN values of the sea wave height data used during the learning of the model 
               orgsshape (ndarray): vector containing the original shape of the sea wave height matrix used during the learning of the model
                
            Returns:
                X(ndarray): matrix of errors
                              
    '''
    
    try:
        #Creates an empyt matrix by using the same size of the sea wave data
        support = np.empty(origshape[1]*origshape[2])
        
        #Fills with NaNs
        support[:] = np.nan
        
        #Use the mask to assign the non-null values to the correct position
        support[colmask] = x
        
        #Reshape the data on the basis of the size of the original sea wvae data
        X = np.reshape(support, (origshape[1], origshape[2]))
        
        return X
    except Exception:
        pass  
    
###############################################################################
def colorbar(mappable): 
    '''
    Reshapes the errors vector into a matrix on the basis of the shape
    of the original sea wave data 

            Parameters:
               mappable(obj): surface/3D object 
                
            Returns:
                cbar(obj): colorbar object 
                              
    '''
    try:
        #Retrieves the current axes 
        last_axes = plt.gca()
        
        #Adds the axes to the current graphic object
        ax = mappable.axes 
        
        #Retrievies the figure object
        fig = ax.figure
        
        #Creates a divisor for the actual axes 
        divider = make_axes_locatable(ax)
        
        #Sets the features of colorbar 
        cax = divider.append_axes("right", size="5%", pad=0.05) 
        
        #Assigns the colorbar to the actual axes and graphic object
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes) 
        return cbar
    except Exception:
        pass  