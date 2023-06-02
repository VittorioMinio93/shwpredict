# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:55:59 2022

@author: Utente
"""
#Importing main packages 

import logging
import numpy as np
import configparser
import ast
from glob import glob
from obspy.core import UTCDateTime
from  obspy import read 
import pandas as pd
from netCDF4 import Dataset
import pickle

   
def ReadConfig(filename):
    '''
    Reads each input paramaters stored in the configuration file

            Parameters:
                    filename (str): name of the configuration file 
            
            Returns:
                network (str): list of the network codes
                list_sta (list): list of station codes 
                list_chan (list): list of channel codes
                sensitivity (list): conversion factor from count to m/s
                local_seism (str):directory containing seismic traces 
                local_sea (str): directory containing sea state data 
                file_out (str): name of the .pickle file used to store the raw seismic/sea dataset                
                folder_save (str): directory in which it saves the output
                starttime, endtime (obj): temporal limits in UTC (default value is “1970-01-01T00:00:00.0Z”) for data processing
                win (float): lenght of the window analysis in seconds used for RMS calculation
                freq (list): list of upper- and down-limits (tuple) of frequency bands in Hz used for RMS calculation 
                rms_step (float): temporal step in hours used to compute RMS time series 
                lat1, lat2, lon1, lon2 (float): area of interest used to extrcat sea wave data
                nhours (float): step in hour to define the time interval to analyze 
                fileformat_sea (str): format of the name of the sea state files 
                dateformat_sea (str): temporal format used in the name of the sea state files
                fileformat_rms (str): format of the name of the seismic files 
                dateformat_rms (str): temporal format used in the name of the seismic files 
    '''
    try:
        #Reading the configuration file
        config_obj = configparser.ConfigParser()
        config_obj.read(filename)
        
        #Reading model and network sections from configuration file:
        net_info = config_obj["Network"]
        rms_info= config_obj["RMS"]
        model_info = config_obj["Model"]
        
        #Reading infomation about the seismic network used
        network=ast.literal_eval(net_info['network'])
        list_sta=ast.literal_eval(net_info['stations'])
        list_chan=ast.literal_eval(net_info['channels'])
        local_seism= net_info['local_seism']
        local_sea= net_info['local_sea']
        file_out=net_info['file_out']
        folder_save=net_info['folder save']
        starttime=UTCDateTime(net_info['starttime'])
        endtime=UTCDateTime(net_info['endtime'])
        nhours=float(net_info['nhours'])
        fileformat_sea=net_info['file format sea']
        dateformat_sea= net_info['date format sea']
        fileformat_rms=net_info['file format rms']
        dateformat_rms = net_info['date format rms']
        sensitivity = ast.literal_eval( net_info['sensitivity'])

        
        #Reading parameters for RMS time series calculation
        win=float(rms_info['time window (s)'])
        freq=ast.literal_eval(rms_info['frequency (Hz)'])
        rms_step=int(rms_info['rms_step'])
 
        #Reading all paramaters for learning the model 
        lat1=float(model_info['lat1'])
        lat2=float(model_info['lat2'])
        lon1=float(model_info['lon1'])
        lon2=float(model_info['lon2'])
            
        #Update log file about the completing of the operation 
        logging.info("OK!Configuration file was read.")
        return network, list_sta,  list_chan, sensitivity, local_seism, local_sea, file_out, folder_save, starttime, endtime, win, freq, rms_step, lat1, lat2, lon1, lon2, nhours, fileformat_sea,dateformat_sea, fileformat_rms,dateformat_rms
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Configuration file was not read.')

###############################################################################        
def LocalRMSAnalysis(network, list_sta,  list_chan, sensitivity, local_seism, starttime, endtime, win, freq, rms_step, nhours, fileformat='{station}..{channel}.D.{time}',dateformat='%Y.%j'):
    '''
    Calculates RMS time series 

            Parameters:
                    network (str): list of the network codes
                    list_sta (list): list of station codes 
                    list_chan (list): list of channel codes 
                    local_seism (str):directory containing seismic traces 
                    starttime, endtime (obj): temporal limits in UTC (default value is “1970-01-01T00:00:00.0Z”) for data processing
                    win (float): lenght of the window analysis in seconds used for RMS calculation
                    freq (list): list of upper- and down-limits (tuple) of frequency bands in Hz used for RMS calculation 
                    rms_step (float): temporal step in hours used to compute RMS time series 
                    nhours (float): step in hour to define the time interval to analyze 
                    fileformat (str): format of the name of the seismic files 
                    dateformat (str): temporal format used in the name of the seismic files  
                   
            
            Returns:
                df_seism (obj): dataframe containing raw RMS times series for each frequency band, station and channel
                
  
    '''
    try:
        #Define the temporal period to analyze
        time_period=np.arange(starttime,endtime, 3600*nhours)
        
        #Initialize an empty variable to storage all RMS data through rows
        df_seism=None
        
        #Iterates through each day
        for time_day in time_period:
            
            #Tries RMS calculation for each file. 
            #Eventually it skips to the next one 
            try:
            
                #Retrievies time information and converts to a string 
                time_day_str=time_day.strftime(dateformat)
                
                #Initialize an empty variable to storage all RMS data through columns
                data_col=None
                
                #Iterates through each channel
                for ch in list_chan:
                    
                    #Iterates through each stations 
                    for staz,calib in zip(list_sta,sensitivity):
                        
                        try:
                         
                            #Retrievies a list of files containing the seismic traces 
                            #by using the network and timing information 
                            fl=glob(local_seism+'/**/*'+fileformat.format(station=staz,channel=ch,time=time_day_str), recursive=True)[0]
                            
                            
                            
                            #Read seismic trace
                            st=read(fl)
                            
                            #Interpolates gaps 
                            st.merge(method=0,fill_value='interpolate')
                            
                            
                            #Calculating the RMS for each frequency band
                            for f1,f2 in freq:
                                
                                
                                #Create an empty dataframe with the correct header
                                #In this case, the dataframe must have three levels (station,channel,frequency band)
                                data_header = pd.DataFrame( [[staz,ch[-1], str(np.round(f1,2))+str(np.round(f2,2))]],
                                                       columns=['station', 'component', 'freq'])
                                
                                #Pre.processing and filtering of each Trace object 
                                tr = st.copy().slice(starttime=time_day,endtime=time_day+3600*nhours).detrend('demean').detrend('linear').filter('bandpass',freqmin=f1, freqmax=f2, corners=4)[0]
                                tr.data=tr.data/calib
                                
                                #Calculating the RMS values by sliding a time window along the whole length of
                                #the band-pass filters traces. Finally, the result consists of the median of RMS 
                                #values retrivied for each time window.
                                rms=[[wtr.stats.starttime.datetime,np.sqrt(np.nanmean(wtr.data**2))] for wtr in tr.slide(window_length=win, step=win, include_partial_windows=True)]
                                
                                #Converts the results into a dataframe and hourly resamples the data 
                                data_frame=pd.DataFrame(data=rms,columns=['time','rms'])
                                df = pd.DataFrame(data= data_frame['rms'].values, columns = pd.MultiIndex.from_frame(data_header),index= pd.to_datetime(data_frame['time'], utc=True))  
                                df=df.resample(str(int(rms_step))+'h').median()
                                #Storage the results with the correct order through the columns 
                                data_col = df if data_col is None else pd.concat([data_col, df], axis =1)
                
                        except Exception:
                            continue 
                #Storage the results with the correct order through the rows     
                df_seism =  data_col if df_seism is None else pd.concat([ df_seism, data_col], axis=0)             
            except Exception:
                continue 

        #Update log file about the completing of the operation 
        logging.info("OK!RMS time series were calculated.")
        return df_seism 
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!RMS time series were not calculated.')
        
###############################################################################
def LocalSeaAnalysis(lat1, lat2, lon1, lon2, local_sea, starttime, endtime,nhours,fileformat,dateformat):
    '''
    Retrieves sea wave height time series for a selected area 

            Parameters:
                    lat1, lat2, lon1, lon2 (float): area of interest used to extrcat sea wave data
                    local_sea (str): directory containing sea state data 
                    starttime, endtime (obj): temporal limits in UTC (default value is “1970-01-01T00:00:00.0Z”) for data processing                    
                    nhours (float): step in hour to define the time interval to analyze
                    fileformat (str): format of the name of the sea state files 
                    dateformat (str): temporal format used in the name of the sea state files
                          
            Returns:
                df_sea (obj): dataframe containing raw sea wave times series for the area of interest
                org_shape (tuple): shape of the raw sea data matrix
    '''
    try:
        #Define the temporal period to analyze
        time_period=np.arange(starttime,endtime, 3600*nhours)
        
        #Initialize an empty dictionary to storage all sea data through rows
        df_sea = {'Y': None, 'time': None, 'lat':[], 'lon':[]}
        
        #Iterates through each day
        for time_day in time_period:
            #Tries reading for each file. 
            #Eventually it skips to the next one 
            try:
                #Retrievies time information and converts to a string 
                time_day_str=time_day.strftime(dateformat)
                #Retrievies a list of files containing the sea state data
                #by using the timing information 
                fl=glob(local_sea+'/**/*'+fileformat.format(time=time_day_str)+'*', recursive=True)[0]
                                
                #Read sea height map, spatial and temporal parameters  
                f = Dataset(fl)
                height=f.variables['VHM0'][:].astype('float64')
                height[height.mask]=np.nan
                time=pd.to_datetime(f.variables['time'][:].astype('float64'),unit='s',utc=True)
                latitude=np.linspace(f.variables['latitude'].valid_min, f.variables['latitude'].valid_max,np.shape(height)[1])
                longitude=np.linspace(f.variables['longitude'].valid_min, f.variables['longitude'].valid_max,np.shape(height)[2])
                
                #Creates latitude/longitude grids
                Lat,Lon = np.meshgrid(latitude, longitude, indexing='ij')
                                
                #Storages the data 
                if df_sea['Y'] is None:
                    df_sea['Y'] = height
                    df_sea['time'] = time
                else:
                    df_sea['Y'] = np.append(df_sea['Y'], height , axis=0)
                    df_sea['time'] = np.append(df_sea['time'], time, axis =0 ) 
                     
            except Exception:
                continue
        
        
        #Search all rows and columns referring to the area of study 
        #in terms of latitude and longitude 
        lat_indices = np.argwhere((latitude >= lat1) & (latitude <= lat2))
        lon_indices = np.argwhere((longitude >= lon1) & (longitude <= lon2))
        
        
        #Filters data in terms of latitude and longitude 
        mr = np.min(lat_indices)
        Mr = np.max(lat_indices)
        mc = np.min(lon_indices)
        Mc = np.max(lon_indices)
        
        df_sea['Y'] = df_sea['Y'][:, mr:Mr, mc:Mc] 
        df_sea['lat'] = Lat[mr:Mr, mc:Mc]
        df_sea['lon'] = Lon[mr:Mr, mc:Mc]
        
        #Convert the time format into a user-friendly format 
        #and rotates the sea state data matrixes (sea wave height, 
        #latitude and longitude) of about 90°
        df_sea['Y'] = np.array([np.flipud(m) for m in df_sea['Y']])
        
        df_sea['lat'] = np.flipud(df_sea['lat'])
        df_sea['lon'] = np.flipud(df_sea['lon'])
        
        #Retrieves shape of the sea wave matrix 
        org_shape=df_sea['Y'].shape
        
        #Update log file about the completing of the operation 
        logging.info("OK!Sea wave time series were extracted.")
        return df_sea, org_shape 
    
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Sea wave time series were not extracted.')
        

###############################################################################
def SaveRawData(folder_save, file_out, df_seism,df_sea, org_shape):
    '''
    Retrieves sea wave height time series for a selected area 

            Parameters:
                    folder_save (str): directory in which it saves the output    
                    file_out (str): name of the .pickle file used to store the raw seismic/sea dataset                
                    df_seism (obj): dataframe containing raw RMS times series for each frequency band, station and channel
                    df_sea (obj): dataframe containing raw sea wave times series for the area of interest
                    org_shape (tuple): shape of the raw sea data matrix      
            Returns:
                    save the results into .pickle format
                
    '''
    try:
        # Store data (serialize)
        with open(folder_save+'/'+ file_out, 'wb') as handle:
            pickle.dump(df_seism, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(df_sea, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(org_shape, handle, protocol=pickle.HIGHEST_PROTOCOL)
           
            
        #Update log file about the completing of the operation 
        logging.info("OK! Raw sea and RMS data were saved.")    
            
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Raw sea and RMS data were not saved.')

###############################################################################
def LoadRawData(filename):
    '''
    Load raw data sea wave and rms data from a local repository.

            Parameters:
               filename (str): directory containing pre-processed data 

            Returns:
                df_sea (obj): dataframe containing sea times series for each position of the area of interest  
                df_seism (obj): dataframe containing RMS times series for each frequency band, station and channel
                org_shape (ndarray): vector containing the original shape of the sea wave height matrix  
                
                             
    '''
    
    try:
        # Store data (serialize)
        with open(filename, 'rb') as handle:
            df_seism=pickle.load(handle)
            df_sea=pickle.load(handle)
            org_shape=pickle.load(handle)
            
        #Update log file about the completing of the operation 
        logging.info("OK!Raw sea and RMS data were loaded.")    
        return df_seism, df_sea, org_shape  
    except Exception:
        #Update log file about the fall of the operation 
        logging.warning('Warning!Raw sea and RMS data were not loaded.')