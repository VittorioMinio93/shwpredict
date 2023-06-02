# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:47:44 2022

@author: Utente
"""
#Importing main packages and functions 
import swhpredict.data_processing.funcs as fn
import logging
import pathlib


def PrepareLocalDataset(filename,save_data=False):
    '''
    Reads seismic traces, calculates RMS time series and reads sea wave data 
        
            Parameters:
                    filename (str): name of the configuration file 
                    save_data (bool): if True, save the pre-processed data into .pickle file


            Returns:
                df_seism (obj): dataframe containing raw RMS times series for each frequency band, station and channel 
                df_sea (obj): dataframe containing raw sea wave times series for the area of interest
                org_shape (tuple): shape of the raw sea data matrix
                
                
    '''
   
    # Try the code until the return of an error
    try:
       
        #Create the log file and its configuration 
        logging.basicConfig( level=logging.DEBUG,
                             format="%(asctime)s %(message)s",force=True,handlers=[logging.FileHandler("log_DatPro.txt"),logging.StreamHandler()])

        logging.getLogger('matplotlib.font_manager').disabled = True
        
        logging.info("OK!Starting processing raw data.") 

        #Read all parameters from the configuration file 
        network, list_sta,  list_chan, sensitivity, local_seism, local_sea, file_out, folder_save, starttime, endtime, win, freq, rms_step, lat1, lat2, lon1, lon2, nhours, fileformat_sea,dateformat_sea, fileformat_rms,dateformat_rms = fn.ReadConfig(filename)
        
        #Checks if it is available an raw dataset in the output folder.
        #If true the code load only the results  without repeats any step
        #of the pre-processed analysis.
        filePrePro = pathlib.Path(folder_save+"/"+ file_out)
        if filePrePro.exists():
            logging.info("OK!Raw data file was found.")
            df_seism, df_sea, org_shape =fn.LoadRawData(filePrePro)
        else:
            ##Read seismic traces and calculate RMS
            df_seism=fn.LocalRMSAnalysis(network, list_sta,  list_chan, sensitivity, local_seism, starttime, endtime, win, freq, rms_step, nhours, fileformat_rms,dateformat_rms)
                    
            #Reads the sea wave maps for the area of interest
            df_sea, org_shape=fn.LocalSeaAnalysis(lat1, lat2, lon1, lon2, local_sea, starttime, endtime, nhours,fileformat_sea,dateformat_sea)
                   
            #Save RMS and sea wave data 
            if save_data==True:
                fn.SaveRawData(folder_save, file_out, df_seism,df_sea, org_shape)
            
        #Fixing the end of the whole process,
        #writing this information in the logfile and closing it
        logging.info("OK! Processing seismic raw terminated.") 
        logging.shutdown()
        return df_seism, df_sea, org_shape  
                  
    except Exception:
        #Fixing the end of the whole process,
        #writing this information in the logfile and closing it
        logging.info("Warning!Something was gone wrong.") 
        logging.shutdown()

