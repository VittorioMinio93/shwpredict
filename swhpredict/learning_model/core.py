# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:47:44 2022

@author: Utente
"""
#Importing main packages and functions 
import swhpredict.learning_model.funcs as fn
import logging
import pathlib


def PrepareDataset(filename,df_seism, df_sea, org_shape, earth_check=False,save_data=False):
    '''
    Returns the dataset used during the learning of the model
        
            Parameters:
                filename (str): name of the configuration file 
                df_seism (obj): dataframe containing raw RMS times series for each frequency band, station and channel 
                df_sea (obj): dataframe containing raw sea wave times series for the area of interest
                org_shape (tuple): shape of the raw sea data matrix
                earth_check (bool): if True, deletes rows affected by earthquakes from data
                save_data (bool): if True, save the pre-processed data into .pickle file


            Returns:
                clean_df_seism (obj): dataframe containing pre-processed RMS times series for each frequency band, station and channel   
                clean_df_sea (obj): dataframe containing pre_processed sea times series for each position of the area of interest
                colmask (ndarray): indexes of non-NaN values of the sea wave height data used during the learning of the model 
                org_shape (ndarray): vector containing the original shape of the sea wave height matrix used during the learning of the model 
                colname (obj): header of  RMS data 
                idx_sks (ndarray): column indexes where the box-cox transformation is applied 
                lmbda (ndarray): values of lambda that maximizes the log-likelihood function
                alpha (ndarray): the 100 * (1-alpha)% confidence interval for each lambda 
                
    '''
   
     # Try the code until the return of an error
    try:      
         #Create the log file and its configuration 
         logging.basicConfig( level=logging.DEBUG,
                             format="%(asctime)s %(message)s",force=True,handlers=[logging.FileHandler("log_PrePro.txt"),logging.StreamHandler()])
      
         logging.getLogger('matplotlib.font_manager').disabled = True
         
         logging.info("OK!Starting Pre-processing data.") 
      
         #Read all parameters from the configuration file 
         network, list_sta, list_chan, lat1, lat2,lon1, lon2, rms_thr, row_thr, skew_limit, percentage, nchunks, kfolds, RF_max_depth, RF_n_estimator, RF_max_features, KNN_n_neighbors, KNN_weights, LGB_n_estimators, LGB_learning_rates, LGB_max_depth, LGB_num_leaves, model_flag, file_out, file_out_err, folder_save, filemodel, eqCatalog, MagMed, MagWorld, mLat,MLat, mLon, MLon, starttime, endtime, hours_del = fn.ReadConfig(filename)
         
         #Checks if it is available an pre-processed dataset in the output folder.
         #If true the code load only the results  without repeats any step
         #of the pre-processed analysis.
         filePrePro = pathlib.Path(folder_save+"/"+ file_out)
         if filePrePro.exists():
             logging.info("OK!Pre-processing data file was found.")
             clean_df_seism, clean_df_sea, colmask, org_shape, colname, idx_sks, lmbda, alpha =fn.LoadData(filePrePro)
         else:
             logging.info("Warning!Pre-processing data file was not found.")
             
             #old version (maybe to delete)
             # #Read times series of Sea data
             # df_sea, org_shape =fn.ReadSeaData(sea_data,lat1, lat2,lon1, lon2)
             
             # #Read time series of RMS for each frequency band, station and channel
             # df_seism=fn.ReadRmsData(seism_data, network, list_sta, list_chan)
             
             #Cleans the sea data
             clean_df_sea,colmask=fn.CleanSeaData(df_sea)
                         
             
             #Cleans the RMS data
             clean_df_seism, colname=fn.CleanRmsData(df_seism,rms_thr,row_thr)
             
             #Removes duplicates from sea wave and RMS data
             clean_df_seism, clean_df_sea=fn.DeleteDuplicates( clean_df_seism, clean_df_sea)
                            
             #Checks the symmetry/asymmetry of RMS values.
             #Eventually, it applies a correction to obtain
             #normally distributed data       
             clean_df_seism,idx_sks, lmbda, alpha=fn.SkewnessCorrection(clean_df_seism,skew_limit)
             
             #Deletes all rows in the seismic and sea wave data affcted by earthquakes
             if earth_check==True:
                 clean_df_seism, clean_df_sea= fn.EarthQuakeCatalog(clean_df_seism, clean_df_sea,starttime,endtime,eqCatalog,MagMed,MagWorld, mLat, MLat, mLon, MLon, hours_del)
            
             
             #Save the pre-processed data for the learning of the model (optionally) 
             if save_data==True:
                 fn.SaveData(folder_save, file_out, clean_df_seism, clean_df_sea, colmask, org_shape, colname, idx_sks, lmbda, alpha)
         
         #Fixing the end of the whole process,
         #writing this information in the logfile and closing it
         logging.info("OK!Pre-processing data terminated.") 
         logging.shutdown()
         return  clean_df_seism, clean_df_sea, colmask, org_shape, colname, idx_sks, lmbda, alpha
        
    except Exception:
        #Fixing the end of the whole process,
        #writing this information in the logfile and closing it
        logging.info("Warning!Something was gone wrong.") 
        logging.shutdown()

###############################################################################
def LearningModel(filename,clean_df_seism, clean_df_sea, colmask, org_shape,colname, CrossVal=False,save_model=False):
    '''
     learns the model predecting sea state date from the RMS time series 
        
            Parameters:
                    filename (str): name of the configuration file 
                    clean_df_seism (obj): dataframe containing pre-processed sea times series for each position of the area of interest  
                    clean_df_sea (obj): dataframe containing  pre-processed RMS times series for each frequency band, station and channel
                    colmask (ndarray): indexes of non-NaN values of the sea wave height data used during the learning of the model 
                    org_shape (ndarray): vector containing the original shape of the sea wave height matrix used during the learning of the model 
                    colname (obj): header of  RMS data 
                    CrossVal (bool): if True, the performance of the model was evaluated through cross-validation 
                    save_model (bool): if True, the predicted model is saved as .sav file 

            Returns:
                model (obj): predictive model
                scaler_x, scaler_y (obj): scaling object containing the direct/inverse transformations
                    
                
    '''
   
    # Try the code until the return of an error
    try:
       
        #Create the log file and its configuration 
        logging.basicConfig( level=logging.DEBUG,
                        format="%(asctime)s %(message)s",force=True,handlers=[logging.FileHandler("log_Model.txt"),logging.StreamHandler()])

        logging.getLogger('matplotlib.font_manager').disabled = True
        
        logging.info("OK!Starting learning model.") 

        #Read all parameters from the configuration file 
        network, list_sta, list_chan, lat1, lat2,lon1, lon2, rms_thr, row_thr, skew_limit, percentage, nchunks, kfolds, RF_max_depth, RF_n_estimator, RF_max_features, KNN_n_neighbors, KNN_weights, LGB_n_estimators, LGB_learning_rates, LGB_max_depth, LGB_num_leaves, model_flag, file_out, file_out_err, folder_save, filemodel, eqCatalog, MagMed, MagWorld, mLat,MLat, mLon, MLon, starttime, endtime, hours_del = fn.ReadConfig(filename)
        
        #Checks if it is available an predictive model in the output folder.
        #If true the code load only the results  without repeats any step
        #of the learning.
        filePrePro = pathlib.Path(folder_save+'/'+filemodel+'_'+model_flag+'.sav')
        if filePrePro.exists():
            logging.info("OK!Predictive model file was found.")
            model,scaler_x,scaler_y=fn.LoadFinalModel(folder_save, filemodel, model_flag)
        else:
            logging.info("Warning!Predictive model file was not found.")
            #It's an optional routine
            if CrossVal==True:
                #Update log file about the onset of the Cross-validation process
                logging.info("OK!Cross-validation process was starting.")
                
                #Splits the dataset into one/more training and test datasets.
                list_of_dataset, T, scaler_x, scaler_y=fn.SplitData(clean_df_seism, clean_df_sea, percentage,nchunks, kfolds)
                
                #Learning the model by using cross-validation datasets  
                Ypred,YTest=fn.LearnPredictData(list_of_dataset,kfolds, scaler_x, scaler_y,RF_max_depth, RF_n_estimator, RF_max_features, KNN_n_neighbors, KNN_weights, LGB_n_estimators, LGB_learning_rates, LGB_max_depth, LGB_num_leaves, model_flag)            
                
                #Evaluates the performance of the model by calculating errors
                #between real and predicted sea wave data
                Sea, SeaExtr,ErrN, ErrMeanN, ErrRelN, ErrRelMeanN, ErrExtr, ErrExtrMean, ErrRelExtr, ErrRelExtrMean,R2, R2_N, R2_E=fn.EvaluatePerformance(Ypred,YTest)  
                
                #Saves errors models 
                fn.SaveResultsPerformance(folder_save,file_out_err, Sea, SeaExtr, ErrN, ErrMeanN, ErrRelN, ErrRelMeanN, ErrExtr, ErrExtrMean, ErrRelExtr, ErrRelExtrMean,R2, R2_N, R2_E)
                            
                #Displays and saves errors diagrams
                fn.PlotResultsPerformance(folder_save,model_flag,org_shape,colmask, Sea, SeaExtr, ErrN, ErrMeanN, ErrRelN, ErrRelMeanN, ErrExtr, ErrExtrMean, ErrRelExtr, ErrRelExtrMean,R2, R2_N, R2_E, lat1, lat2, lon1, lon2)
            
            #Learns the final model 
            model, scaler_x, scaler_y = fn.LearnFinalModel(clean_df_seism, clean_df_sea, RF_max_depth, RF_n_estimator, RF_max_features, KNN_n_neighbors, KNN_weights, LGB_n_estimators, LGB_learning_rates, LGB_max_depth, LGB_num_leaves, model_flag)
     
            if save_model:
               fn.SaveFinalModel(folder_save, filemodel, model,scaler_x,scaler_y,model_flag)
        #Update log file about the finish of the operation 
        logging.info("OK!Learning model terminated.") 
        logging.shutdown()
        return model,scaler_x,scaler_y    
        
    except Exception:
        #Fixing the end of the whole process,
        #writing this information in the logfile and closing it
        logging.info("Warning!Something was gone wrong during learning.") 
        logging.shutdown()