# Function for regression-kriging of hourly temperatures

#############################################################################################
# Import libraries

import pandas as pd
import numpy as np
import _pickle as pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# My functions
import PearsonCorrTempvsPred as corr
import LinearModel as LM
import OrdKriging as OK

#############################################################################################



#############################################################################################

def MLRnKRIG(case, temp_date, MinNbSta, dirout, pred_sta, coord_sta, pred_grid, coord_grid, pred_list, diag=False):
    '''
    Compute multiple linear regression and krige MLR errors
    case       => one date or list of dates (e.g., '2019-01-13 04:00:00')
    temp_date  => temperature measured by the MUSTARDijon network
    dirout     => directory where results will be stored
    pred_sta   => dataframe with predictors for each station (used to construct the MLR model) 
    coord_sta  => geographical coordinates of each station
    pred_grid  => dataframe with predictors for each grid point (used to apply the MLR model)
    coord_grid => geographical coordinates of each grid point
    diag=False => no diag returned (by default)
    pred_list  => list of predictors
    diag=True  => returns diag
		STATS_CROSS_CORR             => correlation between each predictor and temperature 
		Avail_Station		     => Number of stations used to construct the model
		STATS_MLR_REGCOEFFnINTERCEPT => regression coefficients and intercept of the MLR model
		STATS_MLR                    => MLR model skill (MSE, RMSE, R2, R2-Adj)
                VIF_MLR			     => Variance inflation factor for each MLR model
		STATS_LOOCV_MAE              => cross validation (Leave One Out) of the MLR model => MAE
		STATS_LOOCV_RMSE              => cross validation (Leave One Out) of the MLR model => RMSE
                RESIDUAL_MLR                 => MLR residuals (error at each station = measured temp minus predicted temp)
		STATS_KRIG                   => R2 of the best kriging model among the 40 tested
		Best_param_KRIG		     => Parameters of the best kriging model

		TEMP_MLR_grid                => predicted temperature for each grid point 
		TEMP_KRIG_grid               => krigged MLR residual for each grid point
					     => Note: TEMP_MLR_grid + TEMP_KRIG_grid = best temperature map
    '''
#############################################################################################



#############################################################################################
# Loop case

    for ind_cas, cas in enumerate(case):

      print(cas)
      cas_tit = cas
      cas_tit = cas_tit.replace(' ', '_')
      cas_tit = cas_tit.replace(':', '_')

#############################################################################################



#############################################################################################
# Select date

      date_cas = temp_date.index == cas
      data = temp_date.loc[date_cas]

# Info header & check data type & date
      #print(data.head)
      #print(data['Time'].dtypes)
      #print(data['S01'].dtypes)

      start_date = min(data.index)
      finish_date = max(data.index)
      print ('Period: '+start_date+' -- '+finish_date)

#############################################################################################



#############################################################################################
# Create dictionaries or dataframes to store the results

# Dictionary with the list of stations used for the regression at each timestep
      Avail_Station = {}

# Retained predictors
      Retained_Pred = {}

# Cross correlation between Temp and each predictor
      STATS_CROSS_CORR  = pd.DataFrame(index = pd.date_range(start_date, freq='H', periods=len(data)), columns = pred_list)
      STATS_CROSS_CORR.index.name = 'Date'

# Variance inflation factor
      VIF_MLR = {}

################################################


################################################
# MLR

# Regression coefficients for each timestep
      colname = list(pred_list)
      colname.append('Intercept')
      STATS_MLR_REGCOEFFnINTERCEPT  = pd.DataFrame(index = pd.date_range(start_date, freq='H', periods=len(data)), columns = colname)
      STATS_MLR_REGCOEFFnINTERCEPT.index.name = 'Date'
      del colname

# Statistical metrics of the MLR model
      STATS_MLR = pd.DataFrame(index = pd.date_range(start_date, freq='H', periods=len(data)), columns = ['MSE', 'RMSE', 'R2', 'R2-Adj'])
      STATS_MLR.index.name = 'Date'

# Predicted Temp => Sta
      colname = list(data.columns)
      TEMP_MLR = pd.DataFrame(index = pd.date_range(start_date, freq='H', periods=len(data)), columns = colname)
      TEMP_MLR.index.name = 'Date'
      del colname

# Predicted Temp => Grid
      TEMP_MLR_grid = pd.DataFrame(index = pred_grid.index, columns = pd.date_range(start_date, freq='H', periods=len(data)))
      TEMP_MLR_grid.index.name = 'Gridpoint'

# Residuals
      colname = list(data.columns)
      RESIDUAL_MLR = pd.DataFrame(index = pd.date_range(start_date, freq='H', periods=len(data)), columns = colname)
      RESIDUAL_MLR.index.name = 'Date'
      del colname

# Statistical metrics of the crossvalidation
      colname = list(data.columns)
      STATS_LOOCV_MAE = pd.DataFrame(index = pd.date_range(start_date, freq='H', periods=len(data)), columns = colname)
      STATS_LOOCV_MAE.index.name = 'Date'
      STATS_LOOCV_RMSE = pd.DataFrame(index = pd.date_range(start_date, freq='H', periods=len(data)), columns = colname)
      STATS_LOOCV_RMSE.index.name = 'Date'

################################################


################################################
# Kriging

# Statistical metrics of the Kriging model
      STATS_KRIG = pd.DataFrame(index = pd.date_range(start_date, freq='H', periods=len(data)), columns = ['R2'])
      STATS_KRIG.index.name = 'Date'

# Best param kriging
      Best_param_KRIG = {}

# Krigged residuals => Grid
      TEMP_KRIG_grid = pd.DataFrame(index = pred_grid.index, columns = pd.date_range(start_date, freq='H', periods=len(data)))
      TEMP_KRIG_grid.index.name = 'Gridpoint'

#############################################################################################




#############################################################################################
# Build a dataframe with Station name, Temp & predictors 

      Y = data.T
      print(data)
      Y.set_axis(['Temp'], axis=1, inplace=True)
      print(Y.index)
      X = pred_sta
      print(X.index)

      df = pd.merge(Y, X, how='inner',left_index=True,right_index=True)
      print(df)
      df.reset_index(inplace=True)
      df = df.rename(columns = {'index':'Station'})
      print(df)
      df['Temp'] = df['Temp'].astype(float)
      del Y
      del X

#############################################################################################



#############################################################################################
# Select stations with no missing values

      dataReg = df.dropna()

      Avail_Station = dataReg['Station']

#############################################################################################



#############################################################################################
# Regression computed only if number of stations > mini_nb_sta

      if len(Avail_Station)>MinNbSta:

# Select target (y) and predictors (x)
        y = dataReg['Temp']
        x = dataReg[pred_list]

# Compute correlation between Temp (y) and predictors (x)
        STATS_CROSS_CORR.loc[cas,:] = corr.PearsonCorrTempvsPred(y, x)

# Predictor's selection: Wrapper method with Backward Elimination
# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
        retained_predictors = pred_list
        Retained_Pred = retained_predictors

        if len(retained_predictors)>0:

# Variance inflation factor
          toto = add_constant(x[retained_predictors])
          VIF_MLR[cas] = pd.Series([variance_inflation_factor(toto.values, i)
          for i in range(toto.shape[1])],
              index=toto.columns)

# Multiple linear regression => all available stations accounted for
          regcoef, intercept, ysim, mse, rmse, r2, r2adj, mae_LOOCV, rmse_LOOCV = LM.LinearModel(y, x[retained_predictors])

      # Model robustness
          STATS_MLR.loc[cas,'MSE'] = mse
          STATS_MLR.loc[cas,'RMSE'] = rmse
          STATS_MLR.loc[cas,'R2'] = r2
          STATS_MLR.loc[cas,'R2-Adj'] = r2adj

      # Reg coeff & intercept
          i = 0
          for pred in retained_predictors:
            STATS_MLR_REGCOEFFnINTERCEPT.loc[cas, pred] = regcoef[i]
            i = i + 1
          STATS_MLR_REGCOEFFnINTERCEPT.loc[cas, 'Intercept'] = intercept
    
      # Predicted Temp & residual
          for ind_sta, sta in enumerate(Avail_Station):
            ind_sta2 = int(y.index[ind_sta])
            TEMP_MLR.loc[cas,sta] = y[ind_sta2]
            RESIDUAL_MLR.loc[cas,sta] = y[ind_sta2] - ysim[ind_sta]

      # Cross validation
          for ind_sta, sta in enumerate(Avail_Station):
            STATS_LOOCV_MAE.loc[cas,sta] = mae_LOOCV[ind_sta]
            STATS_LOOCV_RMSE.loc[cas,sta] = rmse_LOOCV[ind_sta]

# Apply MLR model to grid
          toto = np.zeros((len(pred_grid), len(retained_predictors)))
          for ind, pred in enumerate(retained_predictors):
            #print(pred)
            toto[:,ind] = pred_grid[pred]*regcoef[ind]
          tutu = np.sum(toto, axis=1)+intercept
          TEMP_MLR_grid.loc[:,cas] = tutu
          del tutu

# Kriging residual
          res = y - ysim
          param_best, r2_best, z = OK.OrdKriging(coord_sta.loc[Avail_Station]['LON']+180, coord_sta.loc[Avail_Station]['LAT'], res, coord_grid['LON']+180, coord_grid['LAT'])
          Best_param_KRIG = param_best
          STATS_KRIG.loc[cas,'R2'] = r2_best
          TEMP_KRIG_grid.loc[:,cas] = z
      
          # Clean
          del regcoef
          del intercept
          del ysim
          del mse
          del rmse
          del r2 
          del mae_LOOCV
          del rmse_LOOCV
          del res
          del param_best
          del r2_best
          del z

#############################################################################################



#############################################################################################
# Save results in csv/pkl

          if diag==True:

              STATS_CROSS_CORR.to_csv(dirout+'STATS_CROSS_CORR_'+cas_tit+'.csv', index = True, header = True)

              STATS_MLR_REGCOEFFnINTERCEPT.to_csv(dirout+'STATS_MLR_REGCOEFFnINTERCEPT_'+cas_tit+'.csv', index = True, header = True)

              STATS_MLR.to_csv(dirout+'STATS_MLR_'+cas_tit+'.csv', index = True, header = True)

              TEMP_MLR.to_csv(dirout+'TEMP_MLR_'+cas_tit+'.csv', index = True, header = True)

              TEMP_MLR_grid.to_csv(dirout+'TEMP_MLR_grid_'+cas_tit+'.csv', index = True, header = True)

              RESIDUAL_MLR.to_csv(dirout+'RESIDUAL_MLR_'+cas_tit+'.csv', index = True, header = True)

              STATS_LOOCV_MAE.to_csv(dirout+'STATS_LOOCV_MAE_'+cas_tit+'.csv', index = True, header = True)
              
              STATS_LOOCV_RMSE.to_csv(dirout+'STATS_LOOCV_RMSE_'+cas_tit+'.csv', index = True, header = True)

              STATS_KRIG.to_csv(dirout+'STATS_KRIG_'+cas_tit+'.csv', index = True, header = True)

              TEMP_KRIG_grid.to_csv(dirout+'TEMP_KRIG_grid_'+cas_tit+'.csv', index = True, header = True)
           
              file = open(dirout+'Avail_Station_'+cas_tit+'.pkl', 'wb')
              pickle.dump(Avail_Station, file)
              file.close()

              file = open(dirout+'Retained_Predictor_'+cas_tit+'.pkl', 'wb')
              pickle.dump(Retained_Pred, file)
              file.close()

              file = open(dirout+'VIF_MLR_exp_'+cas_tit+'.pkl', 'wb')
              pickle.dump(VIF_MLR, file)
              file.close()
              
              file = open(dirout+'Best_param_KRIG_'+cas_tit+'.pkl', 'wb')
              pickle.dump(Best_param_KRIG, file)
              file.close()

#############################################################################################


