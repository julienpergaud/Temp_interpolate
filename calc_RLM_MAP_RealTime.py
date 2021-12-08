
#############################################################################################
# Import libraries

import pandas as pd
import numpy as np
import MLRnKRIG as model

#############################################################################################



#############################################################################################
# Settings

# Select subperiod :Each subperiod = 4999 timesteps
#case = ['2019-01-13 04:00:00', '2019-02-02 02:00:00', '2016-03-04 06:00:00', '2017-04-27 12:00:00', '2015-05-10 07:00:00', '2019-06-15 10:00:00', '2014-07-18 07:00:00', '2017-08-15 07:00:00', '2015-09-21 09:00:00', '2015-10-30 22:00:00', '2014-11-22 11:00:00', '2015-12-31 22:00:00']
case = ['2019-01-13 20:00:00']

# Minimum number of stations with no missing value for interpolating temp
mini_nb_sta = 40


# TODO extract données from hobolink and create dataframe with data in index and stations as columns






# Path in/out
dirin_temp = './'
dirin_pred = './'
dirout = './Results/'


#############################################################################################



#############################################################################################
# Load data

# Hourly temp measurec-d by the MUSTARDijon netork
MUSTARDijon = pd.read_csv(dirin_temp+'TEMPICU_2014_2020.csv', sep=',', quotechar='"', index_col=0)


def spatialMustard(case,MUSTARDijon,dirin_temp,dirin_pred,dirout,mini_nb_sta):
# Predictors
# station's predictors
    coord_list = ['LAT', 'LON']
    pred_NO_BDTOPOnorPLEIADE = ['alt1mean', 'DISTANCE_C']
    pred_list = ['alt1mean',
       'DISTANCE_C', 'FORET+EAU',
       'GRANDECULTURE', 'VEGETATIONBASSE']

    fileIn = dirin_pred+'descriptors_sta_TheMa.csv'
    pred_sta = pd.read_csv(fileIn, sep=';', quotechar='"')
    pred_sta.index = pred_sta.iloc[:,0]
    pred_sta = pred_sta.drop(index='S51')  # Delete S51 => MF Longvic
    pred_sta.index.name = 'Station'
    pred_sta = pred_sta.iloc[:,1::]
    coord_sta = pred_sta[pred_sta.columns & coord_list]  # Select coordinate
    pred_sta = pred_sta[pred_sta.columns & pred_list]  # Select descriptors
# grid's predictors
    fileIn = dirin_pred+'descriptors_grid_TheMa.csv'
    pred_grid = pd.read_csv(fileIn, sep=';', quotechar='"')
    pred_grid = pred_grid.iloc[:,1::]
    coord_grid = pred_grid[pred_grid.columns & coord_list]  # Select coordinates
    pred_grid = pred_grid[pred_grid.columns & pred_list]  # Select descriptors

# Put NaN for grid point where no data for BDTOPO-PLEIADE product
    index_NaN = pred_grid.loc[pd.isna(pred_grid['FORET+EAU']), :].index
    for pred in pred_NO_BDTOPOnorPLEIADE:
        pred_grid[pred][index_NaN] = np.nan

#############################################################################################



#############################################################################################

    model.MLRnKRIG(case, MUSTARDijon, mini_nb_sta, dirout, pred_sta, coord_sta, pred_grid, coord_grid, pred_list, diag=True)

#############################################################################################


spatialMustard(case,MUSTARDijon,dirin_temp,dirin_pred,dirout,mini_nb_sta)

