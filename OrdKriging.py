# Compute ordinary kriging

import numpy as np
from pykrige.rk import Krige
#from pykrige.compat import GridSearchCV
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import GridSearchCV

def OrdKriging(lon_sta, lat_sta, res, lon_grid, lat_grid):
    ''' 
    Ordinary kriging to interpolate residuals from the multiple linear regression analysis using the best model among those tested in param_dict
    Input
	lon_sta = longitude of the stations => must be expressed in the 0-360°range 
    	lat_sta = latitude of the stations
    	res = residual from the multiple linear regression model
    	lon_grid = longitude of the grid
    	lat_grid = latitude of the grid
    Output
	param_best: parameters of the best model
	r2_best:    R2 for the best model
        z: 	    krigged residuals
	
    '''

    # Step 1. Find the best model among the choices tested in param_dict
    param_dict = {"method": ["ordinary"],
                  "variogram_model": ["linear", "power", "gaussian", "spherical"],
                  "nlags": [2, 4, 6, 8, 10],
                  "weight": [True, False],
                  "coordinates_type": ["geographic"],
    }
    estimator = GridSearchCV(Krige(), param_dict, verbose=True, return_train_score=True)
    
    # Run the GridSearch
    print('Warning: lon_sta must be expressed in the 0-360° range')
    coords = np.column_stack((lon_sta, lat_sta))
    estimator.fit(X=coords, y=res)
    
    # Print some results
    #if hasattr(estimator, 'best_score_'):
    #  print('best_score R2 = {:.3f}'.format(estimator.best_score_))
    #  print('best_params = ', estimator.best_params_)
    #print('\nCV results::')
    #if hasattr(estimator, 'cv_results_'):
    #  for key in ['mean_test_score', 'mean_train_score', 'param_method', 'param_variogram_model']:
    #    print(' - {} : {}'.format(key, estimator.cv_results_[key]))
    r2_best = estimator.best_score_
    param_best = estimator.best_params_

    # Step 2. Apply the best model from step 1 
    KRIG = OrdinaryKriging(x=lon_sta, y=lat_sta, z=res,
           variogram_model=estimator.best_params_['variogram_model'],
           nlags=estimator.best_params_['nlags'],
           weight=estimator.best_params_['weight'],
           verbose=False,
           enable_plotting=False,
           coordinates_type=estimator.best_params_['coordinates_type']
           )
    # Interpolate onto grid
    z, ss = KRIG.execute("points", lon_grid, lat_grid)

  
    # Step 3. Return results
    return param_best, r2_best, z 
   
