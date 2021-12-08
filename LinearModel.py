# Compute multiple linear regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from sklearn.model_selection import LeaveOneOut, cross_val_score
from math import sqrt

def LinearModel(fileIn1,fileIn2):
    ''' 
    Compute multiple linear regression
    fileIn1 = temperature
    fileIn2 = predictors
    '''
    y = fileIn1
    x = fileIn2
    # Regression all stations
    regr = linear_model.LinearRegression()   # Create linear regression object
    regr.fit(x, y)                           # Build model
    regcoef = regr.coef_
    intercept = regr.intercept_
    ysim = regr.predict(x)                   # Apply model to simulate Temp
    mse = mean_squared_error(y, ysim)    
    rmse = sqrt(mean_squared_error(y, ysim))
    r2 = r2_score(y, ysim)
    r2adj = 1-(1-r2)*(len(y)-1)/(len(y)-len(x.columns)-1)
    # Cross-validation
    cv = LeaveOneOut()  # Method Leave One Out
    mae_LOOCV = cross_val_score(regr, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    rmse_LOOCV = cross_val_score(regr, x, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    return regcoef, intercept, ysim, mse, rmse, r2, r2adj, mae_LOOCV, rmse_LOOCV
   
