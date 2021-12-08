# Compute Pearson correlation between temperature & predictors

def PearsonCorrTempvsPred(fileIn1,fileIn2):
    ''' 
    Compute Pearson correlation between temperature and each predictor
    fileIn1 = temperature
    fileIn2 = predictors
    '''
    import numpy as np    
    from scipy import stats 
    y = fileIn1
    x = fileIn2
    PREDICTORS = x.columns
    corr = np.zeros(len(PREDICTORS))
    i=0
    for pred in PREDICTORS:
        r, pval = stats.pearsonr(y, x[pred])
        corr[i] = r
        i = i+1 
    return corr
