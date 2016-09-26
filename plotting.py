from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
import random


def cross_validate(test_data,bandwidths,n_folds=5):
    params = {'bandwidth': bandwidths}
    kf = KFold(n=len(test_data),n_folds=n_folds,shuffle=True,random_state=0)
    grid = GridSearchCV(KernelDensity(), params,cv=kf)
    grid.fit(test_data)
    return grid.best_estimator_.bandwidth,grid


def KDE(data,histtype='step',color='b',alpha=1,linewidth=2,linestyle='solid',
        x_range=None,bandwidth=None,n_folds=5,printout=False):
  
    '''Plotting code for plotting a 1d KDE. Arguments listed below:
    
    data: data to be plotted
    histtype: 'step' or 'stepfilled'
    color: same as np.hist
    alpha: same as np.hist
    linewidth: same as np.hist
    linestye: same as np.hist
    x_range: x-axis range, can cut out values here.
    bandwidth: arbitrarily set as None, can override CV metjod here.
    n_folds: set as 5, number of folds for the CV method.
    printout: can print time taken etc.
    '''
    
    start_time = time.time()
    random.seed(0)
    
    select_finite = np.isfinite(data)
    data = data[select_finite]
    data = data.astype(float64)
    
    if x_range == None:
        x_range = [np.min(data),np.max(data)]    
    
    select_range =  (data >= x_range[0]) & (data <= x_range[1]) 
    data = data[select_range]
    data = data[:,np.newaxis]
    x_plot = np.linspace(x_range[0],x_range[1],100)[:,np.newaxis]

    test_data = preprocessing.scale(data)
    N_max = 1000
    if len(test_data) >= N_max:
        np.random.shuffle(test_data)
        test_data = test_data[:N_max]
        
    if bandwidth == None:
        N_steps = 20
        bandwidths = np.logspace(-2,0,N_steps)
        bandwidth,grid = cross_validate(test_data,bandwidths,n_folds)
        i = (np.where(bandwidths == bandwidth))[0]
        bandwidths = np.linspace(bandwidths[i-1],bandwidths[i+1],N_steps)
        bandwidth,grid = cross_validate(test_data,bandwidths,n_folds)
        bandwidth = bandwidth*np.std(data)
        if printout:
            print('Optimal bandwidth found: {0:.3f}'.format(bandwidth))
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    y = kde.score_samples(x_plot)
    y_plot = np.exp(y)
    x_plot = x_plot.squeeze()
    
    if histtype == 'step':
        _ = plt.plot(x_plot,y_plot,
                     color=color,linewidth=linewidth,linestyle=linestyle,alpha=alpha)
    elif histtype == 'stepfilled':
        _ = plt.fill_between(x_plot,0,y_plot,
                             color=color,alpha=alpha)
    else:
        print("histtype must be 'step' or 'stepfilled'; using 'step'")
        _ = plt.plot(x_plot,y_plot,
                     color=color,linewidth=linewidth,linestyle=linestyle,alpha=alpha)
    if printout:
        print("{0:.1f} seconds in total".format(time.time() - start_time))
        
    return x_plot,y_plot,bandwidth