#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

def parallel_sim(func,temperatures):
    '''
    func:
    refernce to the func which calculate mc for a time step
    temperatures : list of temperatures for simulations is performed
    '''
    pool = mp.Pool(mp.cpu_count())
    mag = pool.map(func,temperatures)
    pool.close()
    pool.join()
    return mag

def plotData(x,y,x_label,y_label):
    plt.scatter(x,y, marker='o',color='red',s=50)
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    plt.axis('tight')

def plotState(state,i,N):
    ''' plot the current microstate as a grid
    state : current state to plot
    i     : ith time
    N     : gride size
    n_    : subplot number
    '''
    X, Y = np.meshgrid(range(N),range(N))
    plt.pcolormesh(X,Y,state,cmap=plt.cm.RdBu)
    plt.title(f" Time:{i}")
    plt.axis('tight')
