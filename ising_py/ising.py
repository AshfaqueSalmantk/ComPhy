#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import utils
import multiprocessing as mp

class Ising():

    def __init__(self,N=10,nt=100,eqSweep=2**8,mcSweep=2**9):
        '''
        Arguments:
        ---------
        N       : size of the lattice, NxN
        nt      : number of temperature points
        eqSweep : sweeps for thermallizations
        mcSweep : sweeps for calculation
        '''
        self.N          = N
        self.nt         = nt
        self.eqSweep    = eqSweep
        self.mcSweep    = mcSweep

    def mcMove(self,state,beta):
        '''
        Arguments:
        ---------
        beta : 1/temperature
        calculate the state after one montecarlo sweep
        '''
        # to avoid NxN calls to exponential function, precalculate it, only two values are possible for change in energy, 4J and 8J
        exp1 = np.exp(-4*beta)
        self.exp = [ exp1, exp1*exp1]

        for i in range(self.N):
            for j in range(self.N):
                a,b = np.random.randint(0,self.N),np.random.randint(0,self.N)

                # take the specific cell
                s = state[a,b]

                # calculate the energy change
                nb = state[ (a+1)%self.N,b] + state[ (a-1)%self.N,b] + state[ a,(b+1)%self.N] +state[ a,(b-1)%self.N]
                dE = 2*s*nb

                if dE < 0 or (np.random.rand() < self.exp[ int(dE/4-1)]):
                                  state[a,b] = -state[a,b]
        return state


    def _calcMag(self,state):
        ''' find the magnetization of the current microstate '''
        return np.sum(state)


    def sim(self,Temp):
        ''' simulate for nt number of points between Ti (initial temp) and Tf (final temp)'''

        #initialize the state
        state = 2*np.random.randint(2,size=(self.N,self.N))-1

        n1 = 1.0/(self.N*self.N*self.mcSweep)
        M1 = 0
            # thermalization sweep
        for i in range(self.eqSweep):
            self.mcMove(state,1.0/Temp)
        # montecarlo sweep after reaching equilibrium
        for j in range(self.mcSweep):
            self.mcMove(state,1.0/Temp)
            mag = self._calcMag(state)
            M1 = M1 + mag

        return M1*n1



if __name__ == "__main__":

    # simulation parameters
    N  = 10
    nt = 32
    eqSweep = 2**8
    mcSweep = 2**9

    # temperature points
    T = np.linspace(1,4,nt)

    # initialize the class with parameters
    rm = Ising(N,nt,eqSweep,mcSweep)

    # time calculating for the program
    start= time.time()
    Mag  = utils.parallel_sim(rm.sim,T)
    end  = time.time()
    print(f"time taken for normal computation:{end-start}")

    #plot the data
    f = utils.plt.figure(1,figsize=(8,6))
    utils.plotData(T,Mag,'temp','mag')
    utils.plt.show()












