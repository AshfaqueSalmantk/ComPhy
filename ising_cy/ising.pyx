#~!/usr/bin/ComPhys/python3
from __future__ import division
import cython
import numpy as np
cimport numpy as np
import time
from libc.math cimport sqrt,fabs, int, exp



#@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
#@cython.nonecheck(False)
cdef class Ising2D():

    cdef int N, nt, eqSteps,mcSteps
    cdef readonly np.ndarray expfunc


    def __init__(self,N,nt,eqSteps,mcSteps):
        '''
        args:
        -----
        N : lattice dimension
        nt: number of temperature points
        eqSteps: sweeps for thermalization
        mcSteps: sweeps for calculation
        '''
        self.N = N
        self.nt = nt
        self.eqSteps = eqSteps
        self.mcSteps = mcSteps


    cpdef simTwoD(self, double Temp):
        ''' calculate final microstate for a given temperature'''
        N = self.N
        cdef np.ndarray[long,ndim=2] state
        cdef double n1,M1,mag
        cdef int i

        state = 2*np.random.randint(2,size=(N,N))-1
        n1 = 1.0/(N*N*self.mcSteps)

        for i in range(self.eqSteps):
            self.mcmove(state,1/Temp)

        for i in range(self.mcSteps):
            self.mcmove(state,1/Temp)
            mag = self.calcMag(state)
            M1 = M1 + mag
        print(state)

        return M1*n1

    cdef int mcmove(self, np.ndarray[long,ndim=2] state, double beta):

        cdef int i,j,a,b,dE
        cdef double [:] expfunc = self.expfunc

        expfunc[1] = exp(-4*beta)
        expfunc[2] = expfunc[1]*expfunc[1]

        for i in range(self.N):
            for j in range(self.N):
                a,b = np.random.randint(0,self.N),np.random.randint(0,self.N)

                state[0,b] = state[self.N,b]; state[self.N+1,b] = state[1,b];
                state[a,0] = state[a,self.N]; state[a,self.N+1] = state[a,1];

                dE = 2*state[a,b]*(state[a+1,b]+state[a-1,b]+state[a,b+1]+state[a,b-1])

                if (dE <=0 or (np.random.random()<expfunc[dE//4])):
                        state[a,b] = -state[a,b]

        return 0

    cdef double calcMag(self,np.ndarray[long,ndim=2] state):

        return np.sum(state)





















