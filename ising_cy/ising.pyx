#~!/usr/bin/ComPhys/python3
from __future__ import division
import cython
import numpy as np
cimport numpy as np
import time
from libc.math cimport sqrt,fabs, int, exp



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
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
        self.expfunc = np.zeros((2),dtype=np.float64)


    cpdef twoD(self, double iT):
        cdef int i,j
        cdef double M1,n1,Mag



        n1 = 1.0/(self.N*self.N*self.mcSteps)
        M1 = 0

        # initialise the state using numpy
        state = 2*np.random.randint(2,size=(self.N,self.N))-1


        # thermalization
        for i in range(self.eqSteps):
            self.mcMove(state,1/iT)

        # montecarlo steps for calculation of order parameters
        for j in range(self.mcSteps):
            self.mcMove(state,1/iT)

            Mag = self.calcMag(state)
            M1 = M1 + Mag

        return M1*n1

    cdef mcMove(self, state , double beta):

        cdef int i,j,a,b,cost
        cdef double [:] expfunc = self.expfunc

        expfunc[1] = exp(-4*beta)
        expfunc[2] = expfunc[1]*expfunc[1]

        for i in range(self.N):
            for j in range(self.N):
                a,b = np.random.randint(self.N),np.random.randint(self.N)

                cost = 2*state[a,b]*(state[(a+1)%self.N,b]+state[(a-1)%self.N,b]+state[a,(b+1)%self.N]+state[a,(b-1)%self.N])

                if cost <= 0 or np.random.rand() < expfunc[cost/4]:
                    state[a,b] = -state[a,b]

        return state


    cdef double calcMag(self, state):

        return np.sum(state)





