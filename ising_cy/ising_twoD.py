import numpy as np
import matplotlib.pyplot as plt
import ising
import multiprocessing as mp
import time

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

def main():

    N  = 10
    nt = 32
    eqSweep = 2**8
    mcSweep = 2**9

    # temperature points
    T = np.linspace(1,4,nt)

    # initialize the class with parameters
    Ising = ising.Ising2D(N,nt,eqSweep,mcSweep)

    # time calculating for the program
    start= time.time()
    Mag  = parallel_sim(Ising.simTwoD,T)
    end  = time.time()
    print(f"time taken for normal computation:{end-start}")

    #plot the data
    f = plt.figure(1,figsize=(8,6))
    plt.scatter(T,Mag)
    plt.savefig("mag.png")
    plt.show()

if __name__ == "__main__":
    main()
