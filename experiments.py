import numpy as np
import util
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


#Unlikely to work well on its own because of the way Lanczos
#iteration works. It looks for invariant subspaces by
#repeatedly applying A, which amplifes its _largest_
#eigenvalues. Setting `which='SM'` will have the Lanczos method
#attempt to drive the convergence towards smaller eigenvalues
#by sorting intermediate steps and taking the smaller ones.
def plain_lanczos(m,k,tol=1e-5):
    A = util.CounterOperator(util.make_symmetric(m))
    w,V = spla.eigsh(A,k,which='SM',tol=tol)
    return w,A.nevals()


#Essentially same as above, but we "warm up" by
#repeatedly applying a krylov method in power iteration
#loop. This sets up A polynomial acceleration where 
#the krylov _solver_ damps out the highest eigenmodes
#and amplifies & separates out smaller eigenmodes.
def warmup_lanczos(m,k,tol=1e-5):
    A = util.CounterOperator(util.make_symmetric(m))
    # Get approximate invariant subspace
    K = util.ipower(A.A,k,maxiter=100,inner_maxiter=100)
    # Warm up lanczos by having v0 have every component of K
    v0 = K @ np.ones(k)
    w,V = spla.eigsh(A,k,which='SM',v0=v0,tol=tol)
    # I omit the cost of power iteration because mostly interested in
    # behavior of lanczos
    return w,A.nevals()


#Use lanczos in shift-invert mode with shift of 0
#solve the resulting systems with MINRES
def inverse_lanczos(m,k,tol=1e-5):
    A = util.SolverOperator(util.make_symmetric(m))
    w,V = spla.eigsh(A.counterA,k,tol=tol,sigma=0.0,OPinv=A)
    return w,A.nevals()


#Same as above but using power iteration warmup
#Use lanczos in shift-invert mode with shift of 0
#solve the resulting systems with MINRES
def warmup_inverse_lanczos(m,k,tol=1e-5):
    A = util.SolverOperator(util.make_symmetric(m))
    K = util.ipower(A.A,k,maxiter=1000,inner_maxiter=10)
    # Warm up lanczos by having v0 have every component of K
    v0 = K @ np.ones(k)
    w,V = spla.eigsh(A.counterA,k,tol=tol,sigma=0.0,OPinv=A,v0=v0)
    return w,A.nevals()

def lobpcg_exact(m,k,tol=1e-5):
    seed=243
    rng = np.random.default_rng(seed)
    Ae = util.make_symmetric(m)
    A = util.CounterOperator(Ae)
    luA=spla.splu(Ae)
    X = rng.uniform(-1,1,size=(m,k))
    w,V = spla.lobpcg(A,X,M=lambda y : luA.solve(y), tol=tol,largest=False,maxiter=1000)
    return w,A.nevals()


def lobpcg(m,k,tol=1e-5):
    seed=243
    rng = np.random.default_rng(seed)
    A = util.CounterOperator(util.make_symmetric(m))
    X = rng.uniform(-1,1,size=(m,k))
    w,V = spla.lobpcg(A,X,tol=tol,largest=False,maxiter=200)
    return w,A.nevals()

def lobpcg_krylov(m,k,tol=1e-5):
    seed=243
    rng = np.random.default_rng(seed)
    A = util.CounterOperator(util.make_symmetric(m))
    X = rng.uniform(-1,1,size=(m,k))
    w,V = spla.lobpcg(A,X,M = lambda Y : util.block_minres_indiv(A,Y,maxiter=100)[0], tol=tol,largest=False,maxiter=1000)
    return w,A.nevals()


def ipower(m,k,tol=1e-5):
    seed=243
    rng = np.random.default_rng(seed)
    Ae = util.make_symmetric(m)
    A = util.CounterOperator(Ae)




def report(name,w,nevals):
    logger.info(f"{name}. nevals = {nevals}. min(abs(eig(A))) = {np.amin(np.abs(w))} max(abs(eig(A))) = {np.amax(np.abs(w))}")



def main():
    m=10000
    k=10
    tol=1e-5
    def run(f):
        try:
            w,nevals=f(m,k,tol=tol)
            report(f.__name__,w,nevals)
        except Exception as e:
            print(f"Example {f.__name__} failed with: {e}")

    run(lobpcg_exact)
    #run(lobpcg_krylov)
    #run(plain_lanczos)
    #run(warmup_lanczos)
    #run(inverse_lanczos)
    #run(warmup_inverse_lanczos)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
