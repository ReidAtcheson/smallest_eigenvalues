import scipy.linalg as la
import numpy as np
import util
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

def main():
    m=100000
    k=10
    A = util.make_symmetric(m)
    #Takes too long
    #w,K = spla.eigsh(A, sigma=0.0, which="SM",k=10,tol=1e-6)
    Ac = util.CounterOperator(A)
    prev=0
    it=0
    generations=[]
    nevals=[]
    mineigs = []
    minerrs = []
    maxerrs = []
    conv = []
    total_evals = []
    def callback(V):
        nonlocal minerrs
        nonlocal maxerrs
        nonlocal prev
        nonlocal Ac
        nonlocal it
        nonlocal nevals
        nonlocal generations
        nonlocal total_evals
        logger.info(f"generation {it}, nevals = {Ac.nevals() - prev}")
        nevals.append(Ac.nevals()-prev)
        total_evals.append(Ac.nevals())
        generations.append(it)
        it+=1
        prev = Ac.nevals()
        #A*x = L*x
        #x' * A * x = L
        errs = [np.linalg.norm(A@V[:,i] - np.dot(V[:,i],A@V[:,i])*V[:,i]) for i in range(V.shape[1])]
        minerrs.append(min(errs))
        maxerrs.append(max(errs))
        e = la.eigvalsh(V.T @ (A @ V))
        j=np.argmin(np.abs(e))
        mineigs.append(abs(e[j]))
        #i=np.argmin(np.abs(e))
        #j=np.argmin(np.abs(w))
        #conv.append(abs(e[i]-w[j])/abs(w[j]))
    util.ipower(Ac,k,maxiter=100,inner_maxiter=100,callback=callback,tol=1e-6)
    plt.plot(generations,nevals)
    plt.title("How many iterations away from invariant")
    plt.xlabel("inverse power iteration generation")
    plt.ylabel("evaluations of A*x")
    plt.savefig("invariant.svg")
    plt.close()
    plt.semilogy(total_evals,minerrs,linewidth=2,label="min(err)")
    plt.semilogy(total_evals,maxerrs,linewidth=2,label="max(err)")
    plt.title("Eigenvalue convergence")
    plt.ylabel("||Ax - (x,Ax)*x||")
    plt.xlabel("evaluations of A*x")
    plt.legend()
    plt.savefig("best_errors.svg")
    plt.close()
    plt.semilogy(total_evals,mineigs,linewidth=2,label="mineig")
    plt.ylabel("abs(min(approx eig(A)))")
    plt.xlabel("evaluations of A*x")
    plt.legend()
    plt.savefig("conv.svg")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
