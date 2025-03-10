import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

POSDEF = False

# Just a basic sparse symmetric matrix
# for all functions to use.
def make_symmetric(m, upper_bands = [1,16,128], seed = 42):
    rng = np.random.default_rng(seed)
    for band in upper_bands:
        assert band > 0
        assert band < m
    A = sp.diags([rng.uniform(-1,1,size=m) for _ in upper_bands],upper_bands,shape=(m,m))
    if POSDEF:
        B = A + A.T
        return B.T @ B
    return A + A.T

class CounterOperator(spla.LinearOperator):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype
        self.counter = 0
    def _matvec(self, x):
        self.counter += 1
        return self.A@x
    def _rmatvec(self, x):
        self.counter += 1
        return self.A.T@x

    def nevals(self):
        return self.counter

class SolverOperator(spla.LinearOperator):
    def __init__(self,A,tol=1e-14):
        self.A = A
        self.counterA = CounterOperator(A)
        self.shape = A.shape
        self.dtype = A.dtype
        self.tol = tol
    def _matvec(self, x):
        y,_ = spla.minres(self.counterA, x, tol=self.tol)
        return y
    def _rmatvec(self, x):
        y,_ = spla.minres(self.counterA, x, tol=self.tol)
        return y
    def nevals(self):
        return self.counterA.nevals()




# Generated by ChatGPT
class BatchedLinearOperator(spla.LinearOperator):
    def __init__(self, A, nrhs):
        """
        A: A callable that implements matmul with a (m, nrhs) matrix.
        nrhs: Number of right-hand sides.
        """
        self.A = A
        self.nrhs = nrhs
        self.shape = (A.shape[0] * nrhs, A.shape[1] * nrhs)
        self.dtype = A.dtype

    def _matvec(self, x):
        """Treat x as a (A.shape[1], nrhs) matrix, apply A, and return a vector."""
        X = x.reshape(self.A.shape[1], self.nrhs, order='F')  # Shallow reshape
        AX = self.A @ X  # Compute batched multiplication
        return AX.ravel(order='F')  # Flatten back to 1D

    def _rmatvec(self, x):
        """Same as _matvec but for the adjoint operator."""
        X = x.reshape(self.A.shape[0], self.nrhs, order='F')  # Shallow reshape
        ATX = self.A.T @ X  # Apply transpose
        return ATX.ravel(order='F')  # Flatten back to 1D

# Generated by ChatGPT. lightly modified by me.
def batched_minres(A, B, **kwargs):
    """
    Solve A @ X = B for multiple right-hand sides using minres.

    Parameters:
    - A: (m, n) matrix or LinearOperator
    - B: (m, nrhs) right-hand side matrix
    - kwargs: Additional arguments passed to scipy.sparse.linalg.minres

    Returns:
    - X: Solution matrix of shape (n, nrhs)
    - info: Convergence info from MINRES
    """
    m, n = A.shape
    nrhs = B.shape[1]

    assert B.shape[0] == m, "B must have the same number of rows as A"

    # Create batched linear operator
    A_op = BatchedLinearOperator(A, nrhs)

    # Flatten B into a single vector
    b_vec = B.ravel(order='F')

    # Solve using CG
    x_vec, info = spla.minres(A_op, b_vec, **kwargs)

    # Reshape solution back to (n, nrhs)
    X = x_vec.reshape(n, nrhs, order='F')
    return X, info


def block_minres_indiv(A, B, **kwargs):
    """
    Solve A X = B by calling MINRES independently on each right-hand side.
    All extra keyword arguments are forwarded to scipy.sparse.linalg.minres.

    Parameters
    ----------
    A : {ndarray, sparse matrix, or callable}
        A symmetric matrix or a function that applies A.
    B : ndarray
        Right-hand side(s), shape (m, nrhs).
    **kwargs : dict
        Additional keyword arguments to pass to minres (e.g. tol, maxiter, etc.).

    Returns
    -------
    X : ndarray
        Approximate solution array of shape (m, nrhs).
    infos : list
        List of convergence information for each column.
    """
    m, nrhs = B.shape
    X = np.zeros((m, nrhs))
    infos = []
    
    for i in range(nrhs):
        x, info = spla.minres(A, B[:, i], **kwargs)
        X[:, i] = x
        infos.append(info)
        
    return X, infos


# Just do approximate inverse iteration. This uses
# batched minres with fixed number of iterations as 
# a polynomial acceleration technique. 
def ipower(A,k,maxiter,inner_maxiter,rng = np.random.default_rng(42), tol=1e-3, callback = None):
    m,n=A.shape
    assert m==n
    V = rng.uniform(-1,1,size=(m,k))
    V,_ = la.qr(V,mode="economic")
    errs = np.ones(V.shape[1])
    for it in range(maxiter):
        k = np.argmax(errs < tol)
        V[:,k:],_ = block_minres_indiv(A, V[:,k:], maxiter=inner_maxiter, tol=tol)

        V,_ = la.qr(V,mode="economic")
        AV = A @ V
        errs = np.array([np.linalg.norm(V[:,j] - np.dot(V[:,j],AV[:,j])*AV[:,j]) for j in range(V.shape[1])])
        js = np.argsort(errs)
        errs = errs[js]
        V = V[:,js]
        if callback is not None:
            callback(V)
    return V
