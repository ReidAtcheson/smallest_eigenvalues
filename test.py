import unittest
import numpy as np
import util

class TestArnoldi(unittest.TestCase):


    def test_batched_minres(self):
        import scipy.sparse as sp
        seed=23987432
        rng=np.random.default_rng(seed)
        m=1024
        k=7
        bands = [-32,-1,0,1,32]
        A = sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
        A = A.T @ A + sp.eye(m)
        B = rng.uniform(-1,1,size=(m,k))
        X,info = util.batched_minres(A,B,tol=1e-13)
        self.assertTrue(np.linalg.norm(B - A@X) < 1e-6 )

    def test_batched_minres_indef(self):
        import scipy.sparse as sp
        seed=23987432
        rng=np.random.default_rng(seed)
        m=1024
        k=7
        bands = [-32,-1,0,1,32]
        A = util.make_symmetric(m)
        B = rng.uniform(-1,1,size=(m,k))
        X,info = util.batched_minres(A,B,tol=1e-13)
        self.assertTrue(np.linalg.norm(B - A@X) < 1e-6 )












if __name__ == '__main__':
    unittest.main()
