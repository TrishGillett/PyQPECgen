# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:57:30 2015

@author: Trish
"""
import unittest
import numpy as np
import qpecgen as qpg

class test_qpecgen(unittest.TestCase):
    
    def setUp(self):
        self.Plist = [np.random.rand(n,n) for n in [10, 25, 50]]
    def test_mat_funcs(self):
        for P in self.Plist:
            PU, PD, PV = qpg.schur(P)
            # ensure that PD is a diagonal matrix
            assert np.array_equal(PD, np.diag(np.diag(PD))), repr(PD)
#            raise Exception(PU*PD*PV, np.dot(PU, np.dot(PD, PV)))
            assert np.array_equal(PU*PD*PV, np.dot(PU, np.dot(PD, PV)))
        
        for P in self.Plist:
            self.assertEqual(qpg.mindiag(P), min(P[i, i] for i in range(min(P.shape))))
            self.assertEqual(qpg.maxdiag(P), max(P[i, i] for i in range(min(P.shape))))

if __name__ == '__main__':
    unittest.main()