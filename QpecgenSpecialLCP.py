# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 04:51:40 2015

@author: Trish
"""
import QpecgenProblem
from helpers import *


class QpecgenSpecialLCP(QpecgenProblem):
    def __init__(self, param):    
        QpecgenProblem.__init__(self, param)
        m = self.param['m']
        n = self.param['n']
        # type 800 is the 'Good LCP-MPEC' and type 900 is the 'Bad LCP-MPEC'
        self.ygen = zeros(m)
        self.a = zeros(self.param['l'])
        self.N = conmat([-he.eye(n), he.zeros(m-n, n)], option='v')
        self.M = eye(m)
        self.q = zeros(m)
        
    def return_problem(self):
        problem = {'P': self.P,
                   'c': self.c,
                   'd': self.d,
                   'A': self.A,
                   'a': self.a,
                   'N': self.N,
                   'M': self.M,
                   'q': self.q}
        optsolxy = he.conmat([self.xgen, self.ygen])
        info = {'xgen': self.xgen,
                'ygen': self.ygen,
                'optsol': optsolxy,
                'optval': (0.5*(optsolxy.T)*self.P*optsolxy+he.conmat([self.c, self.d]).T*optsolxy)[0,0]}        
        return problem, info, self.param


class Qpecgen800(QpecgenSpecialLCP):
    def __init__(self, param):
        QpecgenLCP.__init__(self, param)
        self.c = -2.*he.ones(self.n)
        self.d = 4.*he.ones(self.m)
        self.xgen = he.zeros(self.n)
    
    
class Qpecgen900(QpecgenSpecialLCP):
    def __init__(self, param):
        QpecgenLCP.__init__(self, param)
        self.c = 2.*he.ones(self.n)
        self.d = -4.*he.ones(self.m)
        self.xgen = -he.ones(self.n)