# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 05:27:39 2015

@author: Trish
"""
import QpecgenProblem
from helpers import *

class Qpecgen100(QpecgenProblem):
    def __init__(self, param):    
        # QpecgenProblem has param, type, n, m, P, A, M, N
        # Qpecgen100 additionally needs a, D, E, b, E, q, c, d
        # with helper data given by: xgen, ygen, l_nonactive, ulambda, lambd, sigma, pi, eta,
        #                            indexalpha, indexgamma, index
        QpecgenProblem.__init__(self, param)
        self.info = {'xgen': rand(self.n) - rand(self.n),
                     'ygen': rand(self.m) - rand(self.m),
                     # l_nonactive: number ctrs which are not tight at and have lambda=0
                     # randomly decide how many of the non-degenerate first level ctrs should be nonactive
                     'l_nonactive': np.ceil((self.param['l']-self.param['first_deg'])*randcst()),
                     ## randomly decide how many of the non-degenerate second level ctrs should be nonactive
                     'p_nonactive': np.ceil((self.param['p']-self.param['second_deg'])*randcst())   ## Choose a random number of second level ctrs to be nonactive at (xgen, ygen)
}
        
        n = param['n']
        m = param['m']
        p = param['p']
        l = param['l']                  # l: number of first degree ctrs

        ####### FIRST LEVEL CTRS A[x;y] + a <= 0
        # Generate the RHS vector and multipliers for the first level ctrs A*[x;y]+a<=0.
        self.a = zeros(l)
        self.make_a_ulambda()
       
        ###### SECOND LEVEL CTRS Dx + Ey + b <= 0
        self.D = rand(p, n) - rand(p, n)
        self.E = rand(p, m) - rand(p, m)

        self.b = zeros(p)
        self.make_b_lambda()

        ###### STATIONARITY CONDITION FOR LOWER LEVEL PROBLEM
        # Choose q so that Nx + My + E^Tlambda + q = 0 at the solution (xgen, ygen, lambda)
        self.q = -self.N*self.info['xgen'] - self.M*self.info['ygen'] - (self.E.T)*self.info['lambda'] ## KKT conditions of the second level problem.
        
        ###### For later convenience
        self.info['F'] = npvec(N*xgen + M*ygen + q) # this must be equal to -E^T\lambda
        self.info['g'] = npvec(D*xgen + E*ygen + b)   # this is the (negative) amount of slack in the inequalities Dx + Ey + b <= 0
        
        self.make_pi_sigma_index()
        self.info['eta'] = npvec(scipy.linalg.solve(self.E, self.info['sigma']))
        
        ###### Generate coefficients of the linear part of the objective
        self.c = zeros(n)
        self.d = zeros(n)
        self.make_c_d()   
    
    def make_a_ulambda(self):
        l = self.param['l']
        first_deg = self.param['first_deg']
        l_nonactive = self.info['l_nonactive']
        xgen = self.info['xgen']
        ygen = self.info['ygen']
        
        ####### FIRST LEVEL CTRS A[x;y] + a <= 0
        # Generate the first level multipliers  ulambda  associated with A*[x;y]+a<=0.
        # Generate a so that the constraints Ax+a <= 0 are loose or tight where appropriate.
        self.a = -self.A*conmat([xgen, ygen]) - conmat([zeros(first_deg), # A + a = 0
                                rand(l_nonactive),                        # A + a = 0
                                zeros(l-first_deg-l_nonactive)])          # A + a <=0
        
        self.info['ulambda'] = conmat([zeros(first_deg),    # degenerate (ctr is tight and ulambda = 0)
                                       zeros(l_nonactive),  # not active (ulambda = 0)
                                       rand(l-first_deg-l_nonactive)]) # active (let ulambda be Uniform(0,1))
        
        
    def make_b_lambda(self):
        p = self.param['p']
        second_deg = self.param['second_deg']
        p_nonactive = self.info['p_nonactive']
        # p: number of second degree ctrs (and therefore the number of lambda vars)
        # second_deg: number of second level ctrs for which the ctr is active AND lambda=0
        # p_nonactive: number of second level ctrs which aren't active.  The corresponding lambdas must therefore be 0
        
        ## figure out what RHS vector is needed for Dx + Ey + b <= 0
        ## we intentionally build in a gap on the p_nonactive ctrs in the middle
        self.b = -D*self.xgen-E*self.ygen-conmat([zeros(second_deg), rand(p_nonactive), zeros(p-second_deg-p_nonactive)])   ## The first second_deg constraints
        
        ## we let the first second_deg cts be degenerate (ctr is tight and lambda = zero), the next p_nonactive ctrs be not active (lambda = 0), and the remaining ctrs be active (lambda Uniform(0,1))
        self.info['lambda'] = conmat([zeros(second_deg),
                                      zeros(p_nonactive),
                                      rand(p-second_deg-p_nonactive)])
    
    
    def make_pi_sigma_index(self):
        tol_deg = self.param['tol_deg']
        p = self.param['p']
        mix_deg = self.param['mix_deg']
        
        ## Calculate three index sets alpha, beta and gamma at (xgen, ygen)
        indexalpha = []
        indexgamma = []
        index = []
        for i in range(len(self.lambd)):
            if self.lambd[i] + self.g[i] < -tol_deg:
                indexalpha += [1]
            else:
                indexalpha += [0]
            if self.lambd[i] + self.g[i] > tol_deg:
                indexgamma += [-1]
            else:
                indexgamma += [0]
            index += [indexalpha[-1] + indexgamma[-1]]
        
        for i in range(len(indexalpha)):
            if indexalpha and indexgamma:
                index += [0]
            elif indexalpha:
                index += [1]
            elif indexgamma:
                index += [-1]
        ## index(i)=1  iff g(i)+lambda(i)<-tol_deg,
        ## index(i)=0   iff |g(i)+lambda(i)|<=tol_deg,
        ## index(i)=-1  iff g(i)+lambda(i)>tol_deg.
        
        ## Generate the first level multipliers   eta    pi    sigma associated
        ## with other constraints other than the first level constraints 
        ## A*[x;y]+a<=0   in the relaxed nonlinear program. In particular,
        ## eta  is associated with  N*x+M*y+q+E^T*lambda=0,
        ## pi                 with  D*x+E*y+b,
        ## sigma              with  lambda.
        k_mix = 0
        pi = zeros(p)
        sigma = zeros(p)
        for i in range(p):
            if indexalpha[i] and indexgamma[i]:
                if k_mix < mix_deg:
                    pi[i] = 0     ## The first mix_deg constraints associated with D*x+E*y+b<=0 in the set beta are degenerate.
                    sigma[i] = 0  ## The first mix_deg constraints associated with lambda>=0 in the set beta are degenerate.
                    k_mix = k_mix+1
                else:
                    pi[i] = randcst()  
                    sigma[i] = randcst()
            elif indexalpha[i]:
                pi[i] = 0    
                sigma[i] = randcst() - randcst()
            else:
                pi[i] = randcst() - randcst()
                sigma[i] = 0
        self.info.update({ 'sigma': npvec(sigma),
                           'pi': npvec(pi),
                           'indexalpha': indexalpha,
                           'indexgamma': indexgamma,
                           'index': index })
    
    def make_c_d(self):
        ###### Generate coefficients of the linear part of the objective
        xy = conmat([self.info['xgen'], self.info['ygen']])
        dxP = conmat([self.get_Px(), self.get_Pxy()], option='h')
        dyP = conmat([self.get_Pxy().T, self.get_Py], option='h')

        ##  Generate c and d such that (xgen, ygen) satisfies KKT conditions
        ##  of AVI-MPEC as well as the first level degeneracy.
        self.c = -(dxP*xy + (self.N.T)*self.info['eta'] + (self.D.T)*self.info['pi'])
        self.d = -(dyP*xy + (self.M.T)*self.info['eta'] + (self.E.T)*self.info['pi'])
        if self.param['l'] > 0:
            Ax, Ay = self.A[:, :n].T, self.A[:, n:m+n].T
            self.c += -(Ax)*self.ulambda
            self.d += -(Ay)*self.ulambda
        
        optsolxy = conmat([self.info['xgen'], self.info['ygen']])
        optsolxyl = npvec(conmat([optsolxy, lambd]))
        self.info['optsol'] = optsolxyl,
        self.info['optval'] = (0.5*(optsolxy.T)*self.P*optsolxy+conmat([self.c, self.d]).T*optsolxy)[0,0]
    
    def return_problem(self):
        problem = {'P': self.P,
                   'c': self.c,
                   'd': self.d,
                   'A': self.A,
                   'a': self.a,
                   'D': self.D,
                   'b': self.b,
                   'N': self.N,
                   'M': self.M,
                   'E': self.E,
                   'q': self.q}
        return problem, self.info, self.param