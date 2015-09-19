# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 05:27:39 2015

@author: Trish
"""
from QpecgenProblem import *
from QPCCclass import QPCCProblem
from cvxopt import matrix
from helpers import *

class Qpecgen200(QpecgenProblem):
    def __init__(self, param):    
        # QpecgenProblem has param, type, n, m, P, A, M, N
        # Qpecgen200 additionally needs a, u, b, E, q, c, d
        # with helper data given by: xgen, ygen, l_nonactive, ulambda, lambd, sigma, pi, eta,
        #                            indexalpha, indexgamma, index
        QpecgenProblem.__init__(self, param)
        
        ###### Generate xgen
        self.make_info()
        self.make_xgen()
        self.u = zeros(self.param['m']-self.info['m_inf']) # just initializing
        self.make_ygen()
        self.make_a_ulambda()
        self.make_F_pi_sigma_index()
        self.make_c_d()
        
    
    def make_info(self):
        m = self.param['m']
        second_deg = self.param['second_deg']
        
        ## Note that we only consider the following two cases of box constraints:
        ## y(i) in [0, +inf) or  [0, u] where u is a nonnegative scalar.
        ## Clearly, any other interval can be obtained by using the mapping
        ## y <--- c1+c2*y. 
        ## It is assumed that the last m_inf constraints are of the form [0, inf)
        # The remaining m_inf - m_inf_deg - inf_nonactive constraints are where F=0, y>0
        
        ## y variables which are only bounded below
        m_inf = min(m-1, ceil(m*randcst()))     # how many y vars are bounded below only
        m_inf_deg = max(second_deg-m+m_inf, ceil(min(m_inf, second_deg)*randcst()))  # how many degenerate: F=0, y=0
        inf_nonactive = ceil((m_inf-m_inf_deg)*randcst())                            # how many not active: F>0, y=0
        # The remaining m_inf - m_inf_deg - inf_nonactive constraints are where F=0, y>0
        
        ## y variables which are bounded below and above
        ## There will be m - m_inf variables with double sided bounds. each upper bound is chosen uniform in [0,10]
        m_upp_deg = ceil((second_deg-m_inf_deg)*randcst())                           # degenerate with y at upper bound: F=0, y=u
        m_low_deg = second_deg-m_inf_deg-m_upp_deg                                   # degenerate with y at lower bound: F=0, y=0
        upp_nonactive = ceil((m-m_inf-m_upp_deg-m_low_deg)*randcst())                # F not active, y at upper bound: F<0, y=u
        low_nonactive = ceil((m-m_inf-m_upp_deg-m_low_deg-upp_nonactive)*randcst())  # F not active, y at lower bound: F>0, y=0
        
        self.info = { 'm_inf': m_inf,
                      'm_inf_deg': m_inf_deg,
                      'inf_nonactive': inf_nonactive,
                      'm_upp_deg': m_upp_deg,
                      'm_low_deg': m_low_deg,
                      'upp_nonactive': upp_nonactive,
                      'low_nonactive': low_nonactive,
                      # Randomly decide how many of the non-degenerate first level ctrs should be nonactive
                      'l_nonactive': ceil((self.param['l']-self.param['first_deg'])*randcst())}
    
    def make_xgen(self):
        self.info['xgen'] = rand(n) - rand(n)
    
    def make_u(self):
        self.u = 10.*rand(self.info['m']-self.info['m_inf'])
    
    def make_ygen(self):
        ###### y variables with an upper bound
        ## There will be m - m_inf variables with double sided bounds. each upper bound is chosen uniform in [0,10]
        
        m = self.param['m']
        m_inf = self.info['m_inf']
        m_inf_deg = self.info['m_inf_deg']
        inf_nonactive = self.info['inf_nonactive']
        m_upp_deg = self.info['m_upp_deg']
        upp_nonactive = self.info['upp_nonactive']
        m_low_deg = self.info['m_low_deg']
        low_nonactive = self.info['low_nonactive']
        
        self.u = 10.*rand(m-m_inf)
        v1 = self.u[m_upp_deg+upp_nonactive+m_low_deg+low_nonactive:m-m_inf]
        v2 = npvec([randcst()*v1[i] for i in range(len(v1))])
        self.info['ygen'] = conmat([npvec(self.u[:m_upp_deg+upp_nonactive]),             # m_upp_deg (F=0, y=u) and upp_nonactive (F<0, y=u) cases
                                    zeros(m_low_deg+low_nonactive),          # m_low_deg (F=0, y=0) and low_nonactive (F>0, y=0) cases
                                    v2,                                        # for variables with double sided bounds, which do not fall in the above cases, ie. F=0, 0<y<u
                                    zeros(m_inf_deg+inf_nonactive),          # m_inf_deg (F=0, y=0) and  inf_nonactive (F>0, y=0) cases
                                    rand(m_inf-m_inf_deg-inf_nonactive)])    # m_inf-m_inf_deg-inf_nonactive (F=0, y>0)
        
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
        
        self.info['ulambda'] = conmat([zeros(first_deg+l_nonactive), rand(l-first_deg-l_nonactive)])
    
    def make_F_pi_sigma_index(self):
        N = self.N
        M = self.M
        u = self.u
        xgen = self.info['xgen']
        ygen = self.info['ygen']
        
        m = self.param['m']
        m_inf = self.info['m_inf']
        m_inf_deg = self.info['m_inf_deg']
        inf_nonactive = self.info['inf_nonactive']
        
        m_upp_deg = self.info['m_upp_deg']
        upp_nonactive = self.info['upp_nonactive']
        m_low_deg = self.info['m_low_deg']
        low_nonactive = self.info['low_nonactive']
        
        mix_deg = self.param['mix_deg']
        tol_deg = self.param['tol_deg']
        
        ## Choose q so that Nx + My + E^Tlambda + q = 0 at the solution (xgen, ygen)
        q = -N*xgen-M*ygen + conmat([zeros(m_upp_deg),                                              # degenerate upper bounds (on vars with double sided bounds)
                                   -rand(upp_nonactive),                                          # non-active upper bounds (on vars with double sided bounds)
                                   zeros(m_low_deg),                                              # degenerate lower level (on vars with double sided bounds)
                                   rand(low_nonactive),                                           # non-active lower level (on vars with double sided bounds)
                                   zeros(m-m_inf-m_upp_deg-upp_nonactive-m_low_deg-low_nonactive),# ctrs where F=0, 0<y<u (for vars with double sided bounds)
                                   zeros(m_inf_deg),                                              # degenerate lower bounds (on vars with only a lower bound)
                                   rand(inf_nonactive),                                           # nonactive lower bounds (on vars with only a lower bound)
                                   zeros(m_inf-m_inf_deg-inf_nonactive)])                         # ctrs where 0<y (for vars with only a lower bound)
        
        
        #########################################
        ##        For later convenience        ##
        #########################################
        F = N*xgen + M*ygen + q
        
        # Calculate three index sets alpha, beta and gamma at (xgen, ygen).
        # alpha denotes the index set of i at which F(i) is active, but y(i) not.
        # beta_upp and beta_low denote the index sets of i at which F(i) is
        # active, and y(i) is active at the upper and the lower end point of
        # the finite interval [0, u] respectively.
        # beta_inf denotes the index set of i at which both F(i) and y(i) are
        # active for the infinite interval [0, inf).
        # gamma_upp and gamma_low denote the index sets of i at which F(i) is
        # not active, but y(i) is active at the upper and the lower point of
        # the finite interval [0, u] respectively.
        # gamma_inf denotes the index set of i at which F(i) is not active, but y(i)
        # is active for the infinite interval [0, inf).
        index = []
        for i in range(m-m_inf):
            if abs(F[i]) <= tol_deg and ygen[i] > tol_deg and ygen[i]+tol_deg < u[i]:
                index += [1]      ## For the index set alpha.
            elif abs(F[i]) <= tol_deg and abs(ygen[i]-u[i]) <= tol_deg:
                index += [2]     ## For the index set beta_upp.
            elif abs(F[i]) <= tol_deg and abs(ygen[i]) <= tol_deg:
                index += [3]     ## For the index set beta_low.
            elif F[i] < -tol_deg and abs(ygen[i]-u[i]) <= tol_deg:
                index += [-1]     ## For the index set gamma_upp.
            elif F[i] > tol_deg and abs(ygen[i]) <= tol_deg:
                index += [-1]     ## For the index set gamma_low.
                
        for i in range(m-m_inf, m):
            if ygen[i] > F[i]+tol_deg:
                index += [1]     ## For the index set alpha.
            elif abs(ygen[i]-F[i]) <= tol_deg:
                index += [4]    ## For the index set beta_inf.
            else:
                index += [-1]    ## For the index set gamma_inf.
        
        ## Generate the first level multipliers   pi    sigma
        ## associated with other constraints other than the first level constraints 
        ## A*[x;y]+a<=0   in the relaxed nonlinear program. In particular,
        ## pi            is associated with  F(x, y)=N*x+M*y+q, and
        ## sigma                       with  y.
        mix_upp_deg = max(mix_deg-m_low_deg-m_inf_deg, ceil(m_upp_deg*randcst()))
        mix_low_deg = max(mix_deg-mix_upp_deg-m_inf_deg, ceil(m_low_deg*randcst()))
        mix_inf_deg = mix_deg-mix_upp_deg-mix_low_deg
        k_mix_inf = 0
        k_mix_upp = 0
        k_mix_low = 0
        pi = zeros(m, 1)
        sigma = zeros(m, 1)
        for i in range(m):
            if index[i] == 1:
                pi[i] = randcst()-randcst()
                sigma[i] = 0
            elif index[i] == 2:
                if k_mix_upp < mix_upp_deg:
                    pi[i] = 0    ## The first mix_upp_deg constraints associated with F(i)<=0 in the set beta_upp are degenerate. 
                    sigma[i] = 0 ## The first mix_upp_deg constraints associated with y(i)<=u(i) in the set beta_upp are degenerate.
                    k_mix_upp = k_mix_upp+1
                else:
                    pi[i] = randcst()
                    sigma[i] = randcst()
            elif index[i] == 3:
                if k_mix_low < mix_low_deg:
                    pi[i] = 0    ## The first mix_low_deg constraints associated with F(i)>=0 in the set beta_low are degenerate.
                    sigma[i] = 0 ## The first mix_low_deg constraints associated with y(i)>=0 in the set beta_low are degenerate.
                    k_mix_low = k_mix_low+1
                else:
                    pi[i] = -randcst()
                    sigma[i] = -randcst()
            elif index[i] == 4:
                if k_mix_inf < mix_inf_deg:
                    pi[i] = 0    ## The first mix_inf_deg constraints associated with F(i)>=0 in the set beta_inf are degenerate.
                    sigma[i] = 0 ## The first mix_inf_deg constraints associated with y(i)>=0 in the set beta_inf are degenerate.
                    k_mix_inf = k_mix_inf+1
                else:
                    pi[i] = -randcst()
                    sigma[i] = -randcst()
            else:
                pi[i] = 0
                sigma[i] = randcst()-randcst()
        
        self.q = q
        self.info.update({ 'F': F,
                           'mix_upp_deg': mix_upp_deg,
                           'mix_low_deg': mix_low_deg,
                           'mix_inf_deg': mix_inf_deg,
                           'pi': pi,
                           'sigma': sigma })

    def make_c_d(self):
        n = self.param['n']
        m = self.param['m']
        ###### Generate coefficients of the linear part of the objective
        ##  Generate c and d such that (xgen, ygen) satisfies KKT conditions
        ##  of AVI-MPEC as well as the first level degeneracy.
        self.c = self.get_Px()*self.info['xgen'] + self.get_Pxy()*self.info['ygen'] + self.N.T*self.info['pi']
        self.d = self.get_Pxy().T*self.info['xgen'] + self.get_Py()*self.info['ygen'] + self.M.T*self.info['pi'] + self.info['sigma']
        if self.param['l'] > 0:
            
            self.c += -((self.A[:, :n]).T)*self.info['ulambda']
            self.d += -((self.A[:, n:m+n]).T)*self.info['ulambda']
        #ELEPHANT reinstate later
#        self.info['optsol'] = self.make_fullsol()
#        self.info['optval'] = (0.5*(self.info['optsol'].T)*self.P*self.info['optsol']+conmat([self.c, self.d]).T*self.info['optsol'])[0,0]
    
    def make_fullsol(self):
        # computing the full sol at xgen, ygen
        optsolxy = conmat([self.info['xgen'], self.info['ygen']])
        v = conmat([self.N, self.M], option='h')*optsolxy + self.q
        m = self.param['m']
        mdouble = m - self.info['m_inf']
        lambdaL, lambdaU = zeros(m), zeros(mdouble)
        for i in range(mdouble):
            if v[i] >= 0:
                lambdaL[i] = 0.
                lambdaU[i] = v[i]
            else:
                assert v[i] < 0, 'If we are not in the case v[{0}] >= 0, we must have v[{0}] < 0.  Instead, v[{0}] = {1}'.format(i, v[i])
                lambdaL[i] = v[i]
                lambdaU[i] = 0.
        for i in range(mdouble, m):
            assert v[i] >= 0, 'x, y from the provided feasible solution do not satisfy Nx+Ny+q = lambda_{0} >= 0.  Violation: {1}'.format(i, v[i])
            lambdaL[i] = v[i]
        fullsol = conmat([self.info['xgen'], self.info['ygen'], lambdaL, lambdaU])
        return fullsol

    def return_problem(self):
        problem = {'P': self.P,
                   'c': self.c,
                   'd': self.d,
                   'A': self.A,
                   'a': self.a,
                   'u': self.u,
                   'N': self.N,
                   'M': self.M,
                   'q': self.q}
        return problem, self.info, self.param
    
    def make_QPCCProblem(self):
        P, info, param = self.return_problem()
        n = param['n']
        m = param['m']
        mdouble = len(P['u'])
        msingle = m - mdouble
        l = param['l']
        
        names = ['x{0}'.format(i) for i in range(n)] + ['y{0}'.format(i) for i in range(m)] + ['lL{0}'.format(i) for i in range(mdouble)] + ['lU{0}'.format(i) for i in range(mdouble)]
        Q = QPCCProblem(names)
        
        objQ = matrix([[matrix(0.5*P['P']), matrix(zeros(2*mdouble, m+n))], [matrix(zeros(m+n, 2*mdouble)), matrix(zeros(2*mdouble, 2*mdouble))]])
        objp = conmat([P['c'], P['d'], zeros(2*mdouble)])
        objr = 0
        Q.set_obj(Q=objQ, p=objp, r=objr, mode='min')
        
        G1 = conmat([P['A'], zeros(l, 2*mdouble)], option='h')
        h1 = -P['a']
        Q.add_ineqs(G1, h1)
        
        G2 = conmat([zeros(m, n), -eye(m), zeros(m, 2*mdouble)], option='h')
        h2 = zeros(m)
        Q.add_ineqs(G2, h2)
        
        G3 = conmat([zeros(mdouble, n), eye(mdouble), zeros(mdouble, m+mdouble)], option='h')
        h3 = P['u']
        Q.add_ineqs(G3, h3)
        
        G4 = conmat([zeros(mdouble, n+m), -eye(mdouble), zeros(mdouble, mdoub0le)], option='h')
        h4 = zeros(mdouble)
        Q.add_ineqs(G4, h4)
        
        G5 = conmat([P['N'][mdouble:], P['M'][mdouble:], zeros(2*mdouble, m+mdouble)], option='h')                
        h5 = -P['q'][mdouble:]
        Q.add_ineqs(G5, h5)
        
        G6 = conmat([zeros(mdouble, n+m+mdouble), -eye(mdouble)], option='h')
        h6 = zeros(mdouble)
        Q.add_ineqs(G6, h6)
        
        self.add_comps([[l+i, l+mdouble+m+i] for i in range(m+mdouble)])
        
        A = conmat([-P['N'][:mdouble], -P['M'][:mdouble], -eye(mdouble), eye(mdouble)], option='h')
        b = -P['q'][:mdouble]
        Q.add_eqs(A, b)
        
        return Q




class Qpecgen201(Qpecgen200):
    ## Type 201 is a more specific case of type 200 where x variables are constrained
    ## xl <= x < xu
    ## and y variables are constrained
    ## 0 <= y <= u
    def __init__(self, param):
        Qpecgen200.__init__(self, param)
    
    def make_info(self):
        m = self.param['m']
        second_deg = self.param['second_deg']
        
        ## Note that we only consider the following two cases of box constraints:
        ## y(i) in [0, +inf) or  [0, u] where u is a nonnegative scalar.
        ## Clearly, any other interval can be obtained by using the mapping
        ## y <--- c1+c2*y. 
        ## It is assumed that the last m_inf constraints are of the form [0, inf)
        # The remaining m_inf - m_inf_deg - inf_nonactive constraints are where F=0, y>0
        
        ## y variables which are bounded below and above
        ## There will be m - m_inf variables with double sided bounds. each upper bound is chosen uniform in [0,10]
        m_upp_deg = ceil((second_deg)*randcst())                           # degenerate with y at upper bound: F=0, y=u
        m_low_deg = second_deg-m_upp_deg                                   # degenerate with y at lower bound: F=0, y=0
        upp_nonactive = ceil((m-m_upp_deg-m_low_deg)*randcst())                # F not active, y at upper bound: F<0, y=u
        low_nonactive = ceil((m-m_upp_deg-m_low_deg-upp_nonactive)*randcst())  # F not active, y at lower bound: F>0, y=0
        # 'm_ing', 'm_inf_deg', 'inf_nonactive' are all 0 because in this case all y vars are bounded above and below
        self.info = { 'm_inf': 0,
                      'm_inf_deg': 0,
                      'inf_nonactive': 0,
                      'm_upp_deg': m_upp_deg,
                      'm_low_deg': m_low_deg,
                      'upp_nonactive': upp_nonactive,
                      'low_nonactive': low_nonactive,
                      # Randomly decide how many of the non-degenerate first level ctrs should be nonactive
                      'l_nonactive': ceil((self.param['l']-self.param['first_deg'])*randcst())}
    
    def make_xgen(self):
        xl = randint(-10, 0, self.param['n'])
        xu = randint(1, 10, self.param['n'])
        self.info['xl'] = npvec(xl)
        self.info['xu'] = npvec(xu)
        self.info['xgen'] = [xl[i] + (xu[i]-xl[i])*randcst() for i in range(self.param['n'])]
    
    def make_u(self):
        self.u = randint(0, 10, self.info['m']-self.info['m_inf'])
