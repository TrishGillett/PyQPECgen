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
        self.u = None # just initializing
        self.make_ygen()
        self.make_a_ulambda()
        self.make_F_pi_sigma_index()
        self.make_c_d()
    
    def make_info(self):
        m = self.param['m']
        second_deg = self.param['second_deg']
        
        # Divide the lower level variables into groups according to constraints
        # and the role they will have at the generated solution:
        info = {}
        #### single bounded y variables (only bounded below):
        # decide total number
        info['ms'] = min(m-1, choose_num(m))
        # of these, how many will be degenerate at generated solution (F=0, y=0)
        proposed_single_deg = choose_num(min(info['ms'], second_deg))
        info['ms_deg'] = max(proposed_single_deg, second_deg-m+info['ms'])
        info['ms_nonactive'] = choose_num(info['ms']-info['ms_deg']) #F>0, y=0
        info['ms_active'] = info['ms'] - info['ms_deg'] - info['ms_nonactive'] #F=0, y>0
        
        info['md'] = m - info['ms']
        # divvy up degeneracy numbers so there are second_deg degenerate y vars
        info['md_upp_deg'] = choose_num(second_deg-info['ms_deg']) #F=0, y=u
        info['md_low_deg'] = second_deg-info['ms_deg']-info['md_upp_deg'] #F=0, y=0
        
        #### double bounded y variables (bounded below and above):
        md_nondegen = info['md'] - info['md_upp_deg'] - info['md_low_deg']
        info['md_upp_nonactive'] = choose_num(md_nondegen) #F<0, y=u
        info['md_low_nonactive'] = choose_num(md_nondegen - info['md_upp_nonactive']) #F>0, y=0
        info['md_float'] = md_unassigned - info['md_upp_nonactive'] - info['md_low_nonactive']# F=0, 0<y<u
        
        # Randomly decide how many of the non-degenerate first level ctrs should be nonactive
        self.info['l'] = self.param['l']
        self.info['l_deg'] = self.param['first_deg']
        self.info['l_nonactive'] = choose_num(self.info['l']-self.info['l_deg'])
        self.info['l_active'] = self.info['l'] - self.info['l_deg'] - self.info['l_nonactive']
    
    def make_xgen(self):
        self.info['xgen'] = rand(n) - rand(n)
    
    def make_u(self):
        self.u = 10.*rand(self.info['md'])
    
    def make_ygen(self):
        self.make_u()
        self.info['ygen'] = conmat([npvec(self.u[:self.info['md_upp_deg'] + self.info['md_upp_nonactive']]), # y=u cases
                                    zeros(self.info['md_low_deg'] + self.info['md_low_nonactive']), # y=0 cases
                                    npvec([randcst()*x for x in self.u[self.info['md'] - self.info['md_float']:]]), # 0<y<u cases
                                    zeros(self.info['ms_deg'] + self.info['ms_nonactive']), # y=0 cases
                                    rand(self.info['ms_active'])]) # y>0 cases
    
    def make_a_ulambda(self):
        xgen = self.info['xgen']
        ygen = self.info['ygen']
        
        ####### FIRST LEVEL CTRS A[x;y] + a <= 0
        # Generate the first level multipliers  ulambda  associated with A*[x;y]+a<=0.
        # Generate a so that the constraints Ax+a <= 0 are loose or tight where appropriate.
        self.a = -self.A*conmat([xgen, ygen]) - conmat([zeros(self.info['l_deg']), # A + a = 0
                                                        rand(self.info['l_nonactive']), # A + a = 0
                                                        zeros(self.info['l_active'])]) # A + a <=0
        
        self.info['ulambda'] = conmat([zeros(self.info['l_deg']),
                                       zeros(self.info['l_nonactive']),
                                       rand(self.info['l_active'])])
    
    def make_F_pi_sigma_index(self):
        N = self.N
        M = self.M
        u = self.u
        xgen = self.info['xgen']
        ygen = self.info['ygen']
        
        m = self.param['m']
        
        ## Choose q so that Nx + My + E^Tlambda + q = 0 at the solution (xgen, ygen)
        q = -N*xgen-M*ygen + conmat([zeros(self.info['md_upp_deg']),                                              # degenerate upper bounds (on vars with double sided bounds)
                                     -rand(self.info['md_upp_nonactive']),                                          # non-active upper bounds (on vars with double sided bounds)
                                     zeros(self.info['md_low_deg']),                                              # degenerate lower level (on vars with double sided bounds)
                                     rand(self.info['md_low_nonactive']),                                           # non-active lower level (on vars with double sided bounds)
                                     zeros(self.info['md_float']),# ctrs where F=0, 0<y<u (for vars with double sided bounds)
                                     zeros(self.info['ms_deg']),                                              # degenerate lower bounds (on vars with only a lower bound)
                                     rand(self.info['ms_nonactive']),                                           # nonactive lower bounds (on vars with only a lower bound)
                                     zeros(self.info['ms_active'])])                         # ctrs where 0<y (for vars with only a lower bound)
        
        
        #########################################
        ##        For later convenience        ##
        #########################################
        F = N*xgen + M*ygen + q
        
        mix_deg = self.param['mix_deg']
        tol_deg = self.param['tol_deg']
        
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
        for i in range(self.info['md']):
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
                
        for i in range(self.info['md'], m):
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
        mix_upp_deg = max(mix_deg-self.info['md_low_deg']-self.info['ms_deg'], choose_num(self.info['md_upp_deg']))
        mix_low_deg = max(mix_deg-mix_upp_deg-self.info['ms_deg'], choose_num(self.info['md_low_deg']))
        mix_inf_deg = mix_deg - mix_upp_deg - mix_low_deg
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
        lambdaL, lambdaU = zeros(m), zeros(self.info['md'])
        for i in range(self.info['md']):
            if v[i] >= 0:
                lambdaL[i] = 0.
                lambdaU[i] = v[i]
            else:
                assert v[i] < 0, 'If we are not in the case v[{0}] >= 0, we must have v[{0}] < 0.  Instead, v[{0}] = {1}'.format(i, v[i])
                lambdaL[i] = v[i]
                lambdaU[i] = 0.
        for i in range(self.info['md'], m):
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
        md = info['md']
        ms = info['ms']
        l = param['l']
        
        names = ['x{0}'.format(i) for i in range(n)] + ['y{0}'.format(i) for i in range(m)] + ['lamSL{0}'.format(i) for i in range(ms)] + ['lamDL{0}'.format(i) for i in range(md)] + ['lU{0}'.format(i) for i in range(md)]
        Q = QPCCProblem(names)
        
        objQ = matrix([[matrix(0.5*P['P']), matrix(zeros(2*md, m+n))], [matrix(zeros(m+n, 2*md)), matrix(zeros(2*md, 2*md))]])
        objp = conmat([P['c'], P['d'], zeros(2*md)])
        objr = 0
        Q.set_obj(Q=objQ, p=objp, r=objr, mode='min')
        
        G1 = conmat([P['A'], zeros(l, m+md)], option='h')
        h1 = -P['a']
        Q.add_ineqs(G1, h1)
        
        G2 = conmat([zeros(m, n), -eye(m), zeros(m, m+md)], option='h')
        h2 = zeros(m)
        Q.add_ineqs(G2, h2)
        
        G3 = conmat([zeros(md, n+ms), eye(md), zeros(md, m+md)], option='h')
        h3 = P['u']
        Q.add_ineqs(G3, h3)
        
        G4 = conmat([zeros(m+md, n+m), -eye(m+md)], option='h')
        h4 = zeros(m+md)
        Q.add_ineqs(G4, h4)
        
        Q.add_comps([[l+i, l+md+m+i] for i in range(m+md)])
        
        A1 = conmat([-P['N'][:ms], -P['M'][:ms], -eye(ms), zeros(ms, 2*md)], option='h')
        b1 = -P['q'][:ms]
        Q.add_eqs(A1, b1)
        
        A2 = conmat([-P['N'][ms:], -P['M'][ms:], zeros(md,ms), -eye(md), eye(md)], option='h')
        b2 = -P['q'][ms:]
        Q.add_eqs(A2, b2)
        print Q
        return Q




class Qpecgen201(Qpecgen200):
    ## Type 201 is a more specific case of type 200 where x variables are constrained
    ## xl <= x < xu
    ## and y variables are constrained
    ## 0 <= y <= u
    def __init__(self, param):
        Qpecgen200.__init__(self, param)
    
    def make_info(self):
        md = self.param['m']
        l = self.param['l']
        second_deg = self.param['second_deg']
        
        ## Note that we only consider the following two cases of box constraints:
        ## y(i) in [0, +inf) or  [0, u] where u is a nonnegative scalar.
        ## Clearly, any other interval can be obtained by using the mapping
        ## y <--- c1+c2*y. 
        ## It is assumed that the last m_inf constraints are of the form [0, inf)
        # The remaining m_inf - m_inf_deg - inf_nonactive constraints are where F=0, y>0
        
        ## y variables which are bounded below and above
        ## There will be m - m_inf variables with double sided bounds. each upper bound is chosen uniform in [0,10]
        md_upp_deg = choose_num(second_deg)                           # degenerate with y at upper bound: F=0, y=u
        md_low_deg = second_deg-md_upp_deg                                   # degenerate with y at lower bound: F=0, y=0
        md_upp_nonactive = choose_num(md-md_upp_deg-md_low_deg)                # F not active, y at upper bound: F<0, y=u
        md_low_nonactive = choose_num(md-md_upp_deg-md_low_deg-md_upp_nonactive)  # F not active, y at lower bound: F>0, y=0
        
        l_deg = self.param['first_deg']
        l_nonactive = choose_num(l - l_deg)
        self.info = { 'ms': 0,
                      'ms_deg': 0,
                      'ms_nonactive': 0,
                      'ms_active': 0,
                      'md': md,
                      'md_upp_deg': md_upp_deg,
                      'md_low_deg': md_low_deg,
                      'md_upp_nonactive': md_upp_nonactive,
                      'md_low_nonactive': md_low_nonactive,
                      'md_float': md - md_upp_deg - md_low_deg - md_upp_nonactive - md_low_nonactive,
                      # Randomly decide how many of the non-degenerate first level ctrs should be nonactive
                      'l': l,
                      'l_deg': l_deg,
                      'l_nonactive': l_nonactive,
                      'l_active': l - l_deg - l_nonactive}
    
    def make_xgen(self):
        xl = randint(-10, 0, self.param['n'])
        xu = randint(1, 10, self.param['n'])
        self.info['xl'] = npvec(xl)
        self.info['xu'] = npvec(xu)
        self.info['xgen'] = npvec([(xl[i] + (xu[i]-xl[i])*randcst())[0] for i in range(self.param['n'])])
    
    def make_u(self):
        self.u = randint(0, 10, self.info['md'])