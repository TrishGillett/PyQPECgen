# -*- coding: utf-8 -*-
"""
#######################################################################
## Houyuan Jiang, Daniel Ralph, copyright 1997
## Matlab code accompanied the paper: 
##   Jiang, H., Ralph D. 
##   QPECgen, a MATLAB generator for mathematical programs with 
##   Computational Optimization and Applications 13 (1999), 25–59.
##
## Python implementation coded by Patricia Gillett, 2013-2015
## See readme.txt for details about the structures of generated problems.
#######################################################################
"""

import scipy
import numpy as np

from helpers import *
#=============================================================================#









class QpecgenProblem(object):
    def __init__(self, param):
        ## CHECK CONSISTENCY OF PARAMETER DATA ##
        check = True #change to False to not check consistency of data
        if check:
            self.param = sanitize_params(param)
        else:
            self.param = param
        
        # Unpack parameters from param, which is a dictionary.
        self.type = self.param['qpec_type']
        self.n = self.param['n']
        self.m = self.param['m']
        # For now, stop controlling the seed because unless implemented
        # differently this would generate every problem with the same seed.
#        np.random.seed = param['random'] 
        
        # Upper level objective function coefficients
        self.P = self.make_obj_P()
        
        # Matrix part of upper level constraints
        self.A = self.make_ctr_A()
        
        # Lower level objective function coefficients
        self.M = self.make_obj_M()
        self.N = 1.*(rand(self.m, self.n) - rand(self.m, self.n))
        # This ends the data which is the same for all problem types
    
    def get_Px(self):
        return self.P[:self.n, :self.n]
    
    def get_Py(self):
        return self.P[self.n:self.m+self.n, self.n:self.m+self.n]
    
    def get_Pxy(self):
        #Pxy[i,j] gives coeff for x[i]y[j]
        # (but don't forget that P is symmetric so these terms
        # are encountered twice when using P)
        return self.P[:self.n, self.n:self.m+self.n]
    
    def make_obj_P(self):
        m = self.m
        n = self.n
        linobj = self.param['linobj']
        convex_f = self.param['convex_f']
        cond_P = self.param['cond_P']
        scale_P = self.param['scale_P']
        if linobj:
            return zeros(m+n, m+n)
        elif self.param['qpec_type'] in [800, 900]:
            return 2.*he.eye(n+m)
        else:
            ## Generate the quadratic terms of objective function.
            P = rand(m+n, m+n) - rand(m+n, m+n)
            P = P+P.T        ## P is symmetric.
            if convex_f:    ## Convex case.
                PU, PD, PV = schur(P)  ## Schur decomposition. PT is diagonal since P is symmetric.
                PD = tweakdiag(PD)     ## PT will be nonnegative.  subtract something to make the smallest singular value 0, then add (uniform) random numbers to each
            else:    ## Nonconvex case
                PU, PD, PV = svd(P)  ## Singular value decomposition.
            
            # for both convex and nonconvex, we adjust the condition number, reassemble the matrix, and scale if necessary.
            P = adjustcond(PU, PD, PV, cond_P)
            P = (1.*scale_P/cond_P)*P   ## Rescale P when cond_P is large.
            return 0.5*(P+P.T)
    
    def make_ctr_A(self):
        l = self.param['l']
        m = self.param['m']
        n = self.param['n']
        if l == 0:
            A = zeros(0, m+n)
        elif self.param['yinfirstlevel']:  ## yinfirstlevel = True means the upper level yinfirstlevels involve x and y
            A = rand(l, n+m) - rand(l, n+m)
        else:  ## yinfirstlevel = False means they only use x variables
            A = conmat([np.matrix(rand(l, n) - rand(l, n)), zeros(l, m)], option='h')
    #    if A.size[0] > 0:
    #        for i in range(A.size[1]):
    #            A[0,i] = abs(A[0,i])
        return A
    
    
    def make_obj_M(self):
        m = self.param['m']
        mono_M = self.param['mono_M']
        symm_M = self.param['symm_M']
        cond_M = self.param['cond_M']
        scale_M = self.param['scale_M']
        
        M = rand(m, m)-rand(m, m)
        ## Consider the symmetric property of M.
        if symm_M:
            M = M+M.T
        
        ## Consider the monotonicity property and the condition number of M.
        if mono_M and symm_M:           ## Monotone and symmetric case.
            MU, MD, MV = schur(M)
            MD = tweakdiag(MD)                  ## Generate positive diagonal matrix.
            M = adjustcond(MU, MD, MV, cond_M)
            
        elif (not mono_M) and symm_M:           ## Nonmonotone and symmetric case.
            MU, MD, MV = svd(M)
            M = adjustcond(MU, MD, MV, cond_M)
            
        elif mono_M and (not symm_M):           ## Monotone and asymmetric case.
            MU, MD, MV = schur(M)               ## note that since symm_M=False, MD is not a diagonal matrix
            for i in range(m-1):
                MD[i+1, i] = -MD[i+1, i]          ## Make real eigenvalues for MD.
            M = reconstruct(MU, MD, MV)           ## New asymmetric matrix with real eigenvalues.
            
            MU, MD, MV = schur(M)               ## Schur decomposition. MD is upper triangular since M has real eigenvalues.
            MD = tweakdiag(MD)                  ## Generate positive diagonal matrix.
            MMU, MMD, MMV = schur(np.dot(MV, MU)) ## MMD is diagonal.
            MM = adjustcond(MMU, MMD, MMV, cond_M*cond_M)
            MD = (scipy.linalg.cholesky(MM)).T  ## Use the relation of condition numbers for MD and MD.T*MD
            M = reconstruct(MU, MD, MV)         ##  Generate a matrix with the required condition number cond_M.
            
        elif (not mono_M) and (not symm_M):     ## Nonmonotone and asymmetric case.
            MU, MD, MV = svd(M)                   ## Singular value decomposition.
            MD = adjustcond(MU, MD, MV, cond_M)
        
        M = (1.*scale_M/cond_M)*M   ## Rescale M when cond_M is large.
        return M
    
    def return_problem(self):
        raise NotImplementedError
        
#
##=============================================================================#
#
#def qpecgen(param):
##    print '--------------------------------------------------------\n'
##    print '          =================================\n'
##    print '          ||      Start of qpecgen.m      ||\n'
##    print '          =================================\n'
#    
#    
#    ##################################
#    ##           AVI-MPEC           ##
#    ##################################
#    
#    
#    ##########################################
#    ######           BOX-MPEC           ######
#    ##########################################
#    elif qpec_type == 200 or qpec_type == 201:  ## In the case of BOX-MPEC.
#        second_deg = param['second_deg']
#        tol_deg = param['tol_deg']
#        mix_deg = param['mix_deg']
#        ## Note that we only consider the following two cases of box constraints:
#        ## y(i) in [0, +inf) or  [0, u] where u is a nonnegative scalar.
#        ## Clearly, any other interval can be obtained by using the mapping
#        ## y <--- c1+c2*y. 
#        ## It is assumed that the last m_inf constraints are of the form [0, inf)
#        
#        ## Type 201 is a more specific case of type 200 where x variables are constrained
#        ## xl <= x < xu
#        ## and y variables are constrained
#        ## 0 <= y <= u
#		
#        #####################################################################
#        ##        Type 201: x variables have upper and lower bounds        ##
#        #####################################################################
#        if type == 200:
#            xgen = rand(n) - rand(n)
#        elif qpec_type == 201:
#            xl = randint(-10, 0, n)
#            xu = randint(1, 10, n)
#            details['xl'] = npvec(xl)
#            details['xu'] = npvec(xu)
#            xgen = [xl[i] + (xu[i]-xl[i])*randcst() for i in range(n)]
#		
#        self.set_infs()
#        ###################################################
#        ##        y variables with an upper bound        ##
#        ###################################################
#        ## There will be m - m_inf variables with double sided bounds. each upper bound is chosen uniform in [0,10]
#        if qpec_type == 200:
#            u = 10.*rand(m-m_inf)
#        elif qpec_type == 201:
#            u = randint(0, 10, m-m_inf)
#        m_upp_deg = ceil((second_deg-m_inf_deg)*randcst())                           # m_upp_deg F=0, y=u
#        m_low_deg = second_deg-m_inf_deg-m_upp_deg                                   # m_low_deg F=0, y=0
#        upp_nonactive = ceil((m-m_inf-m_upp_deg-m_low_deg)*randcst())                # upp_nonactive F<0, y=u
#        low_nonactive = ceil((m-m_inf-m_upp_deg-m_low_deg-upp_nonactive)*randcst())  # low_nonactive F>0, y=0
#        # The remaining m - m_inf - m_upp_deg - upp_nonactive - m_low-deg - low_nonactive are where F=0, 0<y<u
#        
#        #############################################
#        ##        GENERATE OPTIMAL SOLUTION        ##
#        #############################################
#        v1 = u[m_upp_deg+upp_nonactive+m_low_deg+low_nonactive:m-m_inf]
#        v2 = npvec([randcst()*v1[i] for i in range(len(v1))])
#        ygen = conmat([npvec(u[:m_upp_deg+upp_nonactive]),             # m_upp_deg (F=0, y=u) and upp_nonactive (F<0, y=u) cases
#                     zeros(m_low_deg+low_nonactive),          # m_low_deg (F=0, y=0) and low_nonactive (F>0, y=0) cases
#                     v2,                                        # for variables with double sided bounds, which do not fall in the above cases, ie. F=0, 0<y<u
#                     zeros(m_inf_deg+inf_nonactive),          # m_inf_deg (F=0, y=0) and  inf_nonactive (F>0, y=0) cases
#                     rand(m_inf-m_inf_deg-inf_nonactive)])    # m_inf-m_inf_deg-inf_nonactive (F=0, y>0)
#        
#        
#        ####################################################
#        ##        FIRST LEVEL CTRS A[x;y] + a <= 0        ##
#        ####################################################
#        # Randomly decide how many of the non-degenerate first level ctrs should be nonactive
#        l_nonactive = ceil((l-first_deg)*randcst())
#        
#        # Let the first first_deg cts be degenerate (ctr is tight and ulambda = zero), the next l_nonactive ctrs be not active (ulambda = 0), and the remaining ctrs be active (ulambda Uniform(0,1))
#        ulambda = conmat([zeros(first_deg+l_nonactive), rand(l-first_deg-l_nonactive)])
#        
#        # Generate a so that the A + a is tight for the first first_deg+l_nonactive ctrs and has a random value <= 0 for the rest
#        a = -A*conmat([xgen, ygen])-conmat([zeros(first_deg), rand(l_nonactive), zeros(l-first_deg-l_nonactive)])
#        
#        
#        ##################################################################
#        ##        STATIONARITY CONDITION FOR LOWER LEVEL PROBLEM        ##
#        ##################################################################
#        ## Choose q so that Nx + My + E^Tlambda + q = 0 at the solution (xgen, ygen)
#        q = -N*xgen-M*ygen + conmat([zeros(m_upp_deg),                                              # degenerate upper bounds (on vars with double sided bounds)
#                                   -rand(upp_nonactive),                                          # non-active upper bounds (on vars with double sided bounds)
#                                   zeros(m_low_deg),                                              # degenerate lower level (on vars with double sided bounds)
#                                   rand(low_nonactive),                                           # non-active lower level (on vars with double sided bounds)
#                                   zeros(m-m_inf-m_upp_deg-upp_nonactive-m_low_deg-low_nonactive),# ctrs where F=0, 0<y<u (for vars with double sided bounds)
#                                   zeros(m_inf_deg),                                              # degenerate lower bounds (on vars with only a lower bound)
#                                   rand(inf_nonactive),                                           # nonactive lower bounds (on vars with only a lower bound)
#                                   zeros(m_inf-m_inf_deg-inf_nonactive)])                         # ctrs where 0<y (for vars with only a lower bound)
#        
#        
#        #########################################
#        ##        For later convenience        ##
#        #########################################
#        F = N*xgen + M*ygen + q
#        
#        # Calculate three index sets alpha, beta and gamma at (xgen, ygen).
#        # alpha denotes the index set of i at which F(i) is active, but y(i) not.
#        # beta_upp and beta_low denote the index sets of i at which F(i) is
#        # active, and y(i) is active at the upper and the lower end point of
#        # the finite interval [0, u] respectively.
#        # beta_inf denotes the index set of i at which both F(i) and y(i) are
#        # active for the infinite interval [0, inf).
#        # gamma_upp and gamma_low denote the index sets of i at which F(i) is
#        # not active, but y(i) is active at the upper and the lower point of
#        # the finite interval [0, u] respectively.
#        # gamma_inf denotes the index set of i at which F(i) is not active, but y(i)
#        # is active for the infinite interval [0, inf).
#        index = []
#        for i in range(m-m_inf):
#            if abs(F[i]) <= tol_deg and ygen[i] > tol_deg and ygen[i]+tol_deg < u[i]:
#                index += [1]      ## For the index set alpha.
#            elif abs(F[i]) <= tol_deg and abs(ygen[i]-u[i]) <= tol_deg:
#                index += [2]     ## For the index set beta_upp.
#            elif abs(F[i]) <= tol_deg and abs(ygen[i]) <= tol_deg:
#                index += [3]     ## For the index set beta_low.
#            elif F[i] < -tol_deg and abs(ygen[i]-u[i]) <= tol_deg:
#                index += [-1]     ## For the index set gamma_upp.
#            elif F[i] > tol_deg and abs(ygen[i]) <= tol_deg:
#                index += [-1]     ## For the index set gamma_low.
#                
#        for i in range(m-m_inf, m):
#            if ygen[i] > F[i]+tol_deg:
#                index += [1]     ## For the index set alpha.
#            elif abs(ygen[i]-F[i]) <= tol_deg:
#                index += [4]    ## For the index set beta_inf.
#            else:
#                index += [-1]    ## For the index set gamma_inf.
#        
#        ## Generate the first level multipliers   pi    sigma
#        ## associated with other constraints other than the first level constraints 
#        ## A*[x;y]+a<=0   in the relaxed nonlinear program. In particular,
#        ## pi            is associated with  F(x, y)=N*x+M*y+q, and
#        ## sigma                       with  y.
#        mix_upp_deg = max(mix_deg-m_low_deg-m_inf_deg, ceil(m_upp_deg*randcst()))
#        mix_low_deg = max(mix_deg-mix_upp_deg-m_inf_deg, ceil(m_low_deg*randcst()))
#        mix_inf_deg = mix_deg-mix_upp_deg-mix_low_deg
#        k_mix_inf = 0
#        k_mix_upp = 0
#        k_mix_low = 0
#        pi = zeros(m, 1)
#        sigma = zeros(m, 1)
#        for i in range(m):
#            if index[i] == 1:
#                pi[i] = randcst()-randcst()
#                sigma[i] = 0
#            elif index[i] == 2:
#                if k_mix_upp < mix_upp_deg:
#                    pi[i] = 0    ## The first mix_upp_deg constraints associated with F(i)<=0 in the set beta_upp are degenerate. 
#                    sigma[i] = 0 ## The first mix_upp_deg constraints associated with y(i)<=u(i) in the set beta_upp are degenerate.
#                    k_mix_upp = k_mix_upp+1
#                else:
#                    pi[i] = randcst()
#                    sigma[i] = randcst()
#            elif index[i] == 3:
#                if k_mix_low < mix_low_deg:
#                    pi[i] = 0    ## The first mix_low_deg constraints associated with F(i)>=0 in the set beta_low are degenerate.
#                    sigma[i] = 0 ## The first mix_low_deg constraints associated with y(i)>=0 in the set beta_low are degenerate.
#                    k_mix_low = k_mix_low+1
#                else:
#                    pi[i] = -randcst()
#                    sigma[i] = -randcst()
#            elif index[i] == 4:
#                if k_mix_inf < mix_inf_deg:
#                    pi[i] = 0    ## The first mix_inf_deg constraints associated with F(i)>=0 in the set beta_inf are degenerate.
#                    sigma[i] = 0 ## The first mix_inf_deg constraints associated with y(i)>=0 in the set beta_inf are degenerate.
#                    k_mix_inf = k_mix_inf+1
#                else:
#                    pi[i] = -randcst()
#                    sigma[i] = -randcst()
#            else:
#                pi[i] = 0
#                sigma[i] = randcst()-randcst()
#        
#        #############################################################################
#        ##        Generate coefficients of the linear part of the objective        ##
#        #############################################################################
#        ##  Generate c and d such that (xgen, ygen) satisfies KKT conditions
#        ##  of BOX-MPEC as well as the first level degeneracy.
#        
#        if l == 0:
#            c = -(Px*xgen+Pxy*ygen+(N.T)*pi)
#            d = -(Py*ygen+(Pxy.T)*xgen+(M.T)*pi+sigma)
#        else:
#            c = -(Px*xgen+Pxy*ygen+((A[:, :n]).T)*ulambda+(N.T)*pi)
#            d = -(Py*ygen+(Pxy.T)*xgen+((A[:, n:m+n]).T)*ulambda+(M.T)*pi+sigma)
#        
#        details['u'] = npvec(u)
#        details['m_inf'] = m_inf
#        details['m_inf_deg'] = m_inf_deg
#        details['inf_nonactive'] = inf_nonactive
#        details['m_upp_deg'] = m_upp_deg
#        details['m_low_deg'] = m_low_deg
#        details['upp_nonactive'] = upp_nonactive
#        details['low_nonactive'] = low_nonactive
#        details['l_nonactive'] = l_nonactive
#        details['ulambda'] = npvec(ulambda)
#        details['F'] = npvec(F)
#        details['index'] = index
#        details['mix_upp_deg'] = mix_upp_deg
#        details['mix_low_deg'] = mix_low_deg
#        details['mix_inf_deg'] = mix_inf_deg
#        details['pi'] = npvec(pi)
#        details['sigma'] = npvec(sigma)
#        details['make_with_dbl_comps'] = param['make_with_dbl_comps']
#    
#    
#    ##################################
#    ##           LCP-MPEC           ##
#    ##################################
#    elif qpec_type == 300:
#        #############################################
#        ##        GENERATE OPTIMAL SOLUTION        ##
#        #############################################
#        xgen = rand(n, 1)-rand(n, 1)
#        m_nonactive = ceil((m-second_deg)*randcst())   # The number of indices where the second level objective function is not active at (xgen, ygen).
#        ygen = conmat([zeros(second_deg+m_nonactive, 1), rand(m-second_deg-m_nonactive, 1)])  # The first second_deg+m_nonactive elements of ygen are active.
#        
#        ##  Generate the vector in the second level objective function.
#        q = -N*xgen-M*ygen+conmat([zeros(second_deg, 1), rand(m_nonactive, 1), zeros(m-second_deg-m_nonactive, 1)])
#        #The first second_deg indices are degenerate at (xgen, ygen).
#        F = N*xgen + M*ygen + q       ## The introduction of F is for later convenience.
#        
#        ####################################################
#        ##        FIRST LEVEL CTRS A[x;y] + a <= 0        ##
#        ####################################################
#        ##  Generate the first level multipliers  ulambda  associated with   A*[x;y]+a<=0.
#        l_nonactive = ceil((l-first_deg)*randcst()) ## The number of nonactive
#        #     first level constraints at (xgen, ygen).
#        ulambda = conmat([zeros(first_deg+l_nonactive, 1), rand(l-first_deg-l_nonactive, 1)])
#        
#        ##  Generate the vector in the first level constraints set.
#        a = -A*conmat([xgen, ygen])-conmat([zeros(first_deg, 1), rand(l_nonactive, 1), zeros(l-first_deg-l_nonactive, 1)])   ## The first first_deg constraints
#        ##       are degenerate, the next l_nonative constraints are not active,
#        ##       and the last l-first_deg-l_nonactive constraints are active
#        ##       but nondegenerate at (xgen, ygen).
#        
#        #################################
#        ##        For later use        ##
#        #################################
#        ##  Calculate three index set alpha, beta and gamma at (xgen, ygen).
#        indexalpha = []
#        indexgamma = []
#        index = []
#        for i in len(F):
#            if F[i]+tol_deg < ygen[i]:
#                indexalpha += [1]
#            else:
#                indexalpha += [0]
#            if F[i] > ygen[i]+tol_deg:
#                indexgamma += [-1]
#            else:
#                indexgamma += [0]
#            index += [indexalpha[-1] + indexgamma[-1]]
#        ## index(i)=1 iff F(i)+tol_deg<ygen(i),
#        ##  index(i)=0 iff |F(i)-ygen(i)|<=tol_deg,
#        ##  index(i)=-1 iff F(i)>ygen(i)+tol_deg.
#        ##
#        ## Generate the first level multipliers associated with other constraints
#        ## other than the first level constraints   A*[x;y]+a<=0   in the relaxed
#        ## nonlinear program. In particular,   pi  and  sigma  are associated with  
#        ## F(x, y)=N*x+M*y+q   and    y   in the relaxed nonlinear program.
#        k_mix = 0
#        for i in range(m):
#            if index[i] == -1:
#                pi[i] = 0
#                sigma[i] = randcst()-randcst()
#            elif index[i] == 0:
#                if k_mix < mix_deg:
#                    pi[i] = 0  ## The first mix_deg constraints associated
#                    #with F(x, y)>=0 in the set beta are degenerate.
#                    sigma[i] = 0  ## The first mix_deg constraints
#                    # associated with y>=0 in the set beta are degenerate.
#                    k_mix = k_mix+1
#                else:
#                    pi[i] = randcst()
#                    sigma[i] = randcst()
#            else:
#                pi[i] = randcst()-randcst()
#                sigma[i] = 0
#        
#        #############################################################################
#        ##        Generate coefficients of the linear part of the objective        ##
#        #############################################################################
#        ##  Generate c and d such that (xgen, ygen) satisfies KKT conditions
#        ##  of LCP-MPEC as well as the first level degeneracy.
#        if l == 0:
#            c = (N.T)*pi-Px*xgen-Pxy*ygen
#            d = (M.T)*pi+sigma-Py*ygen-(Pxy.T)*xgen
#        else:
#            c = (N.T)*pi-Px*xgen-Pxy*ygen-((A[:, :n]).T)*ulambda
#            d = (M.T)*pi+sigma-Py*ygen-(Pxy.T)*xgen-((A[:, n:n+m]).T)*ulambda
#        ##  The end of LCP-MPEC.
#        
#        details['m_nonactive'] = m_nonactive
#        details['l_nonactive'] = l_nonactive
#        details['F'] = npvec(F)
#        details['ulambda'] = npvec(ulambda)
#        details['indexalpha'] = indexalpha
#        details['indexgamma'] = indexgamma
#        details['index'] = index
#        details['pi'] = npvec(pi)
#        details['sigma'] = npvec(sigma)
#        ##################################
#        ##    Good and bad LCP-MPEC     ##
#        ##################################
#    
#    elif qpec_type == 800 or qpec_type == 900:
#        # type 800 is the 'Good LCP-MPEC' and type 900 is the 'Bad LCP-MPEC'
#        P = 2.*eye(n+m)
#        c = -2.*ones(n)
#        d = 4.*ones(m)
#        xgen = zeros(n)
#        ygen = zeros(m)
#        if qpec_type == 900:
#            c, d = -c, -d
#            xgen = -ones(n)
#        a = zeros(l)
#        N = conmat([-eye(n), zeros(m-n, n)], option='v')
#        M = eye(m)
#        q = zeros(m)
#        problem = {'P': P,
#                   'c': c,
#                   'd': d,
#                   'A': A,
#                   'a': a,
#                   'N': N,
#                   'M': M,
#                   'q': q}
#        optsolxy = conmat([xgen, ygen])
#        info = {'xgen': xgen,
#                'ygen': ygen,
#                'optsol': conmat([xgen, ygen]),
#                'optval': (0.5*(optsolxy.T)*P*optsolxy+conmat([c, d]).T*optsolxy)[0,0]}
#    
#    ##################################
#    ##       Output                 ##
#    ##################################
#    
#    displaydata = False
#    if displaydata:
#        print '\n\n'
#        for key in param:
#            print '{0} = {1}'.format(key, param[key])
#    
#    # To avoid rounding errors during computation.
#    
#    optsolxy = conmat([xgen, ygen])
#    optval = 0.5*(optsolxy.T)*P*optsolxy+conmat([c, d]).T*optsolxy  
#    details['optval'] = optval[0, 0]
#    if qpec_type == 100:
#        details['optsol'] = npvec(conmat([xgen, ygen, lambd]))
#    else:
#        details['optsol'] = npvec(optsolxy)
#    
#    
#    print "Problem generation complete, passing to makeqpec."
#    return problem, info, param
#
#
#
#
##=============================================================================#
#
#
#def makeqpec(D):
#    """
#    Given a problem's parameters and type, constructs the corresponding
#    Problem.
#    """
##    typ = D['typ']
##    Pobj = D['P']
##    A = D['A']
##    N = D['N']
##    M = D['M']
##    c = D['c']
##    d = D['d']
##    a = D['a']
##    q = D['q']
##    optsol = D['optsol']
##    optval = D['optval']
#    
#    l = len(D['a'])  ## number of first level inequality ctrs
#    n = len(D['c'])  ## number of x vars
#    m = len(D['d'])  ## number of y vars
#    optsol = D['optsol']
#
#    P = {}
#    if D['typ'] == 100:
#        p = len(D['b'])  ## number of g ctrs, number of lambda vars, number of equalities        
#        P['names'] = ['x{0}'.format(i) for i in range(n)] + ['y{0}'.format(i) for i in range(m)] + ['L{0}'.format(i) for i in range(p)]
#        Q1 = conmat([0.5*D['P'], zeros(n+m, p)], option='h')
#        Q2 = conmat([zeros(p, n+m+p)], option='h')
#        P['Q'] = conmat([Q1, Q2])
#        P['p'] = conmat([D['c'], D['d'], zeros(p, 1)])
#    elif D['make_with_dbl_comps'] and (D['typ'] == 200 or D['typ'] == 201):
#        mdouble = len(D['u'])
#        P['names'] = ['x{0}'.format(i) for i in range(n)] + ['y{0}'.format(i) for i in range(m)] + ['lL{0}'.format(i) for i in range(mdouble)] + ['lU{0}'.format(i) for i in range(mdouble)]
#        P['Q'] = matrix([[0.5*D['P'], zeros(2*mdouble, m+n)], [zeros(m+n, 2*mdouble), zeros(mdouble, mdouble)]])
#        P['p'] = conmat([D['c'], D['d'], zeros(2*mdouble)])
#    else:
#        P['names'] = ['x{0}'.format(i) for i in range(n)] + ['y{0}'.format(i) for i in range(m)]
#        P['Q'] = 0.5*D['P']
#        P['p'] = conmat([D['c'], D['d']])
#    P['r'] = 0
#    
#    if D['typ'] == 100:
#        ## keys in D: typ, P, A, N, M, c, d, a, q, optsol, optval, D, E, b
#        p = len(D['b'])  ## number of g ctrs, number of lambda vars, number of equalities
#        
#        # in order of variables: x variables (n), y variables (m), lambda variables (p)
#        P['A'] = conmat([D['N'], D['M'], D['E'].T], option='h')
#        P['b'] = -D['q']
#        G1 = conmat([D['A'], zeros(l, p)], option='h')
#        G2 = conmat([D['D'], D['E'], zeros(p, p)], option='h')
#        G3 = conmat([zeros(p, n+m), -eye(p)], option='h')
#        P['G'] = conmat([G1, G2, G3])
#        P['h'] = conmat([-D['a'], -D['b'], zeros(p, 1)])
#        P['comps'] = [[l+i, l+p+i] for i in range(p)]
#        P['varsused'] = [1]*(n+m) + [0]*p
#    
#    elif D['typ'] >= 200 and D['typ'] <= 202:
#        # in order of variables: x variables (n), double sided y variables (mdouble), single sided y vars (msingle)
#        
#        if not D['make_with_dbl_comps']:
#        # type 200: D's keys: typ, P, A, N, M, c, d, a, q, optsol, optval, u
#            mdouble = len(D['u'])
#            msingle = m - mdouble
#            G1 = conmat([D['A'], zeros(l, 2*mdouble)], option='h')
#            G2 = conmat([zeros(m, n), -eye(m), zeros(m, 2*mdouble)], option='h')
#            G3 = conmat([zeros(mdouble, n), eye(mdouble), zeros(mdouble, m+mdouble)], option='h')
#            G4 = conmat([zeros(mdouble, n+m), -eye(mdouble), zeros(mdouble, msingle)], option='h')
#            G6 = conmat([zeros(mdouble, n+m+mdouble), -eye(mdouble)], option='h')
#            if D['typ'] == 200:
#                G5 = conmat([D['N'][mdouble:], D['M'][mdouble:], zeros(2*mdouble, msingle)], option='h')                
#                P['G'] = conmat([G1, G2, G3, G4. G5, G6])
#                P['h'] = conmat([-D['a'], zeros(m), D['u'], zeros(mdouble), -D['q'][mdouble:], zeros(mdouble)])
#            else:
#                P['G'] = conmat([G1, G2, G3, G4. G6])
#                P['h'] = conmat([-D['a'], zeros(m), D['u'], zeros(mdouble), zeros(mdouble)])
#            P['comps'] = [[l+i, l+mdouble+m+i] for i in range(m+mdouble)]
#            P['A'] = conmat([-D['N'][:mdouble], -D['M'][:mdouble], -eye(mdouble), eye(mdouble)], option='h')
#            P['b'] = -D['q'][:mdouble]
#            v = conmat([D['N'], D['M']], option='h')*optsol + D['q']
#            lambdaL, lambdaU = zeros(m), zeros(mdouble)
#            for i in range(mdouble):
#                if v[i] >= 0:
#                    lambdaL[i] = 0.
#                    lambdaU[i] = v[i]
#                else:
#                    assert v[i] < 0, 'If we are not in the case v[{0}] >= 0, we must have v[{0}] < 0.  Instead, v[{0}] = {1}'.format(i, v[i])
#                    lambdaL[i] = v[i]
#                    lambdaU[i] = 0.
#            for i in range(mdouble, m):
#                assert v[i] >= 0, 'x, y from the provided feasible solution do not satisfy Nx+Ny+q = lambda_{0} >= 0.  Violation: {1}'.format(i, v[i])
#                lambdaL[i] = v[i]
#            optsol = conmat([optsol, lambdaL, lambdaU])
#            
#            
#        if D['typ'] == 200:
#        # type 200: D's keys: typ, P, A, N, M, c, d, a, q, optsol, optval, u
#            mdouble = len(D['u'])
#            msingle = m - mdouble
#            G1 = D['A']
#            G2 = conmat([zeros(mdouble, n), -eye(mdouble), zeros(mdouble, msingle)], option='h')
#            G3 = conmat([zeros(mdouble, n), eye(mdouble), zeros(mdouble, msingle)], option='h')
#            G4 = conmat([zeros(msingle, n), zeros(msingle, mdouble), -eye(msingle)], option='h')
#            G5 = conmat([D['N'][mdouble:, :], D['M'][mdouble:, :]], option='h')
#            P['G'] = conmat([G1, G2, G3, G4, G5])
#            P['h'] = conmat([-D['a'], zeros(mdouble), D['u'], zeros(msingle), -D['q'][mdouble:]])
#            P['comps'] = [[l+2*mdouble + i, l+mdouble+m+i] for i in range(msingle)]
#            P['expF'] = conmat([-D['N'][:mdouble, :], -D['M'][:mdouble, :]], option='h')
#            P['exph'] = D['q'][:mdouble]
#            # doublecomps: [i,j,k] means expression exphi - expFiTx forms a nonpositive product with hj-Gjx and a nonnegative product with hk-Gkx
#            P['doublecomptuples'] = [[i, l+mdouble+i, l+i] for i in range(mdouble)]
#        
#        elif D['typ'] == 201:
#        # type 201: D's keys: typ, P, A, N, M, c, d, a, q, optsol, optval, u, xl, xu
#            G1 = D['A']
#            G2 = conmat([-eye(n), zeros(n, m)], option='h')
#            G3 = conmat([eye(n), zeros(n, m)], option='h')
#            G4 = conmat([zeros(m, n), -eye(m)], option='h')
#            G5 = conmat([zeros(m, n), eye(m)], option='h')
#            P['G'] = conmat([G1, G2, G3, G4, G5])
#            P['h'] = conmat([-D['a'], -D['xl'], D['xu'], zeros(m), D['u']])
#            P['expF'] = conmat([-D['N'], -D['M']], option='h')
#            P['exph'] = D['q']
#            # doublecomps: [i,j,k] means expression exphi - expFiTx forms a nonpositive product with hj-Gjx and a nonnegative product with hk-Gkx
#            P['doublecomptuples'] = [[i, l+2*n+i, l+2*n+m+i] for i in range(m)]
#    
#    elif D['typ'] == 300:
#        print "makeqpec does not currently convert type 300 problems to QPCCs"
###           s.t.  A*[x;y] + a <= 0
###                 0 <= y
###                 N*x + M*y + q >= 0
###                 (N*x + M*y + q)^Ty = 0
#    
#    elif D['typ'] == 800 or D['typ'] == 900:
#        # type 800/900: D's keys: typ, P, A, N, M, c, d, a, q, optsol, optval
#        P['A'] = conmat([zeros(m-n, 2*n), eye(m-n)])
#        P['b'] = zeros(m-n)
#        G1 = conmat([-D['N'][:n, :], -D['M'][:n, :]], option='h')
#        G2 = conmat([zeros(n, n), -eye(n), zeros(m-n, n)], option='h')
#        P['G'] = conmat([G1, G2])
#        P['h'] = conmat([D['q'], zeros(n)])
#        P['comps'] = [[i, n+i] for i in range(n)]
#    
#    P['trueoptsol'] = optsol
#    P['trueoptval'] = D['optval']
##    if typ == 201:
##        B0 = 0
##        for i in range(n):
##            B0 += max(xl[i]**2, xu[i]**2)
##        for i in range(m):
##            B0 += u[i]**2
##        B0 = np.sqrt(B0)[0]
##        B0 = np.sqrt(sum([max(xl[i]*xl[i], xu[i]*xu[i]) for i in range(n)]) + sum([u[i]*u[i] for i in range(m)]))[0,0]
##        P['Bs'] = [1.10*B0]
##        P['Bs'] = [1.10*B0, 2.*B0, 10*B0]
##        P['Bs'] = []
##    else:
##        B0 = np.sqrt(sum(x*x for x in optsol[:n+m])[0, 0])
##        P['Bs'] = [1.10*B0, 1.20*B0, 2.*B0, 10*B0, 100*B0]
##        P['Bs'] = []
##    print "An appropriate B list for this problem would be {0}.".format(P['Bs'])
#    P['Btypes'] = ['varsused']
#    P['Bs'] = []
#    P['details'] = D
#    if 'varsused' not in P:
#        P['varsused'] = [1]*(n+m)
##    print P
#    return P
#
#
#
#
##=============================================================================#
#
#def Problem_from_P(P, pname, timestamp):
#    """
#    Takes a dictionary P, problem name, and timestamp, and constructs a problem
#    dictionary.
#    """
#    trueoptval = P['trueoptval']
#    trueoptsol = matrix(P['trueoptsol']+1-1)
#    Q = matrix(P['Q']+1-1)
#    p = matrix(P['p']+1-1)
#    r = P['r']
#    names = P['names']
#    varsused = P['varsused']
#    
#    s = support.obj_as_str(Q, p, r, names)
#    
#    Problem = {'pname': pname,
#                  'mode': 'min',
#                  'names': names,
#                  'G': matrix(P['G']+1-1),
#                  'h': matrix(P['h']+1-1),
#                  'Bs': P['Bs'],
#                  'N': 1,
#                  'varsused': varsused,
#                  'clas': 'qpecgen',
#                  
#                  'details': P['details'],
#                  'timestamp': timestamp,
#                  'results': [{'solver': 'gen', 'type': 'gen', 'timestamp': '00000000000000', 'status': 'optimal', 'note': 'feasible from QPECgen', 'B': -1,
#                               'value': trueoptval, 'suggestedsol': trueoptsol, 'ID': 0, 'solvetime': 0}],
#                  'nextID': 1,
#                  'Obj': {'Q': Q,
#                          'p': p,
#                          'r': r,
#                          'objstr': s,
#                          'mode': 'min'}}
#    if 'param' in P['details']:
#        Problem['param'] = P['details']['param']
#        Problem['type'] = P['details']['param']['qpec_type']
#    
#    if 'Btypes' in P:
#        Problem['Btypes'] = P['Btypes']
#    if 'A' in P:
#        Problem['A'] = matrix(P['A']+1-1)
#        Problem['b'] = matrix(P['b']+1-1)
#    else:
#        Problem['A'] = matrix(0., (0, len(names)))
#        Problem['b'] = matrix(0., (0, 1))
#    if 'comps' in P:
#        Problem['comps'] = P['comps']
#    else:
#        Problem['comps'] = []
#    if 'doublecomptuples' in P:
#        Problem['doublecomptuples'] = P['doublecomptuples']
#        Problem['expF'] = matrix(P['expF']+1-1)
#        Problem['exph'] = matrix(P['exph']+1-1)
#    else:
#        Problem['doublecomptuples'] = []
#        Problem['expF'] = matrix(0., (0, len(names)))
#        Problem['exph'] = matrix(0., (0, 1))    
##    print "Optimal value of problem {0} is {1} at\n{2}.".format(pname, P['trueoptval'], P['trueoptsol'])
#    return Problem
#
#
#
#
#def qpec_generate(seriesname, param, N):
#    """
#    Generates N problems with the parameters given in the dictionary param,
#    all named seriesname_{i}.
#    """
#    ProblemSeries = []
#    timestamp = support.get_timestamp()
#    
#    for i in range(N):
#        P = qpecgen(param)
#        ProblemSeries += [Problem_from_P(P, seriesname + str(i), timestamp)]
#    return ProblemSeries
#
#def qpec_generate2(param, tuplist, Neach, start=0):
#    """
#    Generates N problems with the parameters given in the dictionary param,
#    all named seriesname_{i}.
#    """
#    ProblemSeries = []
#    timestamp = support.get_timestamp()
#    
#    for tup in tuplist:
#        param['qpec_type'] = tup[0]
#        param['n'] = tup[1]
#        param['m'] = tup[2]
#        param['l'] = tup[2]
#        param['p'] = tup[2]
#        for k in range(start, start+Neach):
#            P = qpecgen(param)
#            ProblemSeries += [Problem_from_P(P, 'qpecgen_{0}_{1}_{2}_no{3}'.format(tup[0], tup[1], tup[2], k), timestamp)]
#    return ProblemSeries
#
#tuplist = [[201, 5, 2], [201, 10, 5], [201, 20, 10], [201, 30, 15], [201, 50, 25], [201, 60, 30]]
#
#
#
#param_inst = {'qpec_type': 201,            # 100, 200, 300, 800, 900.
#              'n': 5,                       # Dimension of the variable x.
#              'm': 3,                      # Dimension of the variable y.
#              'l': 2,                       # Number of the first level constraints.
#              'cond_P': 100,                  # Condition number of the Hessian P.
#              'convex_f']: True,            # True or False. Convexity of the objective function.
#              'linobj': False,
#              'symm_M': True,              # True or False. Symmetry of the matrix M.
#              'mono_M': True,              # True or False. Monotonicity of the matrix M.
#              'cond_M': 100,                  # Condition number of the matrix M.
#              'second_deg': 0,               # Number of the second level degeneracy.
#              'first_deg': 1,                # Number of the first level degeneracy.
#              'mix_deg': 0,                  # Number of mixed degeneracy.
#              'tol_deg': 10**(-6),           # Small positive tolerance for measuring degeneracy.
#              'yinfirstlevel': True,         # Whether or not the lower level variables y are involved in the upper level constraints
#              'random': 0,                   # Indicates the random 'seed'.
#              'make_with_dbl_comps': False }
#param_inst.update({ 'p':param_inst['m'],               # Number of the second level constraints for AVI-MPEC.
#                    'scale_P': param_inst['cond_P'],    # Scaling constant for the Hessian P.
#                    'scale_M': param_inst['cond_M'] }    # Scaling constant for the matrix M.
#
#
#
#
############################################################################
#### functions for importing from qpecgen generated .mat files
############################################################################
#
#
#def dotmats_to_ProblemSeries(filenamebase, typ, N, N0=1):
#    """
#    Reads in problems generated from the matlab qpecgen code, creating a
#    ProblemSeries.
#    """
#    timestamp = support.get_timestamp()
#    ProblemSeries = []
#    for i in range(N0, N+1):
#        filename = 'matlabQPECgen/{0}_{1}_{2}.mat'.format(filenamebase, typ, i)
##        print filename
#        P = qpecgen_dotmat_to_qpcc(filename, typ)
##        print P
#        ProblemSeries += [Problem_from_P(P, '{0}_{1}_{2}'.format(filenamebase, typ, i), timestamp)]
#    return ProblemSeries
#
#
#
#
#
#def qpecgen_dotmat_to_qpcc(filename, typ):
#    """
#    Returns a Problem corresponding to the problem defined by filename's
#    .mat.
#    """
#    Di = dotmat_to_dict(filename)
#    Di['optsol'] = conmat([Di['xgen'], Di['ygen']])
#    Di['optval'] = 0.5*(Di['optsol'].T)*Di['P']*Di['optsol'] + conmat([Di['c'], Di['d']]).T*Di['optsol']
#    Di['typ'] = typ
#    return makeqpec(Di)
#
#        
#        
#
#def dotmat_to_dict(filename):
#    """
#    Given a matlab .mat file, creates a dictionary containing the data with
#    variable names for keys.
#    """
#    D = io.loadmat(filename)
#    for key in D:
#        if key[:2] != '__':
#            D[key] = np.matrix(D[key])
#    return D
#
#
#
#
#
#
#
#
#
#
#
#
#
#def get_solve_timestamp_from_AMPL_folder(amplfolder):
#    kleft = amplfolder.rfind('-')
#    timestamp = amplfolder[kleft+1:]
#    if timestamp[-1]=='\\' or timestamp[-1]=='/':
#        timestamp = timestamp[:-1]
#    if len(timestamp) != 14:
#        raise Exception("timestamp {0} is not 14 characters long like we expect, for example '20140812175247'.".format(timestamp))
#    return timestamp
#
#
#
#
#def resume_from_regular_solves(picklefile, amplfolder):
#    timestamp = get_solve_timestamp_from_AMPL_folder(amplfolder)
#    solvers=['PKNITRO', 'BARONMIP']
#    ProblemSeries = pickling.loadProblemSeries(picklefile)
#    
#    pybatchfile = amplfolder + "pybatch.pickle"
#    pybatch = modelwriter.load_batch(pybatchfile)
#    resultfile = amplfolder + "AMPLresults.csv"
#    resultfileB = amplfolder + "AMPLresultsB.csv"
#    
#    pybatch.solve_all(resultfile)
#    ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfile, timestamp)
#    
#    ProblemSeries, resultfileB = testing.bounded_BARON_solves(solvers, ProblemSeries, pybatch, resultfileB)
#    ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfileB, timestamp)
#    testing.merge_csv([resultfile, resultfileB])
#
#
#
#
#def resume_from_bounded_BARON(picklefile, amplfolder):
#    timestamp = get_solve_timestamp_from_AMPL_folder(amplfolder)
#    ProblemSeries = pickling.loadProblemSeries(picklefile)
#    
#    pybatchfile = amplfolder + "pybatch.pickle"
#    pybatch = modelwriter.load_batch(pybatchfile)
#    pybatch.basefolder = "C:/Users/Trish/Documents/Research/AMPL/20140815013415-ran-at-20140815024744/"
#    resultfile = amplfolder + "AMPLresults.csv"
#    resultfileB = amplfolder + "AMPLresultsB.csv"
#    
#    pybatch.solve_all(resultfileB)
#    ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfileB, timestamp)
#    testing.merge_csv([resultfile, resultfileB])
#
#
#
#
############################################################################
#### generating QPECgen problems
############################################################################
#
##PS = qpec_generate('temp', param_inst, 1)
##filename = pickling.saveProblemSeries(PS, nameoverride='qpecgen_type200_C')
#
##print "saved to {0}".format(filename)
##for P in ProblemSeries:
##    print P
###    print P['AMPLModel']
##    print "==========================================================="
#
##nmlist = [[5, 2], [10, 5], [20, 10], [30, 15], [40, 20]]
##
#
#
#
#
#
#
#
#
#
##picklefile = "ProblemSeries-20140813202421-results-for-report-10-5-5-o0.0001-a0.0001-20140814064555.pickle"
##amplfolder = "C:/Users/Trish/Documents/Research/AMPL/20140813202421-ran-at-20140814064555/"
##resume_from_regular_solves(picklefile, amplfolder)
##
##
##solvers = ['PKNITRO', 'BARONMIP']
##opts = [[0, 0], [0, 0.0001]]
##for tup in opts:
##    print "WAKAWAKA"
##    optcr = tup[0]
##    alpha = tup[1]
##    testing.runallseries("ProblemSeries-20140813202421-results-for-report-10-5-5.pickle", balls='no', solvename='o{0}-a{1}'.format(optcr, alpha), solvers=solvers, BARONbounds=True, optcr=optcr, alpha=alpha)
##
#
#
#
##### run this after verifying file names are done right
##timestamp = '20140812093123'
##folder = "C:/Users/Trish/Documents/Research/AMPL/20140812025245-ran-at-2014081209312she-hulk ceremony part 23/"
##filename = "ProblemSeries-20140721204625-10problems9vars.pickle"
#
##pybatchfile = folder + "pybatch.pickle"
##pybatch = modelwriter.load_batch(pybatchfile)
##resultfile = folder + "AMPLresults.csv"
##resultfileB = folder + "AMPLresultsB.csv"
##pybatch.solve_all(resultfileB)
##ProblemSeries = pickling.loadProblemSeries(filename)
##ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfileB, timestamp)
##testing.merge_csv([resultfile, resultfileB])
#
#
#
#
#
#
##pybatch = modelwriter.load_batch('C:/Users/Trish/Documents/Research/AMPL/20140721204625-ran-at-20140724011009/pybatch.pickle')
##resultfile = "C:/Users/Trish/Documents/Research/AMPL/20140721204625-ran-at-20140724011009/AMPLresultsB.csv"
##timestamp = support.get_timestamp()
##ProblemSeries = pickling.loadProblemSeries(filename)
##ProblemSeries, resultfile = testing.bounded_BARON_solves(solvers, ProblemSeries, pybatch, resultfile)
##ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfile, timestamp)
##testing.merge_csv(["C:/Users/Trish/Documents/Research/AMPL/20140721204625-ran-at-20140724011009/AMPLresults.csv", resultfile])
#
#
##print_averages(PS)
##plot_M_randvsWS_gapdist(loadProblemSeries("ProblemSeries-20140423224259-qpecgen_type201_NC-ballsno-20140423224305.pickle"))
#
#
##mergeMacMPEC()
### PROBLEM: runallseries expects 'param' for each Problem
##PS, filename = runallseries('MacMPECplus.pickle', warmstart=True, balls=balls, solvename='balls' + balls, closest_to_warmstart=True)
##load_and_view('ProblemSeries-20140423224259-qpecgen_type201_NC-ballsyes-20140423224305.pickle', withsols=False, convex=False)
#
#
##print_ProblemSeries_results(PSeries, withsols=False)
#
#
#
############################################################################
#### importing QPECgen problems generated as .mat files from matlab
############################################################################
#
##to import a single .mat:
##P = qpecgen_dotmat_to_qpcc('matlabQPECgen/qpecgen_data_c_100_1.mat', 100)
#
#### ELEPHANT: before running this one, need to update makeqpec to handle type 200
##ProblemSeries = dotmats_to_ProblemSeries('qpecgen_data_c', 200, 1)
##filename = saveProblemSeries(ProblemSeries, nameoverride='qpecgen_matlab')
##print "saved to {0}".format(filename)
##newProblemSeries, newfilename =  runallseries(filename, warmstart=True)
#
##ProblemSeries = dotmats_to_ProblemSeries('qpecgen_data_nc', 100, 20)
##filename = pickling.saveProblemSeries(ProblemSeries, nameoverride='matlab nonconvex unbounded demo')
##print "saved to {0}".format(filename)
#
#
#
#
############################################################################
#### running a pickled problem series
############################################################################
#
### Use these ones to import QPECGEN problems from .mat files and then solve them.
##C1to10 = "aardvark-C-5balls-1-to-10-ProblemSeries-20140110125455.pickle"
##C11to20 = "aardvark-C-5balls-11-to-20-ProblemSeries-20140110125456.pickle"
##NC1to10 = "aardvark-NC-5balls-1-to-10-ProblemSeries-20140110125457.pickle"
##NC11to20 = "aardvark-NC-5balls-11-to-20-ProblemSeries-20140110125457.pickle"
#
##newProblemSeries, newfilename =  runallseries(C1to10, warmstart=True)
##newProblemSeries, newfilename =  runallseries(C11to20, warmstart=True)
##newProblemSeries, newfilename =  runallseries(NC1to10, warmstart=True)
##newProblemSeries, newfilename =  runallseries(NC11to20, warmstart=True)
##newProblemSeries, newfilename =  runallseries("ProblemSeries-20140118172244-C-5balls-13.pickle", warmstart=True)
##load_and_view(newfilename)
#
#
### when a pickled problem series already exists and we just want to solve and store it
##newProblemSeries, newfilename =  runallseries("ProblemSeries-20140110034841-C-5balls.pickle", warmstart=True)
##newProblemSeries, newfilename =  runallseries("ProblemSeries-20140110034842-NC-5balls.pickle", warmstart=True, convex=False)