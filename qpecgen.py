# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:15:45 2013
@author: Patricia Gillett

#######################################################################
## Houyuan Jiang, Daniel Ralph, copyright 1997
## Matlab code accompanied the paper: 
##   Jiang, H., Ralph D. 
##   QPECgen, a MATLAB generator for mathematical programs with 
##   Computational Optimization and Applications 13 (1999), 25–59.
##
## Python implementation coded by Patricia Gillett, 2013
#######################################################################
##
## This code generates random test problems of MPEC with quadratic objective
## functions and affine variational inequality constraints, and certain special
## cases.
##
## The MPEC problem is defined as:
##        #############################################################
##        ##   min   f(x,y)                                          ##
##        ##   s.t.  (x,y) in Z                                      ##
##        ##         y in S(x), S(x) solves AVI(F(x,y), C(x))        ##
##        ##         F(x,y) is linear with respect to both x and y   ##
##        ##         C(x) is polyhedral in y-space                   ## 
##        #############################################################
##
## x:      n dimensional first level variable.
## y:      m dimensional second level variable.
## P:      P=[Px Pxy^T; Pxy Py] -- Hessian of the objective function.
## c, d:   coefficient vectors associated with x and y, respectively.
## A, a:   A is an l by (m+n) matrix, a is an l dimensional vector matrix.
##         Used in the upper level constraints A*[x;y] + a <= 0
##         (* models are described in paper in terms of G and H where A=[G, H])
## F, q, N, M: N is an m by n matrix, M is an m by m matrix, and q is
##             an m by 1 vector.
##             These define F linearly in terms of x and y: F=N*x+M*y+q.
## D, E, b: D is a p by n matrix, E is a p by m matrix, b is an m dimensional
##          vector.  Used in the lower level constraints for type 100 problems.
## u:      m dimensional vector used in lower level constraints for type 200
##         problem.
##
##        ############################################
##        ##           AVI-QPEC (type 100)          ##
##        ############################################
##
##           min   0.5*[x;y]^T*P*[x;y] + [c;d]^T*[x;y]
##           s.t.  A*[x;y] + a <= 0
##                 D*x + E*y + b <= 0
##                 lambda >= 0
##                 (D*x + E*y + b)^Tlambda = 0
##                 N*x + M*y + E^T*lambda + q = 0
##                 
##
##        ############################################
##        ##           BOX-MPEC (type 200)          ##
##        ############################################
##
##        For this case, let y=[y1;y2]　where variables y1 have both upper and
##        lower bounds and y2 variables only have lower bounds.  Because there
##        are no other lower level constraints, the case simplifies and there
##        are no lambda variables.
##
##           min   0.5*[x;y]^T*P*[x;y] + [c;d]^T*[x;y]
##           s.t.  A*[x;y] + a <= 0
##                 0 <= y1 <= u 
##                 0 <= y2
##                 N2*x + M2*y + q2 >= 0    complements    0 <= y2
##                 N1*x + M1*y + q1    complements    0 <= y1 <= u
##                 
## 
##        #####################################################################
##        ##         PATRICIA'S ADDITION: SPECIAL BOX-MPEC (type 201)        ##
##        #####################################################################
##
##        It is convenient for us to have one more type where all second
##        level variables have both lower and upper bounds and all first level
##        variables are bounded above and below as well.
##
##           min   0.5*[x;y]^T*P*[x;y] + [c;d]^T*[x;y]
##           s.t.  A*[x;y] + a <= 0
##                 0 <= x <= ux
##                 0 <= y <= uy
##                 N1*x + M1*y + q1    complements    0 <= y <= uy
##                 
## 
##        ############################################
##        ##          LCP-MPEC type(300)            ##
##        ############################################
##
##           min   0.5*[x;y]^T*P*[x;y] + [c;d]^T*[x;y]
##           s.t.  A*[x;y] + a <= 0
##                 0 <= y
##                 N*x + M*y + q >= 0
##                 (N*x + M*y + q)^Ty = 0
##        
##
##        ############################################
##        ##           GOOD-LCP (type 800)          ##
##        ############################################
##        
##        The objective function is equivalent to sum((x-1)^2) + sum((y+2)^2)
##        shifted by a constant.  It is minimized by the point closest
##        to (1 ... 1,-2 ... -2), which is the origin.
##        
##           min   x^Tx + y^Ty - 2*sum(x) + 4*sum(y)
##           s.t.  x <= y
##                 0 <= y
##                 (y-x)^Ty = 0
##                 
## 
##        ###########################################
##        ##          BAD-LCP type(900)            ##
##        ###########################################
##        
##        This problem has multiple local minima.
##        The objective function is equivalent to sum((x+1)^2) + sum((y-2)^2)
##        shifted by a constant.  It is minimized by the feasible point closest
##        to (-1 ... -1, 2 ... 2), which is (-1 ... -1, 0 ... 0).
##        
##           min   x^Tx + y^Ty + 2*sum(x) - 4*sum(y)
##           s.t.  x <= y
##                 0 <= y
##                 (y-x)^Ty = 0
"""

import datetime
import scipy
import numpy as np
from scipy import io
from cvxopt import matrix

import support
import pickling

#only needed for temporary 'live commands' we're using
import testingpyneos as testing
import resultmanagement as resman
import modelwriter
import datetime

#=============================================================================#

def schur(P):
    """g variables in the
    same order as matlab's schur decomp to make it easy to verify my
    translation from matlab to python.
    Simulates matlab's schur decomposition function, returnin
    """
    PD, PU =  scipy.linalg.schur(P)
    for i in range(len(PD[0, :])):   # make all non-diagonal elements 0.
        for j in range(i+1, len(PD[0, :])):
            PD[i, j] = 0.
            PD[j, i] = 0.
    return PU, PD, PU.T

#=============================================================================#

def svd(P):
    """
    Performs singular value decomposition, forming a matrix PD from the vector
    pd produced by the decomposition.
    """
    [PU, pd, PV] = scipy.linalg.svd(P)
    PD = scipy.linalg.diagsvd(pd, len(pd), len(pd))
    return PU, PD, PV

#=============================================================================#

def rand(m, n=1):
    """
    Convenience function to creates an m by n numpy matrix with each element
    distributed Uniform(0,1).
    """
    return np.matrix(np.random.uniform(size=(m, n)))

#=============================================================================#

def randcst():
    """
    Convenience function to get a single Uniform(0,1) value.
    """
    return np.random.uniform()

#=============================================================================#

def randint(low, high, m, n=1):
    """
    Convenience function to get a random integer in the range [low, high].
    """
    return 1.*np.random.random_integers(low, high=high, size=(m, n))

#=============================================================================#

def zeros(m, n=1):
    """
    Convenience function to create an m by n matrix of all zeroes.
    """
    return np.matrix(np.zeros(shape=(m, n)))

#=============================================================================#

def ones(m, n=1):
    """
    Convenience function to create an m by n matrix of all ones.
    """
    return np.matrix(np.ones(shape=(m, n)))

#=============================================================================#

def eye(n):
    """
    Convenience function for an n by n identity matrix.
    """
    return np.matrix(np.identity(n))

#=============================================================================#

def conmat(L, option='v'):
    """
    Concatenates the matrices L[1]... L[n] into one matrix, arranging them
    horizontally if option='h' and vertically if option='v'.
    """
    if option == 'v':
        return np.matrix(np.vstack(L))
    elif option == 'h':
        return np.matrix(np.hstack(L))

#=============================================================================#

def ceil(x):
    """
    Convenience function for the ceiling of a number x.
    """
    return int(np.ceil(x))

#=============================================================================#

# DOLPHIN CAN WE ELIMINATE THE NEED FOR THIS
def npvec(x):
    """
    Convenience function to make sure x is a vertical vector.
    """
    return np.reshape(x, (len(x), 1))

#=============================================================================#

def mindiag(M):
    """
    Returns the value of the smallest diagonal entry of the matrix M
    """
    return min(M[i, i] for i in range(min(M.shape)))

#=============================================================================#

def maxdiag(M):
    """
    Returns the value of the largest diagonal entry of the matrix M
    """
    return -mindiag(-M)

#=============================================================================#

def randdiag(n):
    """
    Creates an n by n matrix with Uniform(0,1) random numbers on the diag
    """
    M = zeros(n, n)
    for i in range(n):
        M[i, i] = randcst()
    return M

#=============================================================================#

def reconstruct(MU, MD, MV):
    """
    Returns the matrix product MU*MD*MV.
    """
    return np.dot(MU, np.dot(MD, MV))

#=============================================================================#

def adjustcond(PU, PD, PV, cond):
    """
    Constructs the matrix PU*PD*PV after adjusting its condition number to be
    cond.
    """
    size = min(PD.shape)
    ccond = (cond-1)*mindiag(PD)*1./(maxdiag(PD)-mindiag(PD))
    PD = ccond*PD + (1-ccond)*mindiag(PD)*eye(size)
    return reconstruct(PU, PD, PV)

#=============================================================================#

def tweakdiag(MD):
    """
    Takes a square matrix MD, shifts every diagonal element so that the
    smallest diagonal value is 0, and then adds a Uniform(0,1) value to each
    diagonal element.
    """
    m = min(MD.shape)
    return MD - mindiag(MD)*eye(m) + randdiag(m)

#=============================================================================#

def qpecgen(param):
    """
    Constructs a single problem with the desired parameters.
    """
#    print '--------------------------------------------------------\n'
#    print '          =================================\n'
#    print '          ||      Start of qpecgen.m      ||\n'
#    print '          =================================\n'
    
    details = {}
    details['param'] = param
    # Unpack parameters from param, which is a dictionary.
    qpec_type = param['qpec_type']
    n = param['n']
    m = param['m']
    l = param['l']
    p = param['p']
    linobj = param['linobj']
    cond_P = param['cond_P']
    scale_P = param['scale_P']
    convex_f = param['convex_f']
    symm_M = param['symm_M']
    mono_M = param['mono_M']
    cond_M = param['cond_M']
    scale_M = param['scale_M']
    second_deg = param['second_deg']
    first_deg = param['first_deg']
    mix_deg = param['mix_deg']
    tol_deg = param['tol_deg']
    yinfirstlevel = param['yinfirstlevel']
    random = param['random']
    if 'make_with_dbl_comps' in param:
        make_with_dbl_comps = param['make_with_dbl_comps']
    else:
        make_with_dbl_comps = True
    
    np.random.seed = random

    '''
    qpec_type:  Indicate the type of MPEC.
            if qpec_type=100, it is the general AVI-MPEC problem.
            if qpec_type=200, it is BOX-MPEC.
            if qpec_type=300, it is LCP-MPEC.
            if qpec_type=800, it is a special LCP-MPEC having good behaviour.
            if qpec_type=900, it is a special LCP-MPEC having bad behaviour.
    n:  The dimension of the first level variables.
    m:  The dimension of the second level variables.
    l:  The number of the first level inequality constraints.
    p:  The number of the second level inequality constraints for the
            AVI-MPEC problem. In the case of BOX-MPEC, p=m. In the case of
            LCP-MPEC, p=m.
    cond_P:  The condition number of the Hessian of the objective function.
             cond_P can be any number not less than 1.
    scale_P:  Positive scaling constant to roughly control the magnitude 
             of the largest singular value of P.
    convex_f:  This is a boolean element. If convex_f is False, f(x,y) is not
             necessarily convex; if convex_f=True, f(x,y) is convex.
    symm_M:  This is a binary element. If symm_M=0, M is not necessarily
             symmetric; if symm_M=1, M is symmetric.
    mono_M:  This is a binary element. If mono_M=0, M is not necessarily
             monotone; if mono_M=1, M is monotone.
    cond_M:  The condition number of the matrix M. cond_M can be
             any number not less than 1.
    scale_M:  Positive scaling constant to roughly control the magnitude 
             of the largest singular value of M.
    second_deg:  The cardinality of the second level degenerate index set.
             second_deg must not be greater than p for AVI-MPEC,
             than m for BOX-MPEC, and than m for LCP-MPEC.
    first_deg:  The cardinality of the first level degenerate constraints
             index set. first_deg must not be greater than l.
    mix_deg:  The cardinality of the degenerate set associated with 
             inequality constraints coming from the second level problem
             in the relaxed nonlinear program of MPEC.
             mix_deg <= second_deg must be satisfied.
    tol_deg:  a small positive tolerance to measure degeneracy.
    yinfirstlevel: whether or not lower level variables are involved in the
             first level constraints.
    random:  Indicates the random 'seed' for Numpy.'''
        
    ## CHECK CONSISTENCY OF PARAMETER DATA ##
    check = True #change to False to not check consistency of data
    if check:
        if qpec_type not in [100, 200, 201, 300, 800, 900]:
            print 'Warning: Wrong data for qpec_type. \n'
            qpec_type = 100
            print 'qpec_type set to 100.\n'
        if cond_P < 1:
            print 'Warning: cond_P should not be less than 1. \n'
            cond_P = 20
            print 'cond_P set to 20.\n'
        if scale_P <= 0:
            print 'Warning: Wrong data for scale_P.'
            scale_P = cond_P
            print 'scale_P set equal to cond_P.\n'
        if type(convex_f) != bool:
            print 'Warning: Wrong data for convex_f.\n'
            convex_f = True
            print 'convex_f set to True.\n'
        if type(symm_M) != bool:
            print 'Warning: Wrong data for symm_M.\n'
            symm_M = True
            print 'symm_M set to True.\n'
        if type(mono_M) != bool:
            print 'Warning: Wrong data for mono_M.\n'
            mono_M = True
            print 'mono_M set to True.\n'
        if cond_M < 1:
            print 'Warning: cond_M should not be less than 1.\n'
            cond_M = 10
            print 'cond_M set to 10.\n'
        if scale_M <= 0:
            print 'Warning: Wrong data for scale_M.'
            scale_M = cond_M
            print 'scale_M set equal to cond_M.\n'
        if qpec_type == 100 and second_deg > p:
            print 'Warning: second_deg should not be greater than p.\n'
            second_deg = p
            print 'second_deg = p\n'
        elif (qpec_type>=200 and qpec_type<=300) and second_deg > m:
            print 'Warning: second_deg should not be greater than m.\n'
            second_deg = m
            print 'second_deg set equal to m.\n'
        if first_deg > l:
            print 'Warning: first_deg should not be greater than l.\n'
            first_deg = l
            print 'first_deg = l\n'
        if mix_deg > second_deg:
            print 'Warning: mix_deg should not be greater than second_deg.\n'
            mix_deg = second_deg
            print 'mix_deg set equal to second_deg\n'
        if type(yinfirstlevel) != bool:
            print 'Warning: yinfirstlevel should be True or False.\n'
            yinfirstlevel = True
            print 'yinfirstlevel set to True.\n'
        if make_with_dbl_comps and (qpec_type == 200 or qpec_type == 201):
            print 'Warning: the problem type selected does not result in double comps.\n'
            make_with_dbl_comps = False
            print 'make_with_dbl_comps set to False.\n'
    ## The end of checking for consistency of parameter data.
    
    if linobj:
        P = zeros(m+n, m+n)
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
        
        # for later convenience, assign names to the blocks of P
    Px = P[:n, :n]
    Py = P[n:m+n, n:m+n]
    Pxy = P[:n, n:m+n] #Pxy[i,j] gives coeff for x[i]y[j] (but don't forget that P is symmetric so this is doubled)
    
    if l == 0:
        A = zeros(0, m+n)
    elif yinfirstlevel:  ## yinfirstlevel = True means the upper level yinfirstlevels involve x and y
        A = rand(l, n+m) - rand(l, n+m)
    else:  ## yinfirstlevel = False means they only use x variables
        A = conmat([np.matrix(rand(l, n) - rand(l, n)), zeros(l, m)], option='h')
#    if A.size[0] > 0:
#        for i in range(A.size[1]):
#            A[0,i] = abs(A[0,i])
    
    ## Generate matrices of the second level objective function.
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
    N = 1.*(rand(m, n) - rand(m, n))
    # This ends the common data used by all problem types
    
    
    ##################################
    ##           AVI-MPEC           ##
    ##################################
    if qpec_type == 100:
        #############################################
        ##        GENERATE OPTIMAL SOLUTION        ##
        #############################################
        xgen = rand(n) - rand(n)
        ygen = rand(m) - rand(m)
        
        ####################################################
        ##        FIRST LEVEL CTRS A[x;y] + a <= 0        ##
        ####################################################
        # l: number of first degree ctrs
        # first_deg: number of first level ctrs for which the ctr is active AND lambda=0
        # l_nonactive: number ctrs which are not tight at and have lambda=0
        ## randomly decide how many of the non-degenerate first level ctrs should be nonactive
        l_nonactive = np.ceil((l-first_deg)*randcst())
        
        ##  Generate the first level multipliers  ulambda  associated with A*[x;y]+a<=0.
        ##  we let the first first_deg cts be degenerate (ctr is tight and ulambda = zero), the next l_nonactive ctrs be not active (ulambda = 0), and the remaining ctrs be active (ulambda Uniform(0,1))
        ulambda = conmat([zeros(first_deg), zeros(l_nonactive), rand(l-first_deg-l_nonactive)])
        
        ##  Generate a so that the A + a is tight for the first first_deg+l_nonactive ctrs and has a random value <= 0 for the rest
        a = -A*conmat([xgen, ygen]) - conmat([zeros(first_deg), rand(l_nonactive), zeros(l-first_deg-l_nonactive)])
#        
        
        ######################################################
        ##        SECOND LEVEL CTRS Dx + Ey + b <= 0        ##
        ######################################################
        # p: number of second degree ctrs (and therefore the number of lambda vars)
        # second_deg: number of second level ctrs for which the ctr is active AND lambda=0
        # p_nonactive: number of second level ctrs which aren't active.  The corresponding lambdas must therefore be 0
        
        ##  Generate matrices in the second level constraints set.
        D = rand(p, n) - rand(p, n)
        E = rand(p, m) - rand(p, m)
        
        ## randomly decide how many of the non-degenerate second level ctrs should be nonactive
        p_nonactive = np.ceil((p-second_deg)*randcst())   ## Choose a random number of second level ctrs to be nonactive at (xgen, ygen)
        
        ## we let the first second_deg cts be degenerate (ctr is tight and lambda = zero), the next p_nonactive ctrs be not active (lambda = 0), and the remaining ctrs be active (lambda Uniform(0,1))
        lambd = conmat([zeros(second_deg), zeros(p_nonactive), rand(p-second_deg-p_nonactive)])
        
        ## figure out what RHS vector is needed for Dx + Ey + b <= 0
        ## we intentionally build in a gap on the p_nonactive ctrs in the middle
        b = -D*xgen-E*ygen-conmat([zeros(second_deg), rand(p_nonactive), zeros(p-second_deg-p_nonactive)])   ## The first second_deg constraints
        
        
        ##################################################################
        ##        STATIONARITY CONDITION FOR LOWER LEVEL PROBLEM        ##
        ##################################################################
        ## Choose q so that Nx + My + E^Tlambda + q = 0 at the solution (xgen, ygen, lambda)
        q = -N*xgen - M*ygen - (E.T)*lambd ## KKT conditions of the second level problem.
        
        
        #########################################
        ##        For later convenience        ##
        #########################################
        F = N*xgen + M*ygen + q # this must be equal to -E^T\lambda
        g = D*xgen + E*ygen + b   # this is the (negative) amount of slack in the inequalities Dx + Ey + b <= 0
        
        ## Calculate three index sets alpha, beta and gamma at (xgen, ygen)
        indexalpha = []
        indexgamma = []
        index = []
        for i in range(len(lambd)):
            if lambd[i] + g[i] < -tol_deg:
                indexalpha += [1]
            else:
                indexalpha += [0]
            if lambd[i] + g[i] > tol_deg:
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
        
        eta = scipy.linalg.solve(E, sigma)
        
        #############################################################################
        ##        Generate coefficients of the linear part of the objective        ##
        #############################################################################
        xy = conmat([xgen, ygen])
        dxP = conmat([Px, Pxy], option='h')
        dyP = conmat([Pxy.T, Py], option='h')
        Ax, Ay = A[:, :n].T, A[:, n:m+n].T
        ##  Generate c and d such that (xgen, ygen) satisfies KKT conditions
        ##  of AVI-MPEC as well as the first level degeneracy.
        if l == 0:
            c = -(dxP*xy + (N.T)*eta + (D.T)*pi)
            d = -(dyP*xy + (M.T)*eta + (E.T)*pi)
        else:
            c = -(dxP*xy + Ax*ulambda + (N.T)*eta + (D.T)*pi)
            d = -(dyP*xy + Ay*ulambda + (M.T)*eta + (E.T)*pi)
        
        details['D'] = np.matrix(D)
        details['E'] = np.matrix(E)
        details['b'] = npvec(b)
        details['l_nonactive'] = l_nonactive
        details['ulambda'] = npvec(ulambda)
        details['p_nonactive'] = p_nonactive
        details['lambd'] = npvec(lambd)
        details['F'] = npvec(F)
        details['g'] = npvec(g)
        details['indexalpha'] = indexalpha
        details['indexgamma'] = indexgamma
        details['index'] = index
        details['pi'] = npvec(pi)
        details['sigma'] = npvec(sigma)
        details['eta'] = npvec(eta)
    
    
    ##########################################
    ######           BOX-MPEC           ######
    ##########################################
    elif qpec_type == 200 or qpec_type == 201:  ## In the case of BOX-MPEC.
        ## Note that we only consider the following two cases of box constraints:
        ## y(i) in [0, +inf) or  [0, u] where u is a nonnegative scalar.
        ## Clearly, any other interval can be obtained by using the mapping
        ## y <--- c1+c2*y. 
        ## It is assumed that the last m_inf constraints are of the form [0, inf)
        
        ## Type 201 is a more specific case of type 200 where x variables are constrained
        ## xl <= x < xu
        ## and y variables are constrained
        ## 0 <= y <= u
		
        #####################################################################
        ##        Type 201: x variables have upper and lower bounds        ##
        #####################################################################
        if qpec_type == 201 or qpec_type:
            xl = randint(-10, 0, n)
            xu = randint(1, 10, n)
		
        ###################################################
        ##        y variables with no upper bound        ##
        ###################################################
        ## Decide how many variables have bounds of type y in [0, inf)
        m_inf = min(m-1, ceil(m*randcst()))                                           # m_inf: total number of ctrs with this bound type
        m_inf_deg = max(second_deg-m+m_inf, ceil(min(m_inf, second_deg)*randcst()))   # m_inf_deg: F=0, y=0
        inf_nonactive = ceil((m_inf-m_inf_deg)*randcst())                            # inf_nonactive: F>0, y=0
        if qpec_type == 201:
            m_inf = 0
            m_inf_deg = 0
            inf_nonactive = 0
        # The remaining m_inf - m_inf_deg - inf_nonactive constraints are where F=0, y>0
        
        ###################################################
        ##        y variables with an upper bound        ##
        ###################################################
        ## There will be m - m_inf variables with double sided bounds. each upper bound is chosen uniform in [0,10]
        u = 10.*rand(m-m_inf)
        if qpec_type == 201:
			u = randint(0, 10, m-m_inf)
        m_upp_deg = ceil((second_deg-m_inf_deg)*randcst())                           # m_upp_deg F=0, y=u
        m_low_deg = second_deg-m_inf_deg-m_upp_deg                                   # m_low_deg F=0, y=0
        upp_nonactive = ceil((m-m_inf-m_upp_deg-m_low_deg)*randcst())                # upp_nonactive F<0, y=u
        low_nonactive = ceil((m-m_inf-m_upp_deg-m_low_deg-upp_nonactive)*randcst())  # low_nonactive F>0, y=0
        # The remaining m - m_inf - m_upp_deg - upp_nonactive - m_low-deg - low_nonactive are where F=0, 0<y<u
        
        #############################################
        ##        GENERATE OPTIMAL SOLUTION        ##
        #############################################
        xgen = rand(n) - rand(n)
        if type == 201:
            xgen = [xl[i] + (xu[i]-xl[i])*randcst() for i in range(n)]
        v1 = u[m_upp_deg+upp_nonactive+m_low_deg+low_nonactive:m-m_inf]
        v2 = npvec([randcst()*v1[i] for i in range(len(v1))])
        ygen = conmat([npvec(u[:m_upp_deg+upp_nonactive]),             # m_upp_deg (F=0, y=u) and upp_nonactive (F<0, y=u) cases
                     zeros(m_low_deg+low_nonactive),          # m_low_deg (F=0, y=0) and low_nonactive (F>0, y=0) cases
                     v2,                                        # for variables with double sided bounds, which do not fall in the above cases, ie. F=0, 0<y<u
                     zeros(m_inf_deg+inf_nonactive),          # m_inf_deg (F=0, y=0) and  inf_nonactive (F>0, y=0) cases
                     rand(m_inf-m_inf_deg-inf_nonactive)])    # m_inf-m_inf_deg-inf_nonactive (F=0, y>0)
        
        
        ####################################################
        ##        FIRST LEVEL CTRS A[x;y] + a <= 0        ##
        ####################################################
        # Randomly decide how many of the non-degenerate first level ctrs should be nonactive
        l_nonactive = ceil((l-first_deg)*randcst())
        
        # Let the first first_deg cts be degenerate (ctr is tight and ulambda = zero), the next l_nonactive ctrs be not active (ulambda = 0), and the remaining ctrs be active (ulambda Uniform(0,1))
        ulambda = conmat([zeros(first_deg+l_nonactive), rand(l-first_deg-l_nonactive)])
        
        # Generate a so that the A + a is tight for the first first_deg+l_nonactive ctrs and has a random value <= 0 for the rest
        a = -A*conmat([xgen, ygen])-conmat([zeros(first_deg), rand(l_nonactive), zeros(l-first_deg-l_nonactive)])
        
        
        ##################################################################
        ##        STATIONARITY CONDITION FOR LOWER LEVEL PROBLEM        ##
        ##################################################################
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
        
        
        #############################################################################
        ##        Generate coefficients of the linear part of the objective        ##
        #############################################################################
        ##  Generate c and d such that (xgen, ygen) satisfies KKT conditions
        ##  of BOX-MPEC as well as the first level degeneracy.
        
        if l == 0:
            c = -(Px*xgen+Pxy*ygen+(N.T)*pi)
            d = -(Py*ygen+(Pxy.T)*xgen+(M.T)*pi+sigma)
        else:
            c = -(Px*xgen+Pxy*ygen+((A[:, :n]).T)*ulambda+(N.T)*pi)
            d = -(Py*ygen+(Pxy.T)*xgen+((A[:, n:m+n]).T)*ulambda+(M.T)*pi+sigma)
        
        if qpec_type == 201:
            details['xl'] = npvec(xl)
            details['xu'] = npvec(xu)
        details['u'] = npvec(u)
        details['m_inf'] = m_inf
        details['m_inf_deg'] = m_inf_deg
        details['inf_nonactive'] = inf_nonactive
        details['m_upp_deg'] = m_upp_deg
        details['m_low_deg'] = m_low_deg
        details['upp_nonactive'] = upp_nonactive
        details['low_nonactive'] = low_nonactive
        details['l_nonactive'] = l_nonactive
        details['ulambda'] = npvec(ulambda)
        details['F'] = npvec(F)
        details['index'] = index
        details['mix_upp_deg'] = mix_upp_deg
        details['mix_low_deg'] = mix_low_deg
        details['mix_inf_deg'] = mix_inf_deg
        details['pi'] = npvec(pi)
        details['sigma'] = npvec(sigma)
        details['make_with_dbl_comps'] = make_with_dbl_comps
    
    
    ##################################
    ##           LCP-MPEC           ##
    ##################################
    elif qpec_type == 300:
        #############################################
        ##        GENERATE OPTIMAL SOLUTION        ##
        #############################################
        xgen = rand(n, 1)-rand(n, 1)
        m_nonactive = ceil((m-second_deg)*randcst())   # The number of indices where the second level objective function is not active at (xgen, ygen).
        ygen = conmat([zeros(second_deg+m_nonactive, 1), rand(m-second_deg-m_nonactive, 1)])  # The first second_deg+m_nonactive elements of ygen are active.
        
        ##  Generate the vector in the second level objective function.
        q = -N*xgen-M*ygen+conmat([zeros(second_deg, 1), rand(m_nonactive, 1), zeros(m-second_deg-m_nonactive, 1)])
        #The first second_deg indices are degenerate at (xgen, ygen).
        F = N*xgen + M*ygen + q       ## The introduction of F is for later convenience.
        
        ####################################################
        ##        FIRST LEVEL CTRS A[x;y] + a <= 0        ##
        ####################################################
        ##  Generate the first level multipliers  ulambda  associated with   A*[x;y]+a<=0.
        l_nonactive = ceil((l-first_deg)*randcst()) ## The number of nonactive
        #     first level constraints at (xgen, ygen).
        ulambda = conmat([zeros(first_deg+l_nonactive, 1), rand(l-first_deg-l_nonactive, 1)])
        
        ##  Generate the vector in the first level constraints set.
        a = -A*conmat([xgen, ygen])-conmat([zeros(first_deg, 1), rand(l_nonactive, 1), zeros(l-first_deg-l_nonactive, 1)])   ## The first first_deg constraints
        ##       are degenerate, the next l_nonative constraints are not active,
        ##       and the last l-first_deg-l_nonactive constraints are active
        ##       but nondegenerate at (xgen, ygen).
        
        #################################
        ##        For later use        ##
        #################################
        ##  Calculate three index set alpha, beta and gamma at (xgen, ygen).
        indexalpha = []
        indexgamma = []
        index = []
        for i in len(F):
            if F[i]+tol_deg < ygen[i]:
                indexalpha += [1]
            else:
                indexalpha += [0]
            if F[i] > ygen[i]+tol_deg:
                indexgamma += [-1]
            else:
                indexgamma += [0]
            index += [indexalpha[-1] + indexgamma[-1]]
        ## index(i)=1 iff F(i)+tol_deg<ygen(i),
        ##  index(i)=0 iff |F(i)-ygen(i)|<=tol_deg,
        ##  index(i)=-1 iff F(i)>ygen(i)+tol_deg.
        ##
        ## Generate the first level multipliers associated with other constraints
        ## other than the first level constraints   A*[x;y]+a<=0   in the relaxed
        ## nonlinear program. In particular,   pi  and  sigma  are associated with  
        ## F(x, y)=N*x+M*y+q   and    y   in the relaxed nonlinear program.
        k_mix = 0
        for i in range(m):
            if index[i] == -1:
                pi[i] = 0
                sigma[i] = randcst()-randcst()
            elif index[i] == 0:
                if k_mix < mix_deg:
                    pi[i] = 0  ## The first mix_deg constraints associated
                    #with F(x, y)>=0 in the set beta are degenerate.
                    sigma[i] = 0  ## The first mix_deg constraints
                    # associated with y>=0 in the set beta are degenerate.
                    k_mix = k_mix+1
                else:
                    pi[i] = randcst()
                    sigma[i] = randcst()
            else:
                pi[i] = randcst()-randcst()
                sigma[i] = 0
        
        #############################################################################
        ##        Generate coefficients of the linear part of the objective        ##
        #############################################################################
        ##  Generate c and d such that (xgen, ygen) satisfies KKT conditions
        ##  of LCP-MPEC as well as the first level degeneracy.
        if l == 0:
            c = (N.T)*pi-Px*xgen-Pxy*ygen
            d = (M.T)*pi+sigma-Py*ygen-(Pxy.T)*xgen
        else:
            c = (N.T)*pi-Px*xgen-Pxy*ygen-((A[:, :n]).T)*ulambda
            d = (M.T)*pi+sigma-Py*ygen-(Pxy.T)*xgen-((A[:, n:n+m]).T)*ulambda
        ##  The end of LCP-MPEC.
        
        details['m_nonactive'] = m_nonactive
        details['l_nonactive'] = l_nonactive
        details['F'] = npvec(F)
        details['ulambda'] = npvec(ulambda)
        details['indexalpha'] = indexalpha
        details['indexgamma'] = indexgamma
        details['index'] = index
        details['pi'] = npvec(pi)
        details['sigma'] = npvec(sigma)
        ##################################
        ##    Good and bad LCP-MPEC     ##
        ##################################
    
    # The start of Good LCP-MPEC.
    elif qpec_type == 800:
        l = 0
        if n > m:
            n = m
            print '\n The dimensions have been changed: n=m\n'
        P = 2.*eye(n+m)
        c = -2.*ones(n)
        d = 4.*ones(m)
        A = zeros(l, n+m)
        a = zeros(l)
        N = conmat([-eye(n), zeros(m-n, n)], option='v')
        M = eye(m)
        q = zeros(m)
        xgen = zeros(n)
        ygen = zeros(m)
        # The end of Good LCP-MPEC.
    
    ## The start of Bad LCP-MPEC. 
    elif qpec_type == 900:
        l = 0
        if n > m:
            n = m
            print '\n The dimensions have been changed: n=m\n'
        P = 2.*eye(n+m)
        c = 2.*ones(n)
        d = -4.*ones(m)
        A = zeros(l, n+m)
        a = zeros(l)
        N = conmat([-eye(n), zeros(m-n, n)], option='v')
        M = eye(m)
        q = zeros(m)
        xgen = -ones(n)
        ygen = zeros(m)
        # The end of Bad example.
    
    ##################################
    ##       Output                 ##
    ##################################
    
#    print '  \n'
#    if qpec_type == 100:
#        print '************ An example of AVI-MPEC *************\n\n'
#    elif qpec_type == 200:
#        print '************ An example of BOX-MPEC *************\n\n'
#    elif qpec_type == 201:
#        print '************ An example of Bounded BOX-MPEC *************\n\n'
#    elif qpec_type == 300:
#        print '************ An example of LCP-MPEC *************\n\n'
#    elif qpec_type == 800:
#        print '************ A good example of LCP-MPEC *************\n\n'
#    elif qpec_type == 900:
#        print '************ A bad example of LCP-MPEC *************\n\n'
    
    displaydata = False
    if displaydata:
        print '\n\n'
        print 'qpec_type={0}\n'.format(qpec_type)
        print 'n={0}'.format(n)
        print 'm={0}'.format(m)
        print 'l={0}'.format(l)
        print 'p={0}'.format(p)
        print 'cond_P={0}'.format(cond_P)
        print 'convex_f={0}'.format(convex_f)
        print 'symm_M={0}'.format(symm_M)
        print 'mono_M={0}'.format(mono_M)
        print 'cond_M={0}'.format(cond_M)
        print 'second_deg={0}'.format(second_deg)
        print 'first_deg={0}'.format(first_deg)
        print 'mix_deg={0}'.format(mix_deg)
        print 'tol_deg={0}'.format(tol_deg)
        print 'yinfirstlevel={0}'.format(yinfirstlevel)
        print 'random={0}'.format(random)
        print 'make_with_dbl_comps={0}'.format(make_with_dbl_comps)
    
    P = 0.5*(P+P.T)   ## To avoid rounding errors during computation.
    
    details['typ'] = qpec_type
    details['P'] = np.matrix(P)
    details['c'] = npvec(c)
    details['d'] = npvec(d)
    details['A'] = np.matrix(A)
    details['a'] = npvec(a)
    details['N'] = np.matrix(N)
    details['M'] = np.matrix(M)
    details['q'] = npvec(q)
    
    optsolxy = conmat([xgen, ygen])
    optval = 0.5*(optsolxy.T)*P*optsolxy+conmat([c, d]).T*optsolxy  
    details['optval'] = optval[0, 0]
    if qpec_type == 100:
        details['optsol'] = npvec(conmat([xgen, ygen, lambd]))
    else:
        details['optsol'] = npvec(optsolxy)
    
    
    print "Problem generation complete, passing to makeqpec."
    return makeqpec(details)




#=============================================================================#


def makeqpec(D):
    """
    Given a problem's parameters and type, constructs the corresponding
    Problem.
    """
#    typ = D['typ']
#    Pobj = D['P']
#    A = D['A']
#    N = D['N']
#    M = D['M']
#    c = D['c']
#    d = D['d']
#    a = D['a']
#    q = D['q']
#    optsol = D['optsol']
#    optval = D['optval']
    
    l = len(D['a'])  ## number of first level inequality ctrs
    n = len(D['c'])  ## number of x vars
    m = len(D['d'])  ## number of y vars
    optsol = D['optsol']

    P = {}
    if D['typ'] == 100:
        p = len(D['b'])  ## number of g ctrs, number of lambda vars, number of equalities        
        P['names'] = ['x{0}'.format(i) for i in range(n)] + ['y{0}'.format(i) for i in range(m)] + ['L{0}'.format(i) for i in range(p)]
        Q1 = conmat([0.5*D['P'], zeros(n+m, p)], option='h')
        Q2 = conmat([zeros(p, n+m+p)], option='h')
        P['Q'] = conmat([Q1, Q2])
        P['p'] = conmat([D['c'], D['d'], zeros(p, 1)])
    elif D['make_with_dbl_comps'] and (D['typ'] == 200 or D['typ'] == 201):
        mdouble = len(D['u'])
        P['names'] = ['x{0}'.format(i) for i in range(n)] + ['y{0}'.format(i) for i in range(m)] + ['lL{0}'.format(i) for i in range(mdouble)] + ['lU{0}'.format(i) for i in range(mdouble)]
        P['Q'] = matrix([[0.5*D['P'], zeros(2*mdouble, m+n)], [zeros(m+n, 2*mdouble), zeros(mdouble, mdouble)]])
        P['p'] = conmat([D['c'], D['d'], zeros(2*mdouble)])
    else:
        P['names'] = ['x{0}'.format(i) for i in range(n)] + ['y{0}'.format(i) for i in range(m)]
        P['Q'] = 0.5*D['P']
        P['p'] = conmat([D['c'], D['d']])
    P['r'] = 0
    
    if D['typ'] == 100:
        ## keys in D: typ, P, A, N, M, c, d, a, q, optsol, optval, D, E, b
        p = len(D['b'])  ## number of g ctrs, number of lambda vars, number of equalities
        
        # in order of variables: x variables (n), y variables (m), lambda variables (p)
        P['A'] = conmat([D['N'], D['M'], D['E'].T], option='h')
        P['b'] = -D['q']
        G1 = conmat([D['A'], zeros(l, p)], option='h')
        G2 = conmat([D['D'], D['E'], zeros(p, p)], option='h')
        G3 = conmat([zeros(p, n+m), -eye(p)], option='h')
        P['G'] = conmat([G1, G2, G3])
        P['h'] = conmat([-D['a'], -D['b'], zeros(p, 1)])
        P['comps'] = [[l+i, l+p+i] for i in range(p)]
        P['varsused'] = [1]*(n+m) + [0]*p
    
    elif D['typ'] >= 200 and D['typ'] <= 202:
        # in order of variables: x variables (n), double sided y variables (mdouble), single sided y vars (msingle)
        
        if not D['make_with_dbl_comps']:
        # type 200: D's keys: typ, P, A, N, M, c, d, a, q, optsol, optval, u
            mdouble = len(D['u'])
            msingle = m - mdouble
            G1 = conmat([D['A'], zeros(l, 2*mdouble)], option='h')
            G2 = conmat([zeros(m, n), -eye(m), zeros(m, 2*mdouble)], option='h')
            G3 = conmat([zeros(mdouble, n), eye(mdouble), zeros(mdouble, m+mdouble)], option='h')
            G4 = conmat([zeros(mdouble, n+m), -eye(mdouble), zeros(mdouble, msingle)], option='h')
            G6 = conmat([zeros(mdouble, n+m+mdouble), -eye(mdouble)], option='h')
            if D['typ'] == 200:
                G5 = conmat([D['N'][mdouble:], D['M'][mdouble:], zeros(2*mdouble, msingle)], option='h')                
                P['G'] = conmat([G1, G2, G3, G4. G5, G6])
                P['h'] = conmat([-D['a'], zeros(m), D['u'], zeros(mdouble), -D['q'][mdouble:], zeros(mdouble)])
            else:
                P['G'] = conmat([G1, G2, G3, G4. G6])
                P['h'] = conmat([-D['a'], zeros(m), D['u'], zeros(mdouble), zeros(mdouble)])
            P['comps'] = [[l+i, l+mdouble+m+i] for i in range(m+mdouble)]
            P['A'] = conmat([-D['N'][:mdouble], -D['M'][:mdouble], -eye(mdouble), eye(mdouble)], option='h')
            P['b'] = -D['q'][:mdouble]
            v = conmat([D['N'], D['M']], option='h')*optsol + D['q']
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
            optsol = conmat([optsol, lambdaL, lambdaU])
            
            
        if D['typ'] == 200:
        # type 200: D's keys: typ, P, A, N, M, c, d, a, q, optsol, optval, u
            mdouble = len(D['u'])
            msingle = m - mdouble
            G1 = D['A']
            G2 = conmat([zeros(mdouble, n), -eye(mdouble), zeros(mdouble, msingle)], option='h')
            G3 = conmat([zeros(mdouble, n), eye(mdouble), zeros(mdouble, msingle)], option='h')
            G4 = conmat([zeros(msingle, n), zeros(msingle, mdouble), -eye(msingle)], option='h')
            G5 = conmat([D['N'][mdouble:, :], D['M'][mdouble:, :]], option='h')
            P['G'] = conmat([G1, G2, G3, G4, G5])
            P['h'] = conmat([-D['a'], zeros(mdouble), D['u'], zeros(msingle), -D['q'][mdouble:]])
            P['comps'] = [[l+2*mdouble + i, l+mdouble+m+i] for i in range(msingle)]
            P['expF'] = conmat([-D['N'][:mdouble, :], -D['M'][:mdouble, :]], option='h')
            P['exph'] = D['q'][:mdouble]
            # doublecomps: [i,j,k] means expression exphi - expFiTx forms a nonpositive product with hj-Gjx and a nonnegative product with hk-Gkx
            P['doublecomptuples'] = [[i, l+mdouble+i, l+i] for i in range(mdouble)]
        
        elif D['typ'] == 201:
        # type 201: D's keys: typ, P, A, N, M, c, d, a, q, optsol, optval, u, xl, xu
            G1 = D['A']
            G2 = conmat([-eye(n), zeros(n, m)], option='h')
            G3 = conmat([eye(n), zeros(n, m)], option='h')
            G4 = conmat([zeros(m, n), -eye(m)], option='h')
            G5 = conmat([zeros(m, n), eye(m)], option='h')
            P['G'] = conmat([G1, G2, G3, G4, G5])
            P['h'] = conmat([-D['a'], -D['xl'], D['xu'], zeros(m), D['u']])
            P['expF'] = conmat([-D['N'], -D['M']], option='h')
            P['exph'] = D['q']
            # doublecomps: [i,j,k] means expression exphi - expFiTx forms a nonpositive product with hj-Gjx and a nonnegative product with hk-Gkx
            P['doublecomptuples'] = [[i, l+2*n+i, l+2*n+m+i] for i in range(m)]
    
    elif D['typ'] == 300:
        print "makeqpec does not currently convert type 300 problems to QPCCs"
##           s.t.  A*[x;y] + a <= 0
##                 0 <= y
##                 N*x + M*y + q >= 0
##                 (N*x + M*y + q)^Ty = 0
    
    elif D['typ'] == 800 or D['typ'] == 900:
        # type 800/900: D's keys: typ, P, A, N, M, c, d, a, q, optsol, optval
        P['A'] = conmat([zeros(m-n, 2*n), eye(m-n)])
        P['b'] = zeros(m-n)
        G1 = conmat([-D['N'][:n, :], -D['M'][:n, :]], option='h')
        G2 = conmat([zeros(n, n), -eye(n), zeros(m-n, n)], option='h')
        P['G'] = conmat([G1, G2])
        P['h'] = conmat([D['q'], zeros(n)])
        P['comps'] = [[i, n+i] for i in range(n)]
    
    P['trueoptsol'] = optsol
    P['trueoptval'] = D['optval']
#    if typ == 201:
#        B0 = 0
#        for i in range(n):
#            B0 += max(xl[i]**2, xu[i]**2)
#        for i in range(m):
#            B0 += u[i]**2
#        B0 = np.sqrt(B0)[0]
#        B0 = np.sqrt(sum([max(xl[i]*xl[i], xu[i]*xu[i]) for i in range(n)]) + sum([u[i]*u[i] for i in range(m)]))[0,0]
#        P['Bs'] = [1.10*B0]
#        P['Bs'] = [1.10*B0, 2.*B0, 10*B0]
#        P['Bs'] = []
#    else:
#        B0 = np.sqrt(sum(x*x for x in optsol[:n+m])[0, 0])
#        P['Bs'] = [1.10*B0, 1.20*B0, 2.*B0, 10*B0, 100*B0]
#        P['Bs'] = []
#    print "An appropriate B list for this problem would be {0}.".format(P['Bs'])
    P['Btypes'] = ['varsused']
    P['Bs'] = []
    P['details'] = D
    if 'varsused' not in P:
        P['varsused'] = [1]*(n+m)
#    print P
    return P




#=============================================================================#

def Problem_from_P(P, pname, timestamp):
    """
    Takes a dictionary P, problem name, and timestamp, and constructs a problem
    dictionary.
    """
    trueoptval = P['trueoptval']
    trueoptsol = matrix(P['trueoptsol']+1-1)
    Q = matrix(P['Q']+1-1)
    p = matrix(P['p']+1-1)
    r = P['r']
    names = P['names']
    varsused = P['varsused']
    
    s = support.obj_as_str(Q, p, r, names)
    
    Problem = {'pname': pname,
                  'mode': 'min',
                  'names': names,
                  'G': matrix(P['G']+1-1),
                  'h': matrix(P['h']+1-1),
                  'Bs': P['Bs'],
                  'N': 1,
                  'varsused': varsused,
                  'clas': 'qpecgen',
                  
                  'details': P['details'],
                  'timestamp': timestamp,
                  'results': [{'solver': 'gen', 'type': 'gen', 'timestamp': '00000000000000', 'status': 'optimal', 'note': 'feasible from QPECgen', 'B': -1,
                               'value': trueoptval, 'suggestedsol': trueoptsol, 'ID': 0, 'solvetime': 0}],
                  'nextID': 1,
                  'Obj': {'Q': Q,
                          'p': p,
                          'r': r,
                          'objstr': s,
                          'mode': 'min'}}
    if 'param' in P['details']:
        Problem['param'] = P['details']['param']
        Problem['type'] = P['details']['param']['qpec_type']
    
    if 'Btypes' in P:
        Problem['Btypes'] = P['Btypes']
    if 'A' in P:
        Problem['A'] = matrix(P['A']+1-1)
        Problem['b'] = matrix(P['b']+1-1)
    else:
        Problem['A'] = matrix(0., (0, len(names)))
        Problem['b'] = matrix(0., (0, 1))
    if 'comps' in P:
        Problem['comps'] = P['comps']
    else:
        Problem['comps'] = []
    if 'doublecomptuples' in P:
        Problem['doublecomptuples'] = P['doublecomptuples']
        Problem['expF'] = matrix(P['expF']+1-1)
        Problem['exph'] = matrix(P['exph']+1-1)
    else:
        Problem['doublecomptuples'] = []
        Problem['expF'] = matrix(0., (0, len(names)))
        Problem['exph'] = matrix(0., (0, 1))    
#    print "Optimal value of problem {0} is {1} at\n{2}.".format(pname, P['trueoptval'], P['trueoptsol'])
    return Problem




def qpec_generate(seriesname, param, N):
    """
    Generates N problems with the parameters given in the dictionary param,
    all named seriesname_{i}.
    """
    ProblemSeries = []
    timestamp = support.get_timestamp()
    
    for i in range(N):
        P = qpecgen(param)
        ProblemSeries += [Problem_from_P(P, seriesname + str(i), timestamp)]
    return ProblemSeries

def qpec_generate2(param, tuplist, Neach, start=0):
    """
    Generates N problems with the parameters given in the dictionary param,
    all named seriesname_{i}.
    """
    ProblemSeries = []
    timestamp = support.get_timestamp()
    
    for tup in tuplist:
        param['qpec_type'] = tup[0]
        param['n'] = tup[1]
        param['m'] = tup[2]
        param['l'] = tup[2]
        param['p'] = tup[2]
        for k in range(start, start+Neach):
            P = qpecgen(param)
            ProblemSeries += [Problem_from_P(P, 'qpecgen_{0}_{1}_{2}_no{3}'.format(tup[0], tup[1], tup[2], k), timestamp)]
    return ProblemSeries

tuplist = [[201, 5, 2], [201, 10, 5], [201, 20, 10], [201, 30, 15], [201, 50, 25], [201, 60, 30]]



param_inst = {}
param_inst['qpec_type'] = 201            # 100, 200, 300, 800, 900.
param_inst['n'] = 5                       # Dimension of the variable x.
param_inst['m'] = 2                       # Dimension of the variable y.
param_inst['l'] = 2                       # Number of the first level constraints.
param_inst['p'] = param_inst['m']               # Number of the second level constraints for AVI-MPEC.
param_inst['cond_P'] = 100                  # Condition number of the Hessian P.
param_inst['scale_P'] = param_inst['cond_P']    # Scaling constant for the Hessian P.
param_inst['convex_f'] = True            # True or False. Convexity of the objective function.
param_inst['linobj'] = False
#param_inst['linobj'] = True
param_inst['symm_M'] = True              # True or False. Symmetry of the matrix M.
param_inst['mono_M'] = True              # True or False. Monotonicity of the matrix M.
param_inst['cond_M'] = 100                  # Condition number of the matrix M.
param_inst['scale_M'] = param_inst['cond_M']    # Scaling constant for the matrix M.
param_inst['second_deg'] = 0               # Number of the second level degeneracy.
param_inst['first_deg'] = 1                # Number of the first level degeneracy.
param_inst['mix_deg'] = 0                  # Number of mixed degeneracy.
param_inst['tol_deg'] = 10**(-6)           # Small positive tolerance for measuring degeneracy.
param_inst['yinfirstlevel'] = True         # Whether or not the lower level variables y are involved in the upper level constraints
param_inst['random'] = 0                   # Indicates the random 'seed'.
param_inst['make_with_dbl_comps'] = False




###########################################################################
### functions for importing from qpecgen generated .mat files
###########################################################################


def dotmats_to_ProblemSeries(filenamebase, typ, N, N0=1):
    """
    Reads in problems generated from the matlab qpecgen code, creating a
    ProblemSeries.
    """
    timestamp = support.get_timestamp()
    ProblemSeries = []
    for i in range(N0, N+1):
        filename = 'matlabQPECgen/{0}_{1}_{2}.mat'.format(filenamebase, typ, i)
#        print filename
        P = qpecgen_dotmat_to_qpcc(filename, typ)
#        print P
        ProblemSeries += [Problem_from_P(P, '{0}_{1}_{2}'.format(filenamebase, typ, i), timestamp)]
    return ProblemSeries





def qpecgen_dotmat_to_qpcc(filename, typ):
    """
    Returns a Problem corresponding to the problem defined by filename's
    .mat.
    """
    Di = dotmat_to_dict(filename)
    Di['optsol'] = conmat([Di['xgen'], Di['ygen']])
    Di['optval'] = 0.5*(Di['optsol'].T)*Di['P']*Di['optsol'] + conmat([Di['c'], Di['d']]).T*Di['optsol']
    Di['typ'] = typ
    return makeqpec(Di)

        
        

def dotmat_to_dict(filename):
    """
    Given a matlab .mat file, creates a dictionary containing the data with
    variable names for keys.
    """
    D = io.loadmat(filename)
    for key in D:
        if key[:2] != '__':
            D[key] = np.matrix(D[key])
    return D













def get_solve_timestamp_from_AMPL_folder(amplfolder):
    kleft = amplfolder.rfind('-')
    timestamp = amplfolder[kleft+1:]
    if timestamp[-1]=='\\' or timestamp[-1]=='/':
        timestamp = timestamp[:-1]
    if len(timestamp) != 14:
        raise Exception("timestamp {0} is not 14 characters long like we expect, for example '20140812175247'.".format(timestamp))
    return timestamp




def resume_from_regular_solves(picklefile, amplfolder):
    timestamp = get_solve_timestamp_from_AMPL_folder(amplfolder)
    solvers=['PKNITRO', 'BARONMIP']
    ProblemSeries = pickling.loadProblemSeries(picklefile)
    
    pybatchfile = amplfolder + "pybatch.pickle"
    pybatch = modelwriter.load_batch(pybatchfile)
    resultfile = amplfolder + "AMPLresults.csv"
    resultfileB = amplfolder + "AMPLresultsB.csv"
    
    pybatch.solve_all(resultfile)
    ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfile, timestamp)
    
    ProblemSeries, resultfileB = testing.bounded_BARON_solves(solvers, ProblemSeries, pybatch, resultfileB)
    ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfileB, timestamp)
    testing.merge_csv([resultfile, resultfileB])




def resume_from_bounded_BARON(picklefile, amplfolder):
    timestamp = get_solve_timestamp_from_AMPL_folder(amplfolder)
    ProblemSeries = pickling.loadProblemSeries(picklefile)
    
    pybatchfile = amplfolder + "pybatch.pickle"
    pybatch = modelwriter.load_batch(pybatchfile)
    pybatch.basefolder = "C:/Users/Trish/Documents/Research/AMPL/20140815013415-ran-at-20140815024744/"
    resultfile = amplfolder + "AMPLresults.csv"
    resultfileB = amplfolder + "AMPLresultsB.csv"
    
    pybatch.solve_all(resultfileB)
    ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfileB, timestamp)
    testing.merge_csv([resultfile, resultfileB])




###########################################################################
### generating QPECgen problems
###########################################################################

#PS = qpec_generate('temp', param_inst, 1)
#filename = pickling.saveProblemSeries(PS, nameoverride='qpecgen_type200_C')

#print "saved to {0}".format(filename)
#for P in ProblemSeries:
#    print P
##    print P['AMPLModel']
#    print "==========================================================="

#nmlist = [[5, 2], [10, 5], [20, 10], [30, 15], [40, 20]]
#









#picklefile = "ProblemSeries-20140813202421-results-for-report-10-5-5-o0.0001-a0.0001-20140814064555.pickle"
#amplfolder = "C:/Users/Trish/Documents/Research/AMPL/20140813202421-ran-at-20140814064555/"
#resume_from_regular_solves(picklefile, amplfolder)
#
#
#solvers = ['PKNITRO', 'BARONMIP']
#opts = [[0, 0], [0, 0.0001]]
#for tup in opts:
#    print "WAKAWAKA"
#    optcr = tup[0]
#    alpha = tup[1]
#    testing.runallseries("ProblemSeries-20140813202421-results-for-report-10-5-5.pickle", balls='no', solvename='o{0}-a{1}'.format(optcr, alpha), solvers=solvers, BARONbounds=True, optcr=optcr, alpha=alpha)
#



#### run this after verifying file names are done right
#timestamp = '20140812093123'
#folder = "C:/Users/Trish/Documents/Research/AMPL/20140812025245-ran-at-2014081209312she-hulk ceremony part 23/"
#filename = "ProblemSeries-20140721204625-10problems9vars.pickle"

#pybatchfile = folder + "pybatch.pickle"
#pybatch = modelwriter.load_batch(pybatchfile)
#resultfile = folder + "AMPLresults.csv"
#resultfileB = folder + "AMPLresultsB.csv"
#pybatch.solve_all(resultfileB)
#ProblemSeries = pickling.loadProblemSeries(filename)
#ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfileB, timestamp)
#testing.merge_csv([resultfile, resultfileB])






#pybatch = modelwriter.load_batch('C:/Users/Trish/Documents/Research/AMPL/20140721204625-ran-at-20140724011009/pybatch.pickle')
#resultfile = "C:/Users/Trish/Documents/Research/AMPL/20140721204625-ran-at-20140724011009/AMPLresultsB.csv"
#timestamp = support.get_timestamp()
#ProblemSeries = pickling.loadProblemSeries(filename)
#ProblemSeries, resultfile = testing.bounded_BARON_solves(solvers, ProblemSeries, pybatch, resultfile)
#ProblemSeries = testing.import_results_from_AMPL_csv(ProblemSeries, resultfile, timestamp)
#testing.merge_csv(["C:/Users/Trish/Documents/Research/AMPL/20140721204625-ran-at-20140724011009/AMPLresults.csv", resultfile])


#print_averages(PS)
#plot_M_randvsWS_gapdist(loadProblemSeries("ProblemSeries-20140423224259-qpecgen_type201_NC-ballsno-20140423224305.pickle"))


#mergeMacMPEC()
## PROBLEM: runallseries expects 'param' for each Problem
#PS, filename = runallseries('MacMPECplus.pickle', warmstart=True, balls=balls, solvename='balls' + balls, closest_to_warmstart=True)
#load_and_view('ProblemSeries-20140423224259-qpecgen_type201_NC-ballsyes-20140423224305.pickle', withsols=False, convex=False)


#print_ProblemSeries_results(PSeries, withsols=False)



###########################################################################
### importing QPECgen problems generated as .mat files from matlab
###########################################################################

#to import a single .mat:
#P = qpecgen_dotmat_to_qpcc('matlabQPECgen/qpecgen_data_c_100_1.mat', 100)

### ELEPHANT: before running this one, need to update makeqpec to handle type 200
#ProblemSeries = dotmats_to_ProblemSeries('qpecgen_data_c', 200, 1)
#filename = saveProblemSeries(ProblemSeries, nameoverride='qpecgen_matlab')
#print "saved to {0}".format(filename)
#newProblemSeries, newfilename =  runallseries(filename, warmstart=True)

#ProblemSeries = dotmats_to_ProblemSeries('qpecgen_data_nc', 100, 20)
#filename = pickling.saveProblemSeries(ProblemSeries, nameoverride='matlab nonconvex unbounded demo')
#print "saved to {0}".format(filename)




###########################################################################
### running a pickled problem series
###########################################################################

## Use these ones to import QPECGEN problems from .mat files and then solve them.
#C1to10 = "aardvark-C-5balls-1-to-10-ProblemSeries-20140110125455.pickle"
#C11to20 = "aardvark-C-5balls-11-to-20-ProblemSeries-20140110125456.pickle"
#NC1to10 = "aardvark-NC-5balls-1-to-10-ProblemSeries-20140110125457.pickle"
#NC11to20 = "aardvark-NC-5balls-11-to-20-ProblemSeries-20140110125457.pickle"

#newProblemSeries, newfilename =  runallseries(C1to10, warmstart=True)
#newProblemSeries, newfilename =  runallseries(C11to20, warmstart=True)
#newProblemSeries, newfilename =  runallseries(NC1to10, warmstart=True)
#newProblemSeries, newfilename =  runallseries(NC11to20, warmstart=True)
#newProblemSeries, newfilename =  runallseries("ProblemSeries-20140118172244-C-5balls-13.pickle", warmstart=True)
#load_and_view(newfilename)


## when a pickled problem series already exists and we just want to solve and store it
#newProblemSeries, newfilename =  runallseries("ProblemSeries-20140110034841-C-5balls.pickle", warmstart=True)
#newProblemSeries, newfilename =  runallseries("ProblemSeries-20140110034842-NC-5balls.pickle", warmstart=True, convex=False)