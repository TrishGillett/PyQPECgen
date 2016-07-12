# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 05:03:40 2015

@author: Trish
"""
import scipy
import numpy as np

def choose_num(m):
    return ceil(m*randcst())

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

# ELEPHANT CAN WE ELIMINATE THE NEED FOR THIS
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

def sanitize_params(param):
    # Check type value and objective related stuff.
    if param['qpec_type'] not in [100, 200, 201, 300, 800, 900]:
        print 'Warning: Wrong data for qpec_type. \n'
        param['qpec_type'] = 100
        print 'qpec_type set to 100.\n'
    if param['cond_P'] < 1:
        print 'Warning: cond_P should not be less than 1. \n'
        param['cond_P'] = 20
        print 'cond_P set to 20.\n'
    if type(param['convex_f']) != bool:
        print 'Warning: Wrong data for convex_f.\n'
        param['convex_f'] = True
        print 'convex_f set to True.\n'
    if type(param['symm_M']) != bool:
        print 'Warning: Wrong data for symm_M.\n'
        param['symm_M'] = True
        print 'symm_M set to True.\n'
    if type(param['mono_M']) != bool:
        print 'Warning: Wrong data for mono_M.\n'
        param['mono_M'] = True
        print 'mono_M set to True.\n'
    if param['cond_M'] < 1:
        print 'Warning: cond_M should not be less than 1.\n'
        param['cond_M'] = 10
        print 'cond_M set to 10.\n'
    
    # These params take their default values from other params
    if 'scale_P' not in param or param['scale_P'] <= 0:
        print 'Warning: No value or invalid value given for scale_P.'
        param['scale_P'] = param['cond_P']
        print 'scale_P set equal to cond_P.\n'
    if 'scale_M' not in param or param['scale_M'] <= 0:
        print 'Warning: No value or invalid value given for scale_M.'
        param['scale_M'] = param['cond_M']
        print 'scale_M set equal to cond_M.\n'
    if 'p' not in param:
        print 'Warning: No value for p.'
        param['p'] = param['m']
        print 'p set equal to m.\n'
    
    # Type specific checks
    if param['qpec_type'] == 100 and param['second_deg'] > param['p']:
        print 'Warning: second_deg should not be greater than p.\n'
        param['second_deg'] = param['p']
        print 'second_deg = p\n'
    elif (param['qpec_type'] >= 200 and param['qpec_type'] <= 300) and param['second_deg'] > param['m']:
        print 'Warning: second_deg should not be greater than m.\n'
        param['second_deg'] = param['m']
        print 'second_deg set equal to m.\n'
    elif param['qpec_type'] >= 800:
        if param['l'] != 0:
            print "Warning: l must be equal to 0 for this type."
            param['l'] = 0
        if param['n'] > param['m']:
            print '\n The dimensions have been changed: n set equal to m\n'
            param['n'] = param['m']
    
    # Check params controlling constraint structure
    if param['first_deg'] > param['l']:
        print 'Warning: first_deg should not be greater than l.\n'
        param['first_deg'] = param['l']
        print 'first_deg = l\n'
    if param['mix_deg'] > param['second_deg']:
        print 'Warning: mix_deg should not be greater than second_deg.\n'
        param['mix_deg'] = param['second_deg']
        print 'mix_deg set equal to second_deg\n'
    if type(param['yinfirstlevel']) != bool:
        print 'Warning: yinfirstlevel should be True or False.\n'
        param['yinfirstlevel'] = True
        print 'yinfirstlevel set to True.\n'
    
    # Check params controlling output format
    if 'make_with_dbl_comps' not in param:
        param['make_with_dbl_comps'] = False
    elif param['make_with_dbl_comps'] and (param['qpec_type'] == 200 or param['qpec_type'] == 201):
        print 'Warning: the problem type selected does not result in double comps.\n'
        param['make_with_dbl_comps'] = False
        print 'make_with_dbl_comps set to False.\n'
    
    return param