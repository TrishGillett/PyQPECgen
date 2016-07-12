# pylint: disable=E1101
import scipy
import scipy.linalg
import numpy as np


def _is_diagonal(M):
    """ Check a given matrix is a diagonal matrix.

    Args:
      M: Matrix.

    Returns:
      True if the given matrix is a diagonal matrix.

    Raises:
      ValueError: if the given matrix is not square.
    """
    shape = M.shape
    if shape[0] != shape[1]:
        raise ValueError("M is not square.")
    for i in range(shape[0]):
        for j in range(shape[0]):
            if i != j and M[i, j] != 0:
                return False
    return True


def choose_num(m):
    """
    Chooses a number between 0 and m, inclusive.
    """
    if m >= 0 and m < 1:
        return 0
    elif m >= 1:
        return np.random.randint(0, m+1)
    else:
        raise ValueError(
            "A negative value was given to choose_num: {0}".format(m))

#=============================================================================#

def rand(m, n=1):
    """
    Convenience function to create an m by n numpy matrix with each element
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
    Convenience function to create an m by n numpy matrix with each element
    distributed Discrete Uniform in [low, high]
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
    else:
        raise ValueError("option must be one of the 'v' and 'h'.")

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

def schur(P):
    """g variables in the
    same order as matlab's schur decomp to make it easy to verify my
    translation from matlab to python.
    Simulates matlab's schur decomposition function, returning PU, PD, PV
    s.t. np.dot(PU, np.dot(PD, PV))
    """
    PD, PU = scipy.linalg.schur(P)
    for i in range(len(PD[0, :])):   # make all non-diagonal elements 0.
        for j in range(i+1, len(PD[0, :])):
            PD[i, j] = 0.
            PD[j, i] = 0.
    return PU, PD, PU.T

#=============================================================================#

def svd(P):
    """
    Wrapper for singular value decomposition, returning PU, PD, PV s.t.
    P = np.dot(PU, np.dot(PD, PV)).
    """
    [PU, pd, PV] = scipy.linalg.svd(P)
    PD = scipy.linalg.diagsvd(pd, len(pd), len(pd))
    return PU, PD, PV

#=============================================================================#


def adjust_cond(PU, PD, PV, cond):
    """
    Constructs the matrix PU*PD*PV after adjusting its condition number to be
    cond.

    This will throw an error if PD is a multiple of the identity matrix because
    the condition number in that case is always 1 and this adjustment doesn't
    change that.

    PD also must be a diagonal matrix. If not, raise ValueError.
    """
    if mindiag(PD) == maxdiag(PD):
        raise ValueError(
            "A multiple of the identity matrix can't adjustment condition number.")

    if not _is_diagonal(PD):
        raise ValueError("PD must be a diagonal matrix.")

    size = min(PD.shape)
    ccond = (cond-1)*mindiag(PD)*1./(maxdiag(PD)-mindiag(PD))
    PD = ccond*PD + (1-ccond)*mindiag(PD)*eye(size)
    return reconstruct(PU, PD, PV)

#=============================================================================#

def tweak_diag(MD):
    """
    Takes a square matrix MD, shifts every diagonal element so that the
    smallest diagonal value is 0, and then adds a Uniform(0,1) value to each
    diagonal element.
    """
    m = min(MD.shape)
    return MD - mindiag(MD)*eye(m) + randdiag(m)

#=============================================================================#

def gen_general_obj(n, convex, cond, scale):
    ## Generate the quadratic terms of objective function.
    P = rand(n, n) - rand(n, n)
    P = P+P.T        ## P is symmetric.
    if convex:    ## Convex case.
        PU, PD, PV = schur(P)
        ## Schur decomposition. PD is diagonal since P is symmetric.
        PD = tweak_diag(PD)
        ## PD will be nonnegative.  subtract something to make the smallest
        ## singular value 0, then add (uniform) random numbers to each
    else:    ## Nonconvex case
        PU, PD, PV = svd(P)  ## Singular value decomposition.
    # for both convex and nonconvex, we adjust the condition number,
    # reassemble the matrix, and scale if necessary.
    P = adjust_cond(PU, PD, PV, cond)
    P = (1.*scale/cond)*P   ## Rescale P when cond_P is large.
    return 0.5*(P+P.T)


#=============================================================================#

def sanitize_params(param, qpec_type=-1):
    # Check type value and objective related stuff.
    if param['cond_P'] < 1:
        print 'Warning: cond_P should not be less than 1. \n'
        param['cond_P'] = 20
        print 'cond_P set to 20.\n'
    if not isinstance(param['convex_f'], bool):
        print 'Warning: Wrong data for convex_f.\n'
        param['convex_f'] = True
        print 'convex_f set to True.\n'
    if not isinstance(param['symm_M'], bool):
        print 'Warning: Wrong data for symm_M.\n'
        param['symm_M'] = True
        print 'symm_M set to True.\n'
    if not isinstance(param['mono_M'], bool):
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
    if qpec_type == 100 and param['second_deg'] > param['p']:
        print 'Warning: second_deg should not be greater than p.\n'
        param['second_deg'] = param['p']
        print 'second_deg = p\n'
    elif (qpec_type >= 200 and qpec_type <= 300) and param['second_deg'] > param['m']:
        print 'Warning: second_deg should not be greater than m.\n'
        param['second_deg'] = param['m']
        print 'second_deg set equal to m.\n'
    elif qpec_type >= 800:
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
    if not isinstance(param['yinfirstlevel'], bool):
        print 'Warning: yinfirstlevel should be True or False.\n'
        param['yinfirstlevel'] = True
        print 'yinfirstlevel set to True.\n'

    # Check params controlling output format
    if 'make_with_dbl_comps' not in param:
        param['make_with_dbl_comps'] = False
    elif param['make_with_dbl_comps'] and (qpec_type == 200 or qpec_type == 201):
        print 'Warning: the problem type selected does not result in double comps.\n'
        param['make_with_dbl_comps'] = False
        print 'make_with_dbl_comps set to False.\n'

    return param


def create_name(prefix, size, start=0):
    """ Create a list of names.

    If the prefix is "a" and size is 2, the created name list will be ["a0", "a1"].

    Args:
      prefix: Prefix of name.
      size: Size of names.
      start: If given, the fiest id will be set to it. (Default: 0)

    Returns:
      List of names.
    """
    return [prefix + str(i) for i in range(start, start+size)]
