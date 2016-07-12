"""
| Houyuan Jiang, Daniel Ralph, copyright 1997
| Matlab code accompanied the paper:
|   Jiang, H., Ralph D.
|   QPECgen, a MATLAB generator for mathematical programs with
|   Computational Optimization and Applications 13 (1999), 25–59.
|
| Python implementation coded by Patricia Gillett, 2013-2015
| See readme.txt for details about the structures of generated problems.
"""

import scipy
import numpy as np

from helpers import *
from core.qpcc import BasicQPCC

class QpecgenProblem(object):
    def __init__(self, pname, param, qpec_type=-1):
        self.pname = pname

        ## CHECK CONSISTENCY OF PARAMETER DATA ##
        check = True #change to False to not check consistency of data
        if check:
            param = sanitize_params(param, qpec_type=qpec_type)
        self.param = param

        self.type = qpec_type

        # Unpack parameters from param, which is a dictionary.
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
        elif self.type in [800, 900]:
            return 2. * eye(n+m)
        else:
            return gen_general_obj(m+n, convex_f, cond_P, scale_P)

    def make_ctr_A(self):
        l = self.param['l']
        m = self.param['m']
        n = self.param['n']
        if l == 0:
            A = zeros(0, m+n)
        elif self.param['yinfirstlevel']:
            ## yinfirstlevel = True means the upper level yinfirstlevels
            ## involve x and y
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
        if mono_M and symm_M:
            ## Monotone and symmetric case.
            MU, MD, MV = schur(M)
            MD = tweak_diag(MD)
            ## Generate positive diagonal matrix.
            M = adjust_cond(MU, MD, MV, cond_M)

        elif (not mono_M) and symm_M:
            ## Nonmonotone and symmetric case.
            MU, MD, MV = svd(M)
            M = adjust_cond(MU, MD, MV, cond_M)

        elif mono_M and (not symm_M):
            ## Monotone and asymmetric case.
            MU, MD, MV = schur(M)
            ## note that since symm_M=False, MD is not a diagonal matrix
            for i in range(m-1):
                MD[i+1, i] = -MD[i+1, i]
                ## Make real eigenvalues for MD.
            M = reconstruct(MU, MD, MV)
            ## New asymmetric matrix with real eigenvalues.

            MU, MD, MV = schur(M)
            ## Schur decomposition. MD is upper triangular since M has real eigenvalues.
            MD = tweak_diag(MD)
            ## Generate positive diagonal matrix.
            MMU, MMD, MMV = schur(np.dot(MV, MU)) ## MMD is diagonal.
            MM = adjust_cond(MMU, MMD, MMV, cond_M*cond_M)
            MD = (scipy.linalg.cholesky(MM)).T
            ## Use the relation of condition numbers for MD and MD.T*MD
            M = reconstruct(MU, MD, MV)
            ##  Generate a matrix with the required condition number cond_M.

        elif (not mono_M) and (not symm_M):     ## Nonmonotone and asymmetric case.
            MU, MD, MV = svd(M)                   ## Singular value decomposition.
            MD = adjust_cond(MU, MD, MV, cond_M)

        M = (1.*scale_M/cond_M)*M   ## Rescale M when cond_M is large.
        return M

    def return_problem(self):
        raise NotImplementedError

# for qpecgen, we'll add extra class variables varsused, info, param
# note: don't need variable 'type' anymore because we can use isInstance to
# see if a problem inherits from a certain QPECgen type
# note: also need to do an initial add_result with the gen solution

class QpecgenQPCC(BasicQPCC):
    '''
    QpecgenQPCC provides a simple extension of BasicQPCC, adding the instance
    variables varsused, info, and param.
    Initialized by QpecgenQPCC(pname, names, info, param).

    varsused is set using a setter method set_varsused(varsused).

    Elements of the problem (objective, constraints) are managed using the
    BasicQPCC methods.
    '''
    def __init__(self, pname, names, info, param):
        super(QpecgenQPCC, self).__init__(pname, names)
        self.varsused = None
        self.info = info
        self.param = param

    def set_varsused(self, varsused):
        assert len(varsused) == len(self.names)
        self.varsused = varsused
