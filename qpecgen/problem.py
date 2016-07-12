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

from . import helpers


class QpecgenProblem(object):

    def __init__(self, param):
        ## CHECK CONSISTENCY OF PARAMETER DATA ##
        check = True  # change to False to not check consistency of data
        if check:
            self.param = helpers.sanitize_params(param)
        else:
            self.param = param

        # Unpack parameters from param, which is a dictionary.
        self.type = self.param['qpec_type']
        self.n = self.param['n']
        self.m = self.param['m']
        # For now, stop controlling the seed because unless implemented
        # differently this would generate every problem with the same seed.
        # np.random.seed = param['random']

        # Upper level objective function coefficients
        self.P = self.make_obj_P()

        # Matrix part of upper level constraints
        self.A = self.make_ctr_A()

        # Lower level objective function coefficients
        self.M = self.make_obj_M()
        self.N = 1. * (helpers.rand(self.m, self.n) -
                       helpers.rand(self.m, self.n))
        # This ends the data which is the same for all problem types

    def get_Px(self):
        return self.P[:self.n, :self.n]

    def get_Py(self):
        return self.P[self.n:self.m + self.n, self.n:self.m + self.n]

    def get_Pxy(self):
        # Pxy[i,j] gives coeff for x[i]y[j]
        # (but don't forget that P is symmetric so these terms
        # are encountered twice when using P)
        return self.P[:self.n, self.n:self.m + self.n]

    def make_obj_P(self):
        m = self.m
        n = self.n
        linobj = self.param['linobj']
        convex_f = self.param['convex_f']
        cond_P = self.param['cond_P']
        scale_P = self.param['scale_P']
        if linobj:
            return helpers.zeros(m + n, m + n)
        elif self.param['qpec_type'] in [800, 900]:
            return 2. * helpers.eye(n + m)
        else:
            # Generate the quadratic terms of objective function.
            P = helpers.rand(m + n, m + n) - helpers.rand(m + n, m + n)
            P = P + P.T  # P is symmetric.
            if convex_f:  # Convex case.

                # Schur decomposition. PT is diagonal since P is symmetric.
                PU, PD, PV = helpers.schur(P)

                # PT will be nonnegative.
                # subtract something to make the smallest singular value 0,
                # then add (uniform) random numbers to each
                PD = helpers.tweak_diag(PD)

            else:  # Nonconvex case

                # Singular value decomposition.
                PU, PD, PV = helpers.svd(P)

            # for both convex and nonconvex, we adjust the condition number,
            # reassemble the matrix, and scale if necessary.
            P = helpers.adjust_cond(PU, PD, PV, cond_P)
            P = (1. * scale_P / cond_P) * P  # Rescale P when cond_P is large.
            return 0.5 * (P + P.T)

    def make_ctr_A(self):
        l = self.param['l']
        m = self.param['m']
        n = self.param['n']
        if l == 0:
            A = helpers.zeros(0, m + n)
        elif self.param['yinfirstlevel']:

            # yinfirstlevel = True means the upper level yinfirstlevels involve
            # x and y
            A = helpers.rand(l, n + m) - helpers.rand(l, n + m)

        else:

            # yinfirstlevel = False means they only use x variables
            A = helpers.conmat(
                [np.matrix(helpers.rand(l, n) - helpers.rand(l, n)),
                 helpers.zeros(l, m)],
                option='h')

        # if A.size[0] > 0:
        #     for i in range(A.size[1]):
        #         A[0,i] = abs(A[0,i])
        return A

    def make_obj_M(self):
        m = self.param['m']
        mono_M = self.param['mono_M']
        symm_M = self.param['symm_M']
        cond_M = self.param['cond_M']
        scale_M = self.param['scale_M']

        M = helpers.rand(m, m) - helpers.rand(m, m)
        # Consider the symmetric property of M.
        if symm_M:
            M = M + M.T

        # Consider the monotonicity property and the condition number of M.
        if mono_M and symm_M:
            # Monotone and symmetric case.

            MU, MD, MV = helpers.schur(M)

            # Generate positive diagonal matrix.
            MD = helpers.tweak_diag(MD)
            M = helpers.adjust_cond(MU, MD, MV, cond_M)

        elif (not mono_M) and symm_M:
            # Nonmonotone and symmetric case.

            MU, MD, MV = helpers.svd(M)
            M = helpers.adjust_cond(MU, MD, MV, cond_M)

        elif mono_M and (not symm_M):
            # Monotone and asymmetric case.

            # note that since symm_M=False, MD is not a diagonal matrix
            MU, MD, MV = helpers.schur(M)
            for i in range(m - 1):
                # Make real eigenvalues for MD.
                MD[i + 1, i] = -MD[i + 1, i]

            # New asymmetric matrix with real eigenvalues.
            M = helpers.reconstruct(MU, MD, MV)

            # Schur decomposition. MD is upper triangular since M has real
            # eigenvalues.
            MU, MD, MV = helpers.schur(M)
            # Generate positive diagonal matrix.
            MD = helpers.tweak_diag(MD)
            # MMD is diagonal.
            MMU, MMD, MMV = helpers.schur(np.dot(MV, MU))
            MM = helpers.adjust_cond(MMU, MMD, MMV, cond_M * cond_M)
            # Use the relation of condition numbers for MD and MD.T*MD
            MD = (scipy.linalg.cholesky(MM)).T
            #  Generate a matrix with the required condition number cond_M.
            M = helpers.reconstruct(MU, MD, MV)

        elif (not mono_M) and (not symm_M):
            # Nonmonotone and asymmetric case.

            # Singular value decomposition.
            MU, MD, MV = helpers.svd(M)
            MD = helpers.adjust_cond(MU, MD, MV, cond_M)

        # Rescale M when cond_M is large.
        M = (1. * scale_M / cond_M) * M
        return M

    def return_problem(self):
        raise NotImplementedError
