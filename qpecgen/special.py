from __future__ import absolute_import
from qpecgen.base import QpecgenProblem
from . import helpers


class QpecgenSpecialLCP(QpecgenProblem):

    def __init__(self, param, qpec_type=-1):
        super(QpecgenSpecialLCP, self).__init__(param, qpec_type=qpec_type)
        m = self.param['m']
        n = self.param['n']
        # type 800 is the 'Good LCP-MPEC' and type 900 is the 'Bad LCP-MPEC'
        self.ygen = helpers.zeros(m)
        self.a = helpers.zeros(self.param['l'])
        self.N = helpers.conmat(
            [-helpers.eye(n), helpers.zeros(m - n, n)], option='v')
        self.M = helpers.eye(m)
        self.q = helpers.zeros(m)

        # define initial values.
        self.c = None
        self.d = None
        self.xgen = None

    def return_problem(self):
        problem = {
            'P': self.P,
            'c': self.c,
            'd': self.d,
            'A': self.A,
            'a': self.a,
            'N': self.N,
            'M': self.M,
            'q': self.q}
        optsolxy = helpers.conmat([self.xgen, self.ygen])
        info = {
            'xgen': self.xgen,
            'ygen': self.ygen,
            'optsol': optsolxy,
            'optval': (0.5 * (optsolxy.T) * self.P * optsolxy +
                       helpers.conmat([self.c, self.d]).T * optsolxy)[0, 0]
        }
        return problem, info, self.param


class Qpecgen800(QpecgenSpecialLCP):

    def __init__(self, param):
        super(Qpecgen800, self).__init__(param, qpec_type=800)
        self.c = -2. * helpers.ones(self.n)
        self.d = 4. * helpers.ones(self.m)
        self.xgen = helpers.zeros(self.n)


class Qpecgen900(QpecgenSpecialLCP):

    def __init__(self, param):
        super(Qpecgen900, self).__init__(param, qpec_type=900)
        self.c = 2. * helpers.ones(self.n)
        self.d = -4. * helpers.ones(self.m)
        self.xgen = -helpers.ones(self.n)
