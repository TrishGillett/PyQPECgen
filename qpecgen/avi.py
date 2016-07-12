from numpy import sign, array

from core.qpcc import BasicQPCC
from base import QpecgenProblem
from helpers import *

def _indices(slack, tol_deg):
    '''
    Compute index such that:
      index(i)=1  iff slack in (inf, -tol_deg)
      index(i)=0  iff slack in [-tol_deg, tol_deg]
      index(i)=-1 iff slack in (tol_deg, inf)
    '''
    nindex = len(slack)
    sign_indicator = map(sign, slack)
    tol_indicator = [(s <= -tol_deg or s >= tol_deg) for s in slack]
    index = [-(sign_indicator[k]*tol_indicator[k])[0] for k in range(nindex)]
    return index


def _pi_sigma(index, mix_deg):
    """ Generate pi and sigma from index.
    ## Generate the first level multipliers   eta    pi    sigma associated
    ## with other constraints other than the first level constraints
    ## A*[x;y]+a<=0   in the relaxed nonlinear program. In particular,
    ## eta  is associated with  N*x+M*y+q+E^T*lambda=0,
    ## pi                 with  D*x+E*y+b,
    ## sigma              with  lambda.
    """
    p = len(index)
    pi = zeros(p)
    sigma = zeros(p)

    for i in range(p):
        # The first mix_deg ctrs contained in both sets will be degenerate
        if index[i] == 0 and mix_deg > 0:
            pi[i], sigma[i] = 0, 0
            mix_deg = mix_deg - 1
        elif index[i] == 0:
            pi[i], sigma[i] = randcst(), randcst()
        elif index[i] == 1:
            pi[i], sigma[i] = 0, randcst() - randcst()
        elif index[i] == -1:
            pi[i], sigma[i] = randcst() - randcst(), 0
    return pi, sigma


class Qpecgen100(QpecgenProblem):
    def __init__(self, pname, param):
        # QpecgenProblem has param, type, n, m, P, A, M, N
        # Qpecgen100 additionally needs a, D, E, b, E, q, c, d
        # with helper data given by: xgen, ygen, l_nonactive, ulambda, lambd,
        #                            sigma, pi, eta, index
        super(Qpecgen100, self).__init__(pname, param, qpec_type=100)

        self.info = {
            'xgen': rand(self.n) - rand(self.n),
            'ygen': rand(self.m) - rand(self.m),
            # l_nonactive: number ctrs which are not tight at and have lambda=0
            # randomly decide how many of the non-degenerate first level ctrs
            # should be nonactive
            'l_nonactive': choose_num(self.param['l']-self.param['first_deg']),
            ## randomly decide how many of the non-degenerate second level ctrs
            ## should be nonactive
            'p_nonactive': choose_num(self.param['p']-self.param['second_deg'])}
            ## Choose a random number of second level ctrs to be nonactive at
            ## (xgen, ygen)

        self.info.update({
            'l_deg': self.param['first_deg'],
            'l_active': self.param['l']-self.param['first_deg']-self.info['l_nonactive']
        })

        n = param['n']
        m = param['m']
        p = param['p']
        # l: number of first degree ctrs
        l = param['l']

        ####### FIRST LEVEL CTRS A[x;y] + a <= 0
        # Generate the RHS vector and multipliers for the first level ctrs A*[x;y]+a<=0.
        self.a = zeros(l)
        self.make_a_ulambda()

        ###### SECOND LEVEL CTRS Dx + Ey + b <= 0
        self.D = rand(p, n) - rand(p, n)
        self.E = rand(p, m) - rand(p, m)

        self.b = zeros(p)
        self.make_b_lambda()

        N = self.N
        M = self.M
        xgen = self.info['xgen']
        ygen = self.info['ygen']

        ###### STATIONARITY CONDITION FOR LOWER LEVEL PROBLEM
        # Choose q so that Nx + My + E^Tlambda + q = 0 at the solution
        # (xgen, ygen, lambda)
        self.q = -N*xgen - M*ygen - (self.E.T)*self.info['lambda']
        ## KKT conditions of the second level problem.

        ###### For later convenience
        self.info['F'] = npvec(N*xgen + M*ygen + self.q)
        # this must be equal to -E^T\lambda
        self.info['g'] = npvec(self.D*xgen + self.E*ygen + self.b)
        # this is the (negative) amount of slack in the inequalities Dx + Ey + b <= 0

        self.make_pi_sigma_index()
        self.info['eta'] = npvec(scipy.linalg.solve(self.E, self.info['sigma']))

        ###### Generate coefficients of the linear part of the objective
        self.c = zeros(n)
        self.d = zeros(n)
        self.make_c_d()

    def make_a_ulambda(self):
        l_deg = self.info['l_deg']
        l_nonactive = self.info['l_nonactive']
        l_active = self.info['l_active']
        xgen = self.info['xgen']
        ygen = self.info['ygen']

        ####### FIRST LEVEL CTRS A[x;y] + a <= 0
        # Generate the first level multipliers  ulambda  associated with A*[x;y]+a<=0.
        # Generate a so that the constraints Ax+a <= 0 are loose or tight where
        # appropriate.
        self.a = -self.A*conmat([xgen, ygen]) - conmat([zeros(l_deg), # A + a = 0
                                                        rand(l_nonactive), # A + a = 0
                                                        zeros(l_active)]) # A + a <=0

        self.info['ulambda'] = conmat([
            zeros(l_deg),       # degenerate (ctr is tight and ulambda = 0)
            zeros(l_nonactive), # not active (ulambda = 0)
            rand(l_active)])    # active (let ulambda be Uniform(0,1))

    def make_b_lambda(self):
        p = self.param['p']
        second_deg = self.param['second_deg']
        p_nonactive = self.info['p_nonactive']
        # p: number of second degree ctrs (and therefore the number of lambda vars)
        # second_deg: number of second level ctrs for which the ctr is active
        #             AND lambda=0
        # p_nonactive: number of second level ctrs which aren't active.
        #              The corresponding lambdas must therefore be 0

        ## figure out what RHS vector is needed for Dx + Ey + b <= 0
        ## we intentionally build in a gap on the p_nonactive ctrs in the middle
        self.b = -self.D*self.info['xgen']-self.E*self.info['ygen'] - \
                conmat([
                    zeros(second_deg),
                    rand(p_nonactive),
                    zeros(p-second_deg-p_nonactive)])
                    ## The first second_deg constraints

        ## we let the first second_deg cts be degenerate
        ## (ctr is tight and lambda = zero), the next p_nonactive ctrs be not
        ## active (lambda = 0), and the remaining ctrs be active (lambda U(0,1))
        self.info['lambda'] = conmat([zeros(second_deg),
                                      zeros(p_nonactive),
                                      rand(p-second_deg-p_nonactive)])


    def make_pi_sigma_index(self):
        tol_deg = self.param['tol_deg']
        mix_deg = self.param['mix_deg']

        ## Calculate index set at (xgen, ygen)
        slack = array(self.info['lambda']) + array(self.info['g'])
        index = _indices(slack, tol_deg)

        ## Generate the first level multipliers   eta    pi    sigma associated
        ## with other constraints other than the first level constraints
        ## A*[x;y]+a<=0   in the relaxed nonlinear program. In particular,
        ## eta  is associated with  N*x+M*y+q+E^T*lambda=0,
        ## pi                 with  D*x+E*y+b,
        ## sigma              with  lambda.
        pi, sigma = _pi_sigma(index, mix_deg)

        self.info.update({
            'sigma': npvec(sigma),
            'pi': npvec(pi),
            'index': index})

    def make_c_d(self):
        ###### Generate coefficients of the linear part of the objective
        xy = conmat([self.info['xgen'], self.info['ygen']])
        dxP = conmat([self.get_Px(), self.get_Pxy()], option='h')
        dyP = conmat([self.get_Pxy().T, self.get_Py()], option='h')

        ##  Generate c and d such that (xgen, ygen) satisfies KKT conditions
        ##  of AVI-MPEC as well as the first level degeneracy.
        self.c = -(dxP*xy + (self.N.T)*self.info['eta'] + (self.D.T)*self.info['pi'])
        self.d = -(dyP*xy + (self.M.T)*self.info['eta'] + (self.E.T)*self.info['pi'])
        if self.param['l'] > 0:
            Ax, Ay = self.A[:, :self.n].T, self.A[:, self.n:self.m+self.n].T
            self.c += -(Ax)*self.info['ulambda']
            self.d += -(Ay)*self.info['ulambda']

        optsolxy = conmat([self.info['xgen'], self.info['ygen']])
        optsolxyl = npvec(conmat([optsolxy, self.info['lambda']]))
        self.info['optsol'] = optsolxyl,
        self.info['optval'] = (0.5*(optsolxy.T) * self.P * optsolxy + \
                                conmat([self.c, self.d]).T*optsolxy)[0, 0]

    def return_problem(self):
        problem = {
            'P': self.P,
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

    def export_QPCC(self):
        P, info, param = self.return_problem()

        n = param['n']
        m = param['m']
        l = param['l']
        p = len(P['b'])  ## number of g ctrs, number of lambda vars, number of equalities

        names = create_name("x", n) + create_name("y", m) + create_name("L", p)
        Q = BasicQPCC(self.pname, names)

        Q1 = conmat([0.5*P['P'], zeros(n+m, p)], option='h')
        Q2 = conmat([zeros(p, n+m+p)], option='h')
        objQ = conmat([Q1, Q2])
        objp = conmat([P['c'], P['d'], zeros(p, 1)])
        objr = 0
        Q.set_obj(Q=objQ, p=objp, r=objr, mode='min')

        # in order of variables: x variables (n), y variables (m), lambda variables (p)
        A = conmat([P['N'], P['M'], P['E'].T], option='h')
        b = -P['q']
        Q.add_eqs(A, b)

        G1 = conmat([P['A'], zeros(l, p)], option='h')
        G2 = conmat([P['D'], P['E'], zeros(p, p)], option='h')
        G3 = conmat([zeros(p, n+m), -eye(p)], option='h')

        G = conmat([G1, G2, G3])
        h = conmat([-P['a'], -P['b'], zeros(p, 1)])
        Q.add_ineqs(G, h)

        Q.add_comps([[l+i, l+p+i] for i in range(p)])

        varsused = [1]*(n+m) + [0]*p

        assert len(varsused) == len(names)

        gensol = conmat([
            self.info['xgen'],
            self.info['ygen'],
            self.info['lambda']])

        Ax, b = Q.A*gensol, Q.b
        assert np.allclose(Ax, b), "Ax-b=\n{0}".format(Ax-b)

        details = {
            'varsused': varsused,
            'geninfo': info,
            'genparam': param,
            'gensol': gensol}

        Q.add_details(details)

#        print Q
        return Q
