# pylint: disable=unused-variable
from __future__ import absolute_import

from cvxopt import matrix

from qpecgen.base import QpecgenProblem
from qpecgen.helpers import choose_num, rand, conmat, randint, npvec, zeros
from qpecgen.helpers import randcst, create_name, eye


class Qpecgen200(QpecgenProblem):
    '''
    Qpecgen200 generates and manages the data for a qpecgen problem of type 200
    (BOX-QPEC).  In addition to the methods used in problem construction, the
    make_QPCCProblem() method can be used to export the problem as a QpecgenQPCC,
    which is a BasicQPCC object in which some qpecgen specific data is also
    preserved.
    '''

    def __init__(self, pname, param, qpec_type=200):
        super(Qpecgen200, self).__init__(pname, param, qpec_type=qpec_type)

        # Generate xgen
        self.info = {}
        self._make_info()
        self._make_xgen()
        self.u = None  # just initializing
        self._make_ygen()
        self._make_a_ulambda()
        self._make_F_pi_sigma_index()
        self._make_c_d()

    def _make_info(self):
        '''
        Randomly allots statuses to constraints (active, nonactive, degenerate).
        The 'l' series gives the allotment for upper level constraints Ax <= 0.
        The 'ms' series determines what happens with the single-bounded lower
        level variables.  The 'md' series determines what happens with the
        double-bounded lower level variables.

        Since the ordering of variables and constraints within these groups is
        not significant, the generator just determines how many will have a
        given status and then the first l_deg upper level constraints will be
        degenerate, the next l_non will be nonactive, etc.
        '''

        # Decide how many of the non-degenerate first level ctrs should be
        # nonactive
        l = self.param['l']
        l_deg = self.param['first_deg']
        l_nonactive = choose_num(l - l_deg)
        l_active = l - l_deg - l_nonactive
        self.info.update({
            'l': l,
            'l_deg': l_deg,
            'l_nonactive': l_nonactive,
            'l_active': l_active})

        m = self.param['m']
        second_deg = self.param['second_deg']

        md = 1 + choose_num(m - 2)
        ms = m - md

        # single bounded y variables (only bounded below):
        ms_deg = max(choose_num(min(ms, second_deg)),
                     second_deg - m + ms)
        ms_nonactive = choose_num(ms - ms_deg)
        ms_active = ms - ms_deg - ms_nonactive
        self.info.update({
            'm': m,
            'ms': ms,
            'ms_deg': ms_deg,  # F=0, y=0
            'ms_nonactive': ms_nonactive,  # F>0, y=0
            'ms_active': ms_active})  # F=0, y>0

        # divvy up degeneracy numbers so there are second_deg degenerate y vars

        remaining_degen = second_deg - self.info['ms_deg']
        md_upp_deg = choose_num(remaining_degen)  # F=0, y=u
        md_low_deg = remaining_degen - md_upp_deg  # F=0, y=0

        self.info.update({
            'md': md,
            'md_upp_deg': md_upp_deg,
            'md_low_deg': md_low_deg})

        # double bounded y variables (bounded below and above):
        md_nondegen = md - self.info['md_upp_deg'] - self.info['md_low_deg']
        md_upp_non = choose_num(md_nondegen)  # F<0, y=u
        md_low_non = choose_num(md_nondegen - md_upp_non)  # F>0, y=0
        md_float = md_nondegen - md_upp_non - md_low_non  # F=0, 0<y<u
        self.info.update({
            'md_upp_nonactive': md_upp_non,
            'md_low_nonactive': md_low_non,
            'md_float': md_float})

        info = self.info
        param = self.param
        assert self.m == info['ms'] + info['md']
        assert info['ms'] == info['ms_deg'] + \
            info['ms_active'] + info['ms_nonactive']
        assert info['md'] == info['md_upp_deg'] + info['md_upp_nonactive'] + \
            info['md_low_deg'] + info['md_low_nonactive'] + \
            info['md_float']
        assert param['second_deg'] == info['ms_deg'] + info['md_upp_deg'] + \
            info['md_low_deg']

        assert info['ms_deg'] >= 0
        assert info['ms_active'] >= 0
        assert info['ms_nonactive'] >= 0
        assert info['md_upp_deg'] >= 0
        assert info['md_upp_nonactive'] >= 0
        assert info['md_low_deg'] >= 0
        assert info['md_low_nonactive'] >= 0
        assert info['md_float'] >= 0

    def _make_xgen(self):
        self.info['xgen'] = 10 * (rand(self.n) - rand(self.n))

    def _make_u(self):
        self.u = 10. * rand(self.info['md'])

    def _make_ygen(self):
        self._make_u()

        num_double_at_upper = self.info[
            'md_upp_deg'] + self.info['md_upp_nonactive']
        num_not_floating = self.info['md'] - self.info['md_float']

        double_at_upper = npvec(self.u[:num_double_at_upper])
        double_at_lower = zeros(
            self.info['md_low_deg'] + self.info['md_low_nonactive'])
        double_floating = npvec(
            [randcst() * x for x in self.u[num_not_floating:]])

        single_at_lower = zeros(
            self.info['ms_deg'] + self.info['ms_nonactive'])
        single_floating = rand(self.info['ms_active'])

# ygen = conmat([npvec(u[:m_upp_deg+upp_nonactive]),            # double_at_upper
#                     zeros(m_low_deg+low_nonactive),          # double_at_lower
#                     v2,                                      # double_floating
#                     zeros(m_inf_deg+inf_nonactive),          # single_at_lower
# rand(m_inf-m_inf_deg-inf_nonactive)])    # single_floating
        self.info['ygen'] = conmat([
            double_at_upper,  # y=u cases
            double_at_lower,  # y=0 cases
            double_floating,  # 0<y<u cases
            single_at_lower,  # y=0 cases
            single_floating])  # y>0 cases

        for yi in self.info['ygen']:
            assert yi >= 0
        for i in range(self.info['md']):
            assert self.info['ygen'][i] <= self.u[i]

    def _make_a_ulambda(self):
        xgen = self.info['xgen']
        ygen = self.info['ygen']

        # FIRST LEVEL CTRS A[x;y] + a <= 0
        # Generate the first level multipliers  ulambda  associated with A*[x;y]+a<=0.
        # Generate a so the constraints Ax+a <= 0 are loose or tight where
        # appropriate.
        Axy = self.A * conmat([xgen, ygen])
        self.a = -Axy - conmat([
            zeros(self.info['l_deg']),  # A + a = 0
            rand(self.info['l_nonactive']),  # A + a = 0
            zeros(self.info['l_active'])])  # A + a <=0

        self.info['ulambda'] = conmat([
            zeros(self.info['l_deg']),
            zeros(self.info['l_nonactive']),
            rand(self.info['l_active'])])

    def _make_F_pi_sigma_index(self):
        N = self.N
        M = self.M
        u = self.u
        xgen = self.info['xgen']
        ygen = self.info['ygen']

        m = self.param['m']

        # Design q so that Nx + My + E^Tlambda + q = 0 at the solution (xgen,
        # ygen)

        q = -N * xgen - M * ygen
        q += conmat([
            # double bounded, degenerate at upper
            zeros(self.info['md_upp_deg']),
            # double bounded, nonactive at upper
            -rand(self.info['md_upp_nonactive']),
            # double bounded, degenerate at lower
            zeros(self.info['md_low_deg']),
            # double bounded, nonactive at lower
            rand(self.info['md_low_nonactive']),
            zeros(self.info['md_float']),         # double bounded, floating
            # single bounded, degenerate at lower
            zeros(self.info['ms_deg']),
            # single bounded, nonactive at lower
            rand(self.info['ms_nonactive']),
            zeros(self.info['ms_active'])])       # single bounded, floating

        #########################################
        ##        For later convenience        ##
        #########################################
        F = N * xgen + M * ygen + q

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
        md = self.info['md']

        for i in range(md):
            assert ygen[i] >= -tol_deg and ygen[i] < u[i] + tol_deg, \
                "{0} not in [0, {1}]".format(ygen[i], u[i])
            if abs(F[i]) <= tol_deg and ygen[i] > tol_deg and ygen[i] + tol_deg < u[i]:
                index.append(1)  # For the index set alpha.
            elif abs(F[i]) <= tol_deg and abs(ygen[i] - u[i]) <= tol_deg:
                index.append(2)  # For the index set beta_upp.
            elif abs(F[i]) <= tol_deg and abs(ygen[i]) <= tol_deg:
                index.append(3)  # For the index set beta_low.
            elif F[i] < -tol_deg and abs(ygen[i] - u[i]) <= tol_deg:
                index.append(-1)  # For the index set gamma_upp.
            elif F[i] > tol_deg and abs(ygen[i]) <= tol_deg:
                index.append(-1)  # For the index set gamma_low.
            else:
                raise Exception(("didn't know what to do with this case: "
                                 "ygen={0}, u[i] = {1}, F[i]={2}").format(
                                     ygen[i], u[i], F[i]))

        for i in range(md, m):
            if ygen[i] > F[i] + tol_deg:
                index.append(1)  # For the index set alpha.
            elif abs(ygen[i] - F[i]) <= tol_deg:
                index.append(4)  # For the index set beta_inf.
            else:
                index.append(-1)  # For the index set gamma_inf.

        # Generate the first level multipliers   pi    sigma
        # associated with other constraints other than the first level constraints
        # A*[x;y]+a<=0   in the relaxed nonlinear program. In particular,
        # pi            is associated with  F(x, y)=N*x+M*y+q, and
        # sigma                       with  y.
        mix_upp_deg = max(
            mix_deg - self.info['md_low_deg'] - self.info['ms_deg'],
            choose_num(self.info['md_upp_deg']))
        mix_upp_deg = min(mix_upp_deg, mix_deg)

        mix_low_deg = max(
            mix_deg - mix_upp_deg - self.info['ms_deg'],
            choose_num(self.info['md_low_deg']))
        mix_low_deg = min(mix_low_deg, mix_deg - mix_upp_deg)

        mix_inf_deg = mix_deg - mix_upp_deg - mix_low_deg
        mix_inf_deg = min(mix_inf_deg, mix_deg - mix_upp_deg - mix_inf_deg)

        assert mix_deg >= 0
        assert mix_upp_deg >= 0
        assert mix_low_deg >= 0
        assert mix_inf_deg >= 0

#        assert self.param['second_deg'] == self.info['ms_deg'] +
#               self.info['md_low_deg'] + self.info['md_upp_deg'] + mix_deg
        k_mix_inf = 0
        k_mix_upp = 0
        k_mix_low = 0
        pi = zeros(m, 1)
        sigma = zeros(m, 1)
        for i in range(m):
            if index[i] == 1:
                pi[i] = randcst() - randcst()
                sigma[i] = 0
            elif index[i] == 2:
                if k_mix_upp < mix_upp_deg:
                    pi[i] = 0
                    # The first mix_upp_deg constraints associated with F(i)<=0
                    # in the set beta_upp are degenerate.
                    sigma[i] = 0
                    # The first mix_upp_deg constraints associated with
                    # y(i)<=u(i) in the set beta_upp are degenerate.
                    k_mix_upp = k_mix_upp + 1
                else:
                    pi[i] = randcst()
                    sigma[i] = randcst()
            elif index[i] == 3:
                if k_mix_low < mix_low_deg:
                    pi[i] = 0
                    # The first mix_low_deg constraints associated with F(i)>=0
                    # in the set beta_low are degenerate.
                    sigma[i] = 0
                    # The first mix_low_deg constraints associated with
                    # y(i)>=0 in the set beta_low are degenerate.
                    k_mix_low = k_mix_low + 1
                else:
                    pi[i] = -randcst()
                    sigma[i] = -randcst()
            elif index[i] == 4:
                if k_mix_inf < mix_inf_deg:
                    pi[i] = 0
                    # The first mix_inf_deg constraints associated with F(i)>=0
                    # in the set beta_inf are degenerate.
                    sigma[i] = 0
                    # The first mix_inf_deg constraints associated with
                    # y(i)>=0 in the set beta_inf are degenerate.
                    k_mix_inf = k_mix_inf + 1
                else:
                    pi[i] = -randcst()
                    sigma[i] = -randcst()
            else:
                pi[i] = 0
                sigma[i] = randcst() - randcst()

        self.q = q
        self.info.update({
            'F': F,
            'mix_upp_deg': mix_upp_deg,
            'mix_low_deg': mix_low_deg,
            'mix_inf_deg': mix_inf_deg,
            'pi': pi,
            'sigma': sigma})

    def _make_c_d(self):
        n = self.param['n']
        m = self.param['m']
        # Generate coefficients of the linear part of the objective
        # Generate c and d such that (xgen, ygen) satisfies KKT conditions
        # of AVI-MPEC as well as the first level degeneracy.
        Px = self.get_Px()
        Pxy = self.get_Pxy()
        Py = self.get_Py()
        xgen = self.info['xgen']
        ygen = self.info['ygen']
        pi = self.info['pi']
        sigma = self.info['sigma']

        self.c = -(Px * xgen + Pxy * ygen + self.N.T * pi)
        self.d = -(Pxy.T * xgen + Py * ygen + self.M.T * pi + sigma)
        if self.param['l'] > 0:
            self.c += -(self.A[:, :n].T * self.info['ulambda'])
            self.d += -(self.A[:, n:m + n].T * self.info['ulambda'])
        # ELEPHANT reinstate later
#        self.info['optval'] = (0.5*(self.info['optsol'].T)*self.P*self.info['optsol']
#                          +conmat([self.c, self.d]).T*self.info['optsol'])[0,0]

    def make_QPCC_sol(self):
        lamDL, lamS, lamDU = self.get_dual_vals(
            self.info['xgen'], self.info['ygen'])

        self.info.update({
            'lamDL': lamDL,
            'lamS': lamS,
            'lamDU': lamDU})

        gensol = conmat([
            self.info['xgen'],
            self.info['ygen'],
            lamDL,
            lamS,
            lamDU])
        return gensol

    def get_dual_vals(self, x, y):
        """
        Computes the values of the lower level problem's dual variables
        vectors at the given solution (x, y).

        Args:
           x, y: an optimal solution to the QPEC.

        Returns:
           :math:`\lambda_D^L`: vector of dual variable values for the
           constraints :math:`y_D \geq 0`

           :math:`\lambda_S`: vector of dual variable values for the
           constraints :math:`y_S \geq 0`

           :math:`\lambda_D^U`: vector of dual variable values for the
           constraints :math:`y_D \leq y_u`
        """

        # computing the full sol at xgen, ygen
        md = self.info['md']
        ms = self.info['ms']
        lamDL = zeros(md)
        lamDU = zeros(md)
        lamS = zeros(ms)
        ETlambda = -self.N * x - self.M * y - self.q

        assert ms + md == len(ETlambda)
        for i in range(md):
            if ETlambda[i] >= 0:
                lamDL[i] = 0.
                lamDU[i] = ETlambda[i]
            else:
                lamDL[i] = -ETlambda[i]
                lamDU[i] = 0.
#            assert lamDL[i] - lamDU[i] == ETlambda[i]

#        print lamS
        for i in range(ms):
            lamS[i] = -ETlambda[md + i]
#            assert lamS[i] == ETlambda[md+i]
#            assert lamS[i] >= 0, "if this goes wrong it's because this part of
#                   q wasn't generated right!"
#        E1 = conmat([eye(md), zeros(md, ms), -eye(md)], option='h')
#        E2 = conmat([zeros(ms, md), eye(ms), zeros(ms, md)], option='h')
#        lam = conmat([lamDL, lamS, lamDU])
#        ETlambda1 = conmat([E1, E2])*lam
#        assert np.allclose(ETlambda1, ETlambda), "{0} {1}".format(ETlambda1, ETlambda)
        return lamDL, lamS, lamDU

    def return_problem(self):
        """
        Args:
           (None)

        Returns:
           problem: a dictionary with keys ``P``, ``c``, ``d``, ``A``, ``a``,
           ``u``, ``N``, ``M``, ``q`` defining the
        """

        problem = {
            'P': self.P,
            'c': self.c,
            'd': self.d,
            'A': self.A,
            'a': self.a,
            'u': self.u,
            'N': self.N,
            'M': self.M,
            'q': self.q}
        return problem, self.info, self.param

    def export_QPCC_data(self):
        P, info, param = self.return_problem()
        n = param['n']
        m = param['m']
        md = info['md']
        ms = info['ms']
        l = param['l']

        varsused = [1] * (n + m) + [0] * (m + md)
        names = create_name("x", n) + create_name("y", m) + \
            create_name("lamL", md) + create_name("lamL", ms, start=md) + \
            create_name("lamU", md)

        objQ = matrix([
            [matrix(0.5 * P['P']), matrix(zeros(m + md, m + n))],
            [matrix(zeros(m + n, m + md)), matrix(zeros(m + md, m + md))]])
        objp = conmat([P['c'], P['d'], zeros(m + md)])

        objr = 0

        G1 = conmat([P['A'], zeros(l, m + md)], option='h')
        h1 = -P['a']

        G2 = conmat([zeros(md, n), -eye(md), zeros(md, 2 * m)], option='h')
        h2 = zeros(md)

        G3 = conmat([zeros(md, n), eye(md), zeros(md, 2 * m)], option='h')
        h3 = P['u']

        G4 = conmat([zeros(ms, n + md), -eye(ms),
                     zeros(ms, m + md)], option='h')
        h4 = zeros(ms)

        G5 = conmat([zeros(m, n + m), -eye(m), zeros(m, md)], option='h')
        h5 = zeros(m)

        G6 = conmat([zeros(md, n + 2 * m), -eye(md)], option='h')
        h6 = zeros(md)

        if isinstance(self, Qpecgen201):
            G7 = conmat([-eye(n), zeros(n, 2 * m + md)], option='h')
            h7 = -self.info['xl']

            G8 = conmat([eye(n), zeros(n, 2 * m + md)], option='h')
            h8 = self.info['xu']

        A1 = conmat(
            [P['N'][:md], P['M'][:md], -eye(md), zeros(md, ms), eye(md)], option='h')
        b1 = -P['q'][:md]

        A2 = conmat(
            [P['N'][md:], P['M'][md:], zeros(ms, md), -eye(ms), zeros(ms, md)],
            option='h')
        b2 = -P['q'][md:]

        details = {
            'varsused': varsused,
            'geninfo': info,
            'genparam': param,
            'gensol': self.make_QPCC_sol()}

        return locals()


class Qpecgen201(Qpecgen200):
    """
    This subclass of ``qpecgen.box.Qpecgen200`` generates a more specific type
    of BOX-QPEC problem known as the FULL-BOX-QPEC.  Type 201 is a more
    specific case of type 200 where x variables are constrained
    :math:`x_l \leq x \leq x_u` and y variables are constrained
    :math:`0 \leq y \leq y_u \leq 10` for integers :math:`x_l \in [-10, 0]`,
    :math:`x_u \in [1, 10]`, :math:`y_u \in [1, 10]`.

    Some class methods (not shown here due to private status) are overridden
    for this class so that problems are generated with the full box structure.
    Methods for
    """

    def __init__(self, pname, param):
        super(Qpecgen201, self).__init__(pname, param, qpec_type=201)

    def _make_info(self):
        md = self.param['m']
        l = self.param['l']
        second_deg = self.param['second_deg']

        # Note that we only consider the following two cases of box constraints:
        # y(i) in [0, +inf) or  [0, u] where u is a nonnegative scalar.
        # Clearly, any other interval can be obtained by using the mapping
        # y <--- c1+c2*y.
        # It is assumed that the last m_inf constraints are of the form [0, inf)
        # The remaining m_inf - m_inf_deg - inf_nonactive constraints are where
        # F=0, y>0

        # y variables which are bounded below and above
        # There will be m - m_inf variables with double sided bounds. each upper
        # bound is chosen uniform in [0,10]
        md_upp_deg = choose_num(second_deg)
        # degenerate with y at upper bound: F=0, y=u
        md_low_deg = second_deg - md_upp_deg
        # degenerate with y at lower bound: F=0, y=0
        md_upp_nonactive = choose_num(md - md_upp_deg - md_low_deg)
        # F not active, y at upper bound: F<0, y=u
        md_low_nonactive = choose_num(
            md - md_upp_deg - md_low_deg - md_upp_nonactive)
        # F not active, y at lower bound: F>0, y=0

        l_deg = self.param['first_deg']
        l_nonactive = choose_num(l - l_deg)
        self.info.update({
            'ms': 0,
            'ms_deg': 0,
            'ms_nonactive': 0,
            'ms_active': 0,
            'md': md,
            'md_upp_deg': md_upp_deg,
            'md_low_deg': md_low_deg,
            'md_upp_nonactive': md_upp_nonactive,
            'md_low_nonactive': md_low_nonactive,
            'md_float': (
                md - md_upp_deg - md_low_deg - md_upp_nonactive - md_low_nonactive),
            # Randomly decide how many of the non-degenerate first level ctrs
            # should be nonactive
            'l': l,
            'l_deg': l_deg,
            'l_nonactive': l_nonactive,
            'l_active': l - l_deg - l_nonactive})

    def _make_xgen(self):
        xl = randint(-10, 0, self.param['n'])
        xu = randint(1, 10, self.param['n'])
        self.info['xl'] = npvec(xl)
        self.info['xu'] = npvec(xu)
        self.info['xgen'] = npvec(
            [(xl[i] + (xu[i] - xl[i]) * randcst())[0] for i in range(self.param['n'])])
#        raise Exception(xl, xu, self.info['xgen'])

    def _make_u(self):
        self.u = randint(0, 10, self.info['md'])
