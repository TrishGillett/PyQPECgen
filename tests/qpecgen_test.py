#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=E0401,E1101
import unittest

import numpy as np

from qpecgen import Qpecgen100, Qpecgen200, Qpecgen201
import qpecgen.helpers as hp


def make_param_inst():
    param_inst = {
        'qpec_type': 200,       # 100, 200, 300, 800, 900.
        'n': 30,                # Dimension of the variable x.
        'm': 20,                # Dimension of the variable y.
        'l': 20,                # Number of the first level constraints.
        'cond_P': 100,          # Condition number of the Hessian P.
        # True or False. Convexity of the objective function.
        'convex_f': True,
        'linobj': False,
        'symm_M': True,         # True or False. Symmetry of the matrix M.
        'mono_M': True,         # True or False. Monotonicity of the matrix M.
        'cond_M': 100,          # Condition number of the matrix M.
        'second_deg': 30,       # Number of the second level degeneracy.
        'first_deg': 0,         # Number of the first level degeneracy.
        'mix_deg': 10,          # Number of mixed degeneracy.
        # Small positive tolerance for measuring degeneracy.
        'tol_deg': 10**(-6),
        'yinfirstlevel': True,  # Whether or not the lower level variables y are
                                # involved in the upper level constraints
        'random': 0,            # Indicates the random 'seed'.
        'make_with_dbl_comps': False
    }
    param_inst.update({
        # Number of the second level constraints
        'p': param_inst['m'],
        # for AVI-MPEC.
        'scale_P': param_inst['cond_P'],   # Scaling constant for the Hessian P.
        'scale_M': param_inst['cond_M']    # Scaling constant for the matrix M.
    })
    return param_inst


class TestHelpers(unittest.TestCase):

    def setUp(self):
        param_inst = make_param_inst()
        self.P200list = []
        for _ in range(5):
            self.P200list.append(Qpecgen200('test', param_inst))
        self.Mlist = [np.random.rand(n, n) for n in [10, 25, 50]]
        self.symmMlist = [0.5 * (M + M.T) for M in self.Mlist]

    def test_min_max_diag(self):
        for M in self.Mlist:
            self.assertEqual(hp.mindiag(M), min(
                M[i, i] for i in range(min(M.shape))))
            self.assertEqual(hp.maxdiag(M), max(
                M[i, i] for i in range(min(M.shape))))

    def test_schur(self):
        for M in self.symmMlist:
            MU, MD, MV = hp.schur(M)
            # ensure that MD is a diagonal matrix
            np.testing.assert_almost_equal(MD, np.diag(np.diag(MD)))
            reconstructedM = hp.reconstruct(MU, MD, MV)
            np.testing.assert_almost_equal(M, reconstructedM)

    def test_svd(self):
        for M in self.Mlist:
            MU, MD, MV = hp.svd(M)
            # ensure that MD is a diagonal matrix
            np.testing.assert_almost_equal(MD, np.diag(np.diag(MD)))
            reconstructedM = hp.reconstruct(MU, MD, MV)
            np.testing.assert_almost_equal(M, reconstructedM)

    def test_adjust_cond(self):
        for M in self.Mlist:
            MU, MD, MV = hp.svd(M)
            M2 = hp.adjust_cond(MU, MD, MV, 20)
            self.assertAlmostEqual(np.linalg.cond(M2), 20)

    @unittest.expectedFailure
    def test_gen_obj(self):
        # A very large matrix generated in a general way is statistically
        # very unlikely to be PSD by accident, so we want to observe that
        # the generator with convex option makes very large PSD matrices
        # and the generator with nonconvex option will nor 'accidentally'
        # make PSD matrices (though there is always some chance of a fluke)

        # Generate a convex Q for 200 variables with condition number 20
        Q = hp.gen_general_obj(200, True, 20, 20)
        self.assertAlmostEqual(np.linalg.cond(Q), 20)
        self.assertTrue(mh.isPSD(Q))
        np.testing.assert_almost_equal(Q, 0.5 * (Q + Q.T))

        # Generate a nonconvex Q for 200 variables with condition number 20
        Q = hp.gen_general_obj(200, False, 20, 20)
        self.assertAlmostEqual(np.linalg.cond(Q), 20)
        self.assertFalse(mh.isPSD(Q))
        np.testing.assert_almost_equal(Q, 0.5 * (Q + Q.T))

        # TODO test monotonicity


class TestBase(unittest.TestCase):

    def setUp(self):
        param_inst = make_param_inst()
        K = 10
        self.Plist = []
        for i in range(K):
            param_inst.update({'convex_f': True})
            self.Plist.append(Qpecgen100(
                'test_100_c_{0}'.format(i), param_inst))
            self.Plist.append(Qpecgen200(
                'test_200_c_{0}'.format(i), param_inst))
            self.Plist.append(Qpecgen201(
                'test_201_c_{0}'.format(i), param_inst))
            param_inst.update({'convex_f': False})
            self.Plist.append(Qpecgen100(
                'test_100_nc_{0}'.format(i), param_inst))
            self.Plist.append(Qpecgen200(
                'test_200_nc_{0}'.format(i), param_inst))
            self.Plist.append(Qpecgen201(
                'test_201_nc_{0}'.format(i), param_inst))
        self.param_inst = param_inst

    @unittest.expectedFailure
    def test_problems_feasible(self):
#        raise Exception()
        for P in self.Plist:
            Q = P.export_QPCC()
            self.assertEqual(len(Q.names), len(Q.details['gensol']))
#            for k in range(len(Q.names)):
#                print "{0} = {1}".format(Q.names[k], Q.details['gensol'][k])
#            print investigate_200_info(P)
            certificates_dict = Q.get_violations(
                Q.details['gensol'], silence=True)
            for key in certificates_dict:
                self.assertLessEqual(
                    len(certificates_dict[key]), 0,
                    "Problem {0} is not feasible: {1}".format(
                        Q.pname, str(certificates_dict)))

    def test_params(self):
        for P in self.Plist:
            param = P.param
            self.assertGreaterEqual(param['cond_P'], 1)
            self.assertGreater(param['scale_P'], 0)
            self.assertGreaterEqual(param['cond_M'], 1)
            self.assertGreater(param['scale_P'], 0)
            self.assertLessEqual(param['first_deg'], param['l'])
    #        if isinstance(P, Qpecgen100):
    #            self.assertLessEqual(param['second_deg'], param['p'])
            if isinstance(P, Qpecgen200):  # or isinstance(P, Qpecgen300):
                self.assertLessEqual(param['second_deg'], param['m'])
            self.assertLessEqual(param['first_deg'], param['l'])
            self.assertLessEqual(param['mix_deg'], param['second_deg'])

    def test_suitable_A(self):
        for P in self.Plist:
            # what size should it have
            self.assertEqual(P.A.shape[0], P.param['l'])
            self.assertEqual(P.A.shape[1], P.param['m'] + P.param['n'])

            # what structure should it have?
            if not P.param['yinfirstlevel']:
                ypart = P.A[:, P.param['n']:]
                np.testing.assert_equal(
                    ypart, hp.zeros(P.param['l'], P.param['m']))

    def test_suitable_M(self):
        for P in self.Plist:
            if P.param['symm_M']:
                np.testing.assert_almost_equal(P.M, P.M.T)
            self.assertAlmostEqual(np.linalg.cond(P.M), P.param['cond_M'])

    def test_suitable_N(self):
        for P in self.Plist:
            # matrix size (m,n)
            self.assertEqual(P.N.shape[0], P.m)
            self.assertEqual(P.N.shape[1], P.n)
            # all elemants in [-1,1]
            self.assertTrue(np.alltrue(P.N >= -1))
            self.assertTrue(np.alltrue(P.N <= 1))

    def gen_200(self):
        P200list = []
        for _ in range(1):
            P200list.append(Qpecgen200('test', self.param_inst))

        return P200list

    def test_info(self):
        for P in self.Plist:
            info = P.info
            param = P.param

            if isinstance(P, Qpecgen100) or isinstance(P, Qpecgen200):
                self.assertEqual(
                    param['l'],
                    info['l_deg'] + info['l_nonactive'] + info['l_active'])

            if isinstance(P, Qpecgen200):
                self.assertLessEqual(param['second_deg'], param['m'])
                for i in range(len(P.info['ygen'])):
                    self.assertGreaterEqual(P.info['ygen'][i], 0)

                # m variables consist of ms vars with 0 <= yi, md vars with 0 <= yi <= u
                # CHECKPOINT 1
                self.assertEqual(P.m, info['ms'] + info['md'])
                self.assertEqual(
                    info['ms'],
                    info['ms_deg'] + info['ms_active'] + info['ms_nonactive'])
                self.assertEqual(
                    info['md'],
                    sum([
                        info['md_upp_deg'], info['md_upp_nonactive'],
                        info['md_low_deg'], info['md_low_nonactive'],
                        info['md_float']]))
                self.assertEqual(
                    param['second_deg'],
                    info['ms_deg'] + info['md_upp_deg'] + info['md_low_deg'])

                self.assertGreaterEqual(info['ms_deg'], 0)
                self.assertGreaterEqual(info['ms_active'], 0)
                self.assertGreaterEqual(info['ms_nonactive'], 0)
                self.assertGreaterEqual(info['md_upp_deg'], 0)
                self.assertGreaterEqual(info['md_upp_nonactive'], 0)
                self.assertGreaterEqual(info['md_low_deg'], 0)
                self.assertGreaterEqual(info['md_low_nonactive'], 0)
                self.assertGreaterEqual(info['md_float'], 0)

            if isinstance(P, Qpecgen201):
                for i in range(len(info['xgen'])):
                    self.assertGreaterEqual(info['xgen'][i], P.info['xl'][i])
                    self.assertLessEqual(info['xgen'][i], P.info['xu'][i])
                self.assertEqual(len(P.u), info['md'])
                for i in range(info['md']):
                    self.assertLessEqual(info['ygen'][i], P.u[i])

    @unittest.expectedFailure
    def test_200_info(self, P):
        '''
        For P which is an instance of Qpecgen200, check various identities
        involving P.info
        '''
        info = P.info
        param = P.param
        self.assertEqual(info['l'], info['l_deg'] +
                         info['l_nonactive'] + info['l_active'])
        self.assertEqual(info['m'], info['ms'] + info['md'])
        self.assertEqual(param['second_deg'], info[
                         'ms_deg'] + info['md_upp_deg'] + info['md_low_deg'])


def investigate_200_info(P):
    cL = [
        'md_upp_deg', 'md_upp_nonactive', 'md_low_deg',
        'md_low_nonactive', 'md_float',
        'ms_deg', 'ms_nonactive', 'ms_active'
    ]
    vL = ['ygen', 'F', 'ulambda', 'pi', 'sigma']

    s = '\n\n================\n'
    # for key in P.info:
    #     try:
    #         s += "{0} = {1}\n".format(key, int(P.info[key]))
    #     except:
    #         pass
    # s += '\n\n================\n'

    s += " ".join([
        'lamL'.ljust(20), 'lamU'.ljust(20), 'u'.ljust(20)
    ] + [x.ljust(20) for x in vL]) + "\n"

    for cat in cL:
        s += '\n{0}={1}\n'.format(cat, P.info[cat])
        for k in range(P.info[cat]):
            if k < P.info['md']:
                lamLi = str(P.info['lamDL'][k])
                lamUi = str(P.info['lamDU'][k])
                ui = str(P.u[k])
            else:
                lamLi = str(P.info['lamS'][k - P.info['md']])
                lamUi = ''
                ui = ''
            s += " ".join([
                lamLi.ljust(20), lamUi.ljust(20), ui.ljust(20)
            ] + [str(P.info[x][k, 0]).ljust(20) for x in vL]) + "\n"

    return s + '================\n\n'


class TestQpecgen(unittest.TestCase):

    def setUp(self):
        self.Plist = [np.random.rand(n, n) for n in [10, 25, 50]]

    @unittest.expectedFailure
    def test_mat_funcs(self):

        for P in self.Plist:
            PU, PD, PV = hp.schur(P)
            # ensure that PD is a diagonal matrix
            self.assertTrue(np.array_equal(PD, np.diag(np.diag(PD))), repr(PD))
            # raise Exception(PU*PD*PV, np.dot(PU, np.dot(PD, PV)))
            self.assertTrue(np.allclose(PU * PD * PV, np.dot(PU, np.dot(PD, PV))))

        for P in self.Plist:
            self.assertEqual(hp.mindiag(P), min(
                P[i, i] for i in range(min(P.shape))))
            self.assertEqual(hp.maxdiag(P), max(
                P[i, i] for i in range(min(P.shape))))


if __name__ == '__main__':
    unittest.main()
