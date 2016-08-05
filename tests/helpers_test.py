#! /usr/bin/env python
# pylint: disable=E1101,E0401,too-many-public-methods,relative-import
""" Unit tests for helpers module.
"""
import functools
import random
import unittest

import numpy as np
import numpy.linalg
import numpy.random

from qpecgen import helpers


NUMBER_OF_TRIALS = 10000
""" Number of trials for tests generating random values. """


class TestChooseNum(unittest.TestCase):
    """ Unittest for choose_num.
    """

    def test_zero(self):
        """ Test giving 0.
        """
        self.assertEqual(helpers.choose_num(0), 0)

    def test_between_zero_and_one(self):
        """ Test giving a number between 0 and 1.
        """
        self.assertEqual(helpers.choose_num(0.5), 0)

    def test_greater_one(self):
        """ Test giving a number greater than 1.

        This test runs choose_num NUMBER_OF_TRIALS times.
        """
        m = 10
        for _ in xrange(NUMBER_OF_TRIALS):
            res = helpers.choose_num(m)
            self.assertGreaterEqual(res, 0)
            self.assertLessEqual(res, m)

    def test_negative(self):
        """ Test giving negative number.
        """
        self.assertRaises(ValueError, helpers.choose_num, -5)

    def test_ransom_seed(self):
        """Test setting random seeds.
        """
        np.random.seed(12345)
        value = helpers.choose_num(10)
        self.assertNotEqual(value, helpers.choose_num(10))

        np.random.seed(12345)
        self.assertEqual(value, helpers.choose_num(10))

        np.random.seed(12346)
        self.assertNotEqual(value, helpers.choose_num(10))


class TestRand(unittest.TestCase):
    """ Unittest for rand.
    """

    def evaluate(self, shape, **kwargs):
        """ Evaluate rand.

        Args:
          shape: Shape of result matrix of randint.
          kwargs: To be passed to randint.
        """
        m = helpers.rand(**kwargs)
        self.assertEqual(m.shape, shape)
        self.assertGreaterEqual(m.min(), 0)
        self.assertLessEqual(m.max(), 1)

    def test_one(self):
        """ Test for 1 by 1 matrix.

        This test runs choose_num NUMBER_OF_TRIALS times.
        """
        for _ in range(NUMBER_OF_TRIALS):
            self.evaluate((1, 1), m=1)

    def test_two_by_one(self):
        """ Test for 2 by 1 matrix.

        This test runs choose_num NUMBER_OF_TRIALS times.
        """
        for _ in range(NUMBER_OF_TRIALS):
            self.evaluate((2, 1), m=2)

    def test_two_by_two(self):
        """ Test for 2 by 2 matrix.

        This test runs choose_num NUMBER_OF_TRIALS times.
        """
        for _ in range(NUMBER_OF_TRIALS):
            self.evaluate((2, 2), m=2, n=2)

    def test_ransom_seed(self):
        """Test setting random seeds.
        """
        np.random.seed(12345)
        value = helpers.rand(2, 2)
        self.assertFalse(np.allclose(value, helpers.rand(2, 2)))

        np.random.seed(12345)
        self.assertTrue(np.allclose(value, helpers.rand(2, 2)))

        np.random.seed(12346)
        self.assertFalse(np.allclose(value, helpers.rand(2, 2)))


class TestRandcst(unittest.TestCase):
    """ Unittest for randcst.
    """

    def test(self):
        """ Check outputs.

        This test runs choose_num NUMBER_OF_TRIALS times.
        """
        for _ in range(NUMBER_OF_TRIALS):
            m = helpers.randcst()
            self.assertGreaterEqual(m, 0)
            self.assertLessEqual(m, 1)

    def test_ransom_seed(self):
        """Test setting random seeds.
        """
        np.random.seed(12345)
        value = helpers.randcst()
        self.assertNotEqual(value, helpers.randcst())

        np.random.seed(12345)
        self.assertEqual(value, helpers.randcst())

        np.random.seed(12346)
        self.assertNotEqual(value, helpers.randcst())


class TestRandint(unittest.TestCase):
    """ Unittest for randind.
    """

    def setUp(self):
        """ Prepare low and high variables.
        """
        self.low = random.randint(0, 1024)
        self.high = self.low + random.randint(0, 1024)

    def evaluate(self, shape, **kwargs):
        """ Evaluate randint.

        Args:
          shape: Shape of result matrix of randint.
          kwargs: To be passed to randint.
        """
        m = helpers.randint(low=self.low, high=self.high, **kwargs)
        self.assertEqual(m.shape, shape)
        self.assertGreaterEqual(m.min(), self.low)
        self.assertLessEqual(m.max(), self.high)

    def test_one(self):
        """ Test for 1 by 1 matrix.

        This test runs choose_num NUMBER_OF_TRIALS times.
        """
        for _ in range(NUMBER_OF_TRIALS):
            self.evaluate((1, 1), m=1)

    def test_two_by_one(self):
        """ Test for 2 by 1 matrix.

        This test runs choose_num NUMBER_OF_TRIALS times.
        """
        for _ in range(NUMBER_OF_TRIALS):
            self.evaluate((2, 1), m=2)

    def test_two_by_two(self):
        """ Test for 2 by 2 matrix.

        This test runs choose_num NUMBER_OF_TRIALS times.
        """
        for _ in range(NUMBER_OF_TRIALS):
            self.evaluate((2, 2), m=2, n=2)

    def test_ransom_seed(self):
        """Test setting random seeds.
        """
        np.random.seed(12345)
        value = helpers.randint(10, 256, 2, 2)
        self.assertFalse(np.allclose(value, helpers.randint(10, 256, 2, 2)))

        np.random.seed(12345)
        self.assertTrue(np.allclose(value, helpers.randint(10, 256, 2, 2)))

        np.random.seed(12346)
        self.assertFalse(np.allclose(value, helpers.randint(10, 256, 2, 2)))


class TestZeros(unittest.TestCase):
    """ Unittest for zeros.
    """

    def evaluate(self, shape, **kwargs):
        """ Evaluate zeros.

        Args:
          shape: Shape of result matrix of randint.
          kwargs: To be passed to randint.
        """
        m = helpers.zeros(**kwargs)
        self.assertEqual(m.shape, shape)
        self.assertGreaterEqual(m.min(), 0)
        self.assertLessEqual(m.max(), 0)

    def test_one(self):
        """ Test for 1 by 1 matrix.
        """
        self.evaluate((1, 1), m=1)

    def test_two_by_one(self):
        """ Test for 2 by 1 matrix.
        """
        self.evaluate((2, 1), m=2)

    def test_two_by_two(self):
        """ Test for 2 by 2 matrix.
        """
        self.evaluate((2, 2), m=2, n=2)


class TestOnes(TestZeros):
    """ Unittest for ones.
    """

    def evaluate(self, shape, **kwargs):
        """ Evaluate ones.

        Args:
          shape: Shape of result matrix of randint.
          kwargs: To be passed to randint.
        """
        m = helpers.ones(**kwargs)
        self.assertEqual(m.shape, shape)
        self.assertGreaterEqual(m.min(), 1)
        self.assertLessEqual(m.max(), 1)


class TestEye(unittest.TestCase):
    """ Unittest for eye.
    """

    def test_one(self):
        """ Test for 1 by 1 matrix.
        """
        m = helpers.eye(1)
        v = helpers.rand(1)
        self.assertTrue(np.array_equal(m.dot(v), v))
        self.assertTrue(np.array_equal(v.dot(m), v))

    def test_two(self):
        """ Test for 2 by 2 matrix.
        """
        m = helpers.eye(2)
        v = helpers.rand(2, 2)
        self.assertTrue(np.array_equal(m.dot(v), v))
        self.assertTrue(np.array_equal(v.dot(m), v))


class TestConmat(unittest.TestCase):
    """ Unittest for conmat.
    """

    def setUp(self):
        """ Prepare tests.
        """
        u = helpers.rand(3, 3)
        v = helpers.rand(3, 3)
        self.tup = (u, v)

    def test_v(self):
        """ Test option v.
        """
        res = helpers.conmat(self.tup, "v")
        ans = np.matrix(np.vstack(self.tup))
        self.assertTrue(np.array_equal(res, ans))

    def test_h(self):
        """ Test option h.
        """
        res = helpers.conmat(self.tup, "h")
        ans = np.matrix(np.hstack(self.tup))
        self.assertTrue(np.array_equal(res, ans))

    def test_others(self):
        """ Test an invalid option.
        """
        self.assertRaises(ValueError, helpers.conmat, self.tup, ":3")


class TestNpvec(unittest.TestCase):
    """ Unittest for npvec.
    """

    def test(self):
        """ Test with a simple input.
        """
        size = 10
        vec = [x for x in range(size)]
        res = helpers.npvec(vec)
        self.assertEqual(res.shape, (size, 1))


class TestMindiag(unittest.TestCase):
    """ Unittest for mindiag.
    """

    def test(self):
        """ Test with a simple input.
        """
        m = helpers.rand(100, 100)
        ans = m.diagonal().min()
        self.assertEqual(helpers.mindiag(m), ans)


class TestMaxdiag(unittest.TestCase):
    """ Unittest for maxdiag.
    """

    def test(self):
        """ Test with a simple input.
        """
        m = helpers.rand(100, 100)
        ans = m.diagonal().max()
        self.assertEqual(helpers.maxdiag(m), ans)


class TestRanddiag(unittest.TestCase):
    """ Unittest for randdiag.
    """

    def test_scalar(self):
        """ Test for a single value i.e. n = 1.
        """
        m = helpers.randdiag(1)
        self.assertEqual(m.shape, (1, 1))
        self.assertGreaterEqual(m[0, 0], 0)
        self.assertLessEqual(m[0, 0], 1)

    def test_matrix(self):
        """ Test for a matrix i.e. n > 1.
        """
        size = 100
        m = helpers.randdiag(size)
        self.assertEqual(m.shape, (size, size))
        for i in range(size):
            for j in range(size):
                if i == j:
                    self.assertGreaterEqual(m[i, j], 0)
                    self.assertLessEqual(m[i, j], 1)
                else:
                    self.assertEqual(m[i, j], 0)


class TestReconstruct(unittest.TestCase):
    """ Unittest for reconstruct.
    """

    def test_scalar(self):
        """ Test with simple inputs.
        """
        self.evaluate(1)

    def test_matrix(self):
        """ Test for a matrix i.e. n > 1.
        """
        self.evaluate(2)

    def evaluate(self, size):
        """ Evaluate with a give size matrix.

        Args:
          size: Matrix size.
        """
        ms = [helpers.rand(size, size) for _ in range(3)]
        ans = functools.reduce(lambda u, v: u.dot(v), ms)
        self.assertTrue(np.allclose(helpers.reconstruct(*ms), ans))


class TestSchur(unittest.TestCase):
    """ Unittest for schur.
    """

    def evaluate(self, size):
        """ Evaluate with a give size matrix.

        Args:
          size: Matrix size.
        """
        m = helpers.rand(size, size)
        PU, PD, PV = helpers.schur(m)
        self.assertTrue(np.allclose(np.dot(PU, np.dot(PD, PV)), m))

    def test_one(self):
        """ Test with a 1 by 1 matrix.
        """
        self.evaluate(1)

    @unittest.skip("Issue #66 is not solved.")
    def test_two(self):
        """ Test with a 2 by 2 matrix.
        """
        self.evaluate(2)

    @unittest.skip("Issue #66 is not solved.")
    def test_100(self):
        """ Test with a 100 by 100 matrix.
        """
        self.evaluate(100)


class TestSvd(TestSchur):
    """ Unittest for svd.
    """

    def evaluate(self, size):
        """ Evaluate with a give size matrix.

        Args:
          size: Matrix size.
        """
        m = helpers.rand(size, size)
        PU, PD, PV = helpers.svd(m)

        self.assertTrue(np.allclose(np.dot(PU, np.dot(PD, PV)), m))
        for i in range(size):
            for j in range(size):
                if i != j:
                    self.assertEqual(PD[i, j], 0)


class TestAdjustCond(unittest.TestCase):
    """ Unittest for adjust_cond.
    """

    def test(self):
        """ Test for matrixes of which size are 2 to 100.

        This method checks;

          1. minimum diagonal value of PD is not changed,
          2. all diagonal values are different,
          3. condition number of output PD is same as the desired value.
        """
        for size in range(2, 101):
            PU = PV = np.identity(size)
            while True:
                PD = helpers.randdiag(size)
                if helpers.mindiag(PD) != helpers.maxdiag(PD):
                    break
            cond = 73

            res = helpers.adjust_cond(PU, PD, PV, cond)
            self.assertAlmostEqual(helpers.mindiag(res), helpers.mindiag(PD))
            self.assertNotEqual(helpers.mindiag(res), helpers.maxdiag(res))
            self.assertAlmostEqual(numpy.linalg.cond(res), cond)

    def test_identity_matrix(self):
        """ Test for multiple of identity matrix and raise errors.
        """
        for size in range(2, 101):
            PU = PV = np.identity(size)
            PD = random.randint(1, 1024) * np.identity(size)
            cond = 73
            self.assertRaises(
                ValueError, helpers.adjust_cond, PU, PD, PV, cond)

    def test_non_diagonal_matrix(self):
        """ Test for multiple of non diagonal matrix and raise errors.
        """
        for size in range(2, 101):
            PU = PV = np.identity(size)
            PD = helpers.rand(size)
            cond = 73
            self.assertRaises(
                ValueError, helpers.adjust_cond, PU, PD, PV, cond)


class TestTweakDiag(unittest.TestCase):
    """ Unittest for tweak_diag.
    """

    def test(self):
        """ Test with a 100 by 100 matrix.
        """
        size = 100
        m = helpers.rand(size, size)
        min_diagonal = m.diagonal().min()
        res = helpers.tweak_diag(m)

        for i in range(size):
            self.assertGreaterEqual(res[i, i], m[i, i] - min_diagonal)
            self.assertLessEqual(res[i, i], m[i, i] - min_diagonal + 1)


class TestGenGeneralObj(unittest.TestCase):
    """ Unittest for gen_general_obj.
    """

    def test_convex(self):
        """ Test for convex cases.
        """
        for size in range(2, 101):
            cond = random.randint(1, 100)

            numpy.random.seed(cond)
            res = helpers.gen_general_obj(size, True, cond, 1)

            self.assertEqual(res.shape, (size, size))
            self.assertGreaterEqual(helpers.mindiag(res), 0)
            self.assertGreater(helpers.maxdiag(res), 0)
            self.assertAlmostEqual(numpy.linalg.cond(res), cond)

            numpy.random.seed(cond)
            half = helpers.gen_general_obj(size, True, cond, 0.5)
            self.assertTrue(np.allclose(res, half * 2))

    def test_non_convex(self):
        """ Test for non convex cases.
        """
        for size in range(2, 101):
            cond = random.randint(1, 100)

            numpy.random.seed(cond)
            res = helpers.gen_general_obj(size, False, cond, 1)

            self.assertEqual(res.shape, (size, size))
            if size > 50:
                self.assertLess(helpers.mindiag(res), 0)
            self.assertAlmostEqual(numpy.linalg.cond(res), cond)

            numpy.random.seed(cond)
            half = helpers.gen_general_obj(size, False, cond, 0.5)
            self.assertTrue(np.allclose(res, half * 2))


class TestSanitizeParams(unittest.TestCase):
    """ Unittest for sanitize_params.
    """

    def setUp(self):
        """ Prepare test.
        """
        self.param = {
            "cond_P": 20,
            "convex_f": True,
            "symm_M": True,
            "mono_M": True,
            "cond_M": 10,
            "scale_P": 1,
            "scale_M": 1,
            "p": 123,
            "first_deg": 50,
            "second_deg": 100,
            "mix_deg": 10,
            "l": 100,
            "n": 100,
            "m": 50,
            "yinfirstlevel": True,
            "make_with_dbl_comps": False
        }

    def test_fix_cond_P(self):
        """ Test fixing cond_P.
        """
        self.param["cond_P"] = 0
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["cond_P"], 20)

    def test_fix_convex_f(self):
        """ Test fixing convex_f.
        """
        self.param["convex_f"] = None
        res = helpers.sanitize_params(self.param)
        self.assertTrue(res["convex_f"])

    def test_fix_symm_M(self):
        """ Test fixing symm_M.
        """
        self.param["symm_M"] = None
        res = helpers.sanitize_params(self.param)
        self.assertTrue(res["symm_M"])

    def test_fix_mono_M(self):
        """ Test fixing mono_M.
        """
        self.param["mono_M"] = None
        res = helpers.sanitize_params(self.param)
        self.assertTrue(res["mono_M"])

    def test_fix_cond_M(self):
        """ Test fixing cond_M.
        """
        self.param["cond_M"] = 0
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["cond_M"], 10)

    def test_fix_scale_P(self):
        """ Test fixing scale_P.
        """
        self.param["scale_P"] = 0
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["scale_P"], self.param["cond_P"])

    def test_fix_missing_scale_P(self):
        """ Test fixing missing scale_P.
        """
        del self.param["scale_P"]
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["scale_P"], self.param["cond_P"])

    def test_fix_scale_M(self):
        """ Test fixing scale_M.
        """
        self.param["scale_M"] = 0
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["scale_M"], self.param["cond_M"])

    def test_fix_missing_scale_M(self):
        """ Test fixing missing scale_M.
        """
        del self.param["scale_M"]
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["scale_M"], self.param["cond_M"])

    def test_fix_missing_p(self):
        """ Test fixing missing p.
        """
        del self.param["p"]
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["p"], self.param["m"])

    def test_fix_second_deg_100(self):
        """ Test fixing second_deg when qpec_type = 100.
        """
        self.param["second_deg"] = self.param["p"] + 100
        res = helpers.sanitize_params(self.param, qpec_type=100)
        self.assertEqual(res["second_deg"], self.param["p"])

    def test_fix_second_deg_200(self):
        """ Test fixing second_deg when qpec_type = 200.
        """
        self.param["second_deg"] = self.param["m"] + 100
        res = helpers.sanitize_params(self.param, qpec_type=200)
        self.assertEqual(res["second_deg"], self.param["m"])

    def test_fix_l_800(self):
        """ Test fixing l when qpec_type = 800.
        """
        res = helpers.sanitize_params(self.param, qpec_type=800)
        self.assertEqual(res["l"], 0)

    def test_fix_n_800(self):
        """ Test fixing n when qpec_type = 800.
        """
        self.param["n"] = self.param["m"] + 100
        res = helpers.sanitize_params(self.param, qpec_type=800)
        self.assertEqual(res["n"], self.param["m"])

    def test_fix_first_deg(self):
        """ Test fixing first_deg.
        """
        self.param["first_deg"] = self.param["l"] + 100
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["first_deg"], self.param["l"])

    def test_fix_mix_deg(self):
        """ Test fixing mix_deg.
        """
        self.param["mix_deg"] = self.param["second_deg"] + 100
        res = helpers.sanitize_params(self.param)
        self.assertEqual(res["mix_deg"], self.param["second_deg"])

    def test_fix_yinfirstlevel(self):
        """ Test fixing yinfirstlevel.
        """
        self.param["yinfirstlevel"] = None
        res = helpers.sanitize_params(self.param)
        self.assertTrue(res["yinfirstlevel"])

    def test_fix_missing_make_with_dbl_comps(self):
        """ Test fixing missing yinfirstlevel.
        """
        del self.param["make_with_dbl_comps"]
        res = helpers.sanitize_params(self.param)
        self.assertFalse(res["make_with_dbl_comps"])

    def test_fix_make_with_dbl_comps_200(self):
        """ Test fixing make_with_dbl_comps when qpec_type = 200.
        """
        self.param["make_with_dbl_comps"] = True
        res = helpers.sanitize_params(self.param, qpec_type=200)
        self.assertFalse(res["make_with_dbl_comps"])

    def test_fix_make_with_dbl_comps_201(self):
        """ Test fixing make_with_dbl_comps when qpec_type = 201.
        """
        self.param["make_with_dbl_comps"] = True
        res = helpers.sanitize_params(self.param, qpec_type=201)
        self.assertFalse(res["make_with_dbl_comps"])


class TestCreateName(unittest.TestCase):
    """ Unittest for create_name.
    """

    def test(self):
        """ Test with a simple input.
        """
        prefix = "abd"
        size = 10
        start = 5

        res = helpers.create_name(prefix, size, start=start)
        for number, name in zip(range(start, start + size), res):
            desired = "{prefix}{number}".format(prefix=prefix, number=number)
            self.assertEqual(name, desired)


if __name__ == "__main__":
    unittest.main()
