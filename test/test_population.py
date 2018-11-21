from population import BaseParameters, Population
import unittest


class NumbaPopulationTest(unittest.TestCase):
    def test_triangle_n(self):
        """ Verify the triangle number generator for a few happy paths.

        :return: None.
        """
        self.assertEqual(triangle_n(0), 0)
        self.assertEqual(triangle_n(1), 1)
        self.assertEqual(triangle_n(2), 3)
        self.assertEqual(triangle_n(3), 6)
        self.assertEqual(triangle_n(4), 10)
        self.assertEqual(triangle_n(5), 15)
        self.assertEqual(triangle_n(6), 21)
        self.assertEqual(triangle_n(7), 28)
        self.assertEqual(triangle_n(8), 36)
        self.assertEqual(triangle_n(9), 45)
        self.assertEqual(triangle_n(10), 55)
        self.assertEqual(triangle_n(100), 5050)
        self.assertEqual(triangle_n(500), 125250)
        self.assertEqual(triangle_n(1000), 500500)

    def test_mutate_n(self):
        """ Verify the mutation method for our edge cases and several happy paths.

        :return: None.
        """
        from numpy.random import uniform
        from numpy import iinfo, int32
        from math import inf

        # We should remain at 5, if our upward and downward mutation rates are equal to 0.
        self.assertEqual(5, mutate_n(5, 0.0, 1.0, 0.0, 3, 10))

        # We should ascend to omega=30, if our upward mutation rate = 1 and our downward mutation is equal to 0.
        i = 5
        for _ in range(35):
            i = mutate_n(i, 1.0, 1.0, 0.0, 3, 30)
        self.assertEqual(30, i)

        # We should descend to kappa=5, if our downward mutation rate > 1 and our downward mutation rate is equal to 1.
        i = 10
        for _ in range(10):
            i = mutate_n(i, 0.0, inf, 1.0, 5, 30)
        self.assertEqual(5, i)

        # We should remain at i=10, if our downward mutation rate equals our upward mutation rate.
        i = 10
        for _ in range(100):
            i = mutate_n(i, 0.0, 1.0, 1.0, 5, 30)
        self.assertEqual(10, i)

        # We should only be mutating no more than one step for each mutate_n call.
        for _ in range(1000):
            i = mutate_n(5, uniform(0, 1), uniform(1, iinfo(int32).max), uniform(0, 1), 2, 30)
            self.assertLessEqual(i, 6)
            self.assertGreaterEqual(i, 4)

        # We should not be able to mutate if we are stuck at kappa=5.
        self.assertEqual(5, mutate_n(5, 1.0, 1.0, 0.0, 5, 30))

        # We should be able to mutate down, if we have reached our maximum omega=30.
        self.assertEqual(29, mutate_n(30, 0.0, inf, 1.0, 5, 30))

    def test_coalesce_n(self):
        """ Verify the sub-tree generating method for several happy paths.

        :return: None.
        """
        from numpy import array

        # We generate an negative 1D array to map our tree to.
        ell = array([-1000 for _ in range(triangle_n(2 * 1000))])

        # After the 1st coalescent event, ell should contain [-1000, 0, 0, ...] where -1000 is not altered.
        for _ in range(100):
            coalesce_n(0, ell)
            self.assertEqual(-1000, ell[0])
            self.assertEqual(0, ell[1])
            self.assertEqual(0, ell[2])

        # After the 2nd coalescent event, ell should still contain [-1000, 0, 0, a_1, a_2, a_3].
        for _ in range(100):
            coalesce_n(1, ell)
            self.assertEqual(-1000, ell[0])
            self.assertEqual(0, ell[1])
            self.assertEqual(0, ell[2])
            self.assertLess(0, ell[3]), self.assertGreater(3, ell[3])
            self.assertLess(0, ell[4]), self.assertGreater(3, ell[4])

        # We should be able to address our chain using the triangle numbers.
        coalesce_n(99, ell)  # 100th event, we use tau = 99.
        for v in ell[triangle_n(100):triangle_n(101)]:
            self.assertIn(v, list(range(triangle_n(99), triangle_n(100))))

        # After 2000 coalescent events, all of ell should be populated.
        [coalesce_n(tau, ell) for tau in range(0, 2 * 1000 - 1)]
        for i in ell[1:]:
            self.assertGreaterEqual(i, 0)

    # @unittest.skip("'test_evolve_n' takes too long. Include with major rewrites.")
    def test_evolve_n(self):
        """ Verify the repeat length determination method for several happy paths.

        :return: None.
        """
        from numpy import empty, average, array

        # We generate an empty 1D array to map our tree to.
        ell = empty([triangle_n(2 * 1000)], dtype='int')

        # For single ancestors, and a focal bias \hat{L} ~= 11, we should end up with an average length close to 11.
        a = 0
        for _ in range(50):
            [coalesce_n(tau, ell) for tau in range(0, 2 * 1000 - 1)]
            ell[0] = 11
            [evolve_n(ell, tau, 100, 1.0, 0.0001, 1.1, 0.0001, 3, 100) for tau in range(0, 2 * 1000 - 1)]
            a += average(ell[-2 * 1000:])
        self.assertAlmostEqual(a / 50.0, 11, delta=1.0)

        # For ancestors of two, we should again end up with an average length close to 11.
        a = 0
        for _ in range(50):
            [coalesce_n(tau, ell) for tau in range(0, 2 * 1000 - 1)]
            ell[1:3] = array([11, 11])
            [evolve_n(ell, tau, 100, 1.0, 0.0001, 1.1, 0.0001, 3, 100) for tau in range(1, 2 * 1000 - 1)]
            a += average(ell[-2 * 1000:])
        self.assertAlmostEqual(a / 50.0, 11, delta=1.0)

        # For ancestors of four, we should again end up with an average length close to 11.
        a = 0
        for _ in range(50):
            [coalesce_n(tau, ell) for tau in range(0, 2 * 1000 - 1)]
            ell[triangle_n(3):triangle_n(4)] = array([11, 11, 11, 11])
            [evolve_n(ell, tau, 100, 1.0, 0.0001, 1.1, 0.0001, 3, 100) for tau in range(3, 2 * 1000 - 1)]
            a += average(ell[-2 * 1000:])
        self.assertAlmostEqual(a / 50.0, 11, delta=1.0)


class BaseParametersTest(unittest.TestCase):
    def test_constructor(self):
        """ Verify the BaseParameter constructor for several edge cases and several happy paths.

        :return: None.
        """
        # All of our parameters should be equal to 0.
        theta = BaseParameters(0, 0, 0, 0, 0, 0)
        self.assertEqual(theta.n, 0)
        self.assertEqual(theta.f, 0)
        self.assertEqual(theta.c, 0)
        self.assertEqual(theta.d, 0)
        self.assertEqual(theta.kappa, 0)
        self.assertEqual(theta.omega, 0)

        # Given floats, our constructor should round to the nearest integer for n, kappa, and omega.
        theta = BaseParameters(0.6, 1, 1, 1, 30.2, 100.9)
        self.assertEqual(theta.n, 1)
        self.assertEqual(theta.kappa, 30)
        self.assertEqual(theta.omega, 101)

        # Given theta = (0, 1, 2, 3, 4, 5, 6), our parameters should be set accordingly.
        theta = BaseParameters(0, 1, 2, 4, 5, 6)
        self.assertEqual(theta.n, 0)
        self.assertEqual(theta.f, 1)
        self.assertEqual(theta.c, 2)
        self.assertEqual(theta.d, 4)
        self.assertEqual(theta.kappa, 5)
        self.assertEqual(theta.omega, 6)

    def test_iter(self):
        """ Verify the BaseParameter iterator for several edge cases and several happy paths.

        :return: None.
        """
        # Our BaseParameter should be able to work in a for loop, we test from [0, 5].
        for a in zip(range(6), BaseParameters(0, 1, 2, 3, 4, 5)):
            self.assertEqual(a[0], a[1])

        # Our BaseParameter should be able to work in a for loop, we test from [-5, 0].
        for a in zip(range(0, -5, -1), BaseParameters(0, -1, -2, -3, -4, -5)):
            self.assertEqual(a[0], a[1])

        # The length in some BaseParameter object should determine the number of iterations to perform.
        a, b = iter([0, 1, 2, 4, 5, 6]), BaseParameters(0, 1, 2, 4, 5, 6)
        c = iter(b)
        for _ in range(len(b)):
            self.assertEqual(next(a), next(c))

    def test_from_args(self):
        """ Verify the namespace BaseParameter instance creator for several happy paths.

        :return: None.
        """
        from types import SimpleNamespace

        # We simulate the parsing of our arguments by creating a namespace.
        namespace = SimpleNamespace(n=1, f=1, c=1, d=1, kappa=1, omega=1,
                                    n_sigma=-1, f_sigma=-1, c_sigma=-1, d_sigma=-1,
                                    kappa_sigma=-1, omega_sigma=-1)

        # Without altering the is_sigma flag, we should only get our initial parameters.
        for a in BaseParameters.from_args(namespace):
            self.assertEqual(1, a)

        # By lowering the is_sigma flag, we should only get our initial parameters.
        for a in BaseParameters.from_args(namespace, False):
            self.assertEqual(1, a)

        # By raising the is_sigma flag, we should only get our pi parameters.
        for a in BaseParameters.from_args(namespace, True):
            self.assertEqual(-1, a)

    def test_from_walk(self):
        """ Verify the pi function within the BaseParameter class for several happy paths.

        :return: None.
        """
        from numpy.random import normal

        # We always walk using some distribution. 'a' represents the past state, 'b' is some parameter of the walk.
        walk = lambda a, b: normal(a, b)

        # Given deviations of 0, we should remain at the same state.
        c, d = BaseParameters(1, 1, 1, 1, 1, 1), BaseParameters(0, 0, 0, 0, 0, 0)
        e = BaseParameters.from_walk(c, d, walk)
        self.assertListEqual(list(c), list(e))

        # Given a small deviation, we should not walk far.
        c, d = BaseParameters(1, 1, 1, 1, 1, 1), BaseParameters(0.001, 0.001, 0.001, 0.001, 0.001, 0.001)
        e = BaseParameters.from_walk(c, d, walk)
        for f in zip(c, e):
            self.assertAlmostEqual(f[0], f[1], delta=1.0)

        # Given a larger deviation, we walk farther.
        c, d = BaseParameters(1, 1, 1, 1, 1, 1), BaseParameters(1000, 1000, 1000, 1000, 1000, 1000)
        e = BaseParameters.from_walk(c, d, walk)
        for f in zip(c, e):
            self.assertNotAlmostEqual(f[0], f[1], delta=1.0)


class PopulationTest(unittest.TestCase):
    def test_constructor(self):
        """ Verify the Population constructor for several edge cases and happy paths.

        :return: None.
        """
        from numpy import nextafter

        # None of our parameters instead the population object should be out of bounds.
        a = Population(BaseParameters(-1, -1, 0, -1, -1, -1))
        self.assertEqual(a.theta.n, 0)
        self.assertEqual(a.theta.f, 0)
        self.assertEqual(a.theta.c, nextafter(0, 1))
        self.assertEqual(a.theta.d, 0)
        self.assertEqual(a.theta.kappa, 0)
        self.assertEqual(a.theta.omega, 0)

        # If kappa > omega, kappa = omega.
        a = Population(BaseParameters(1.0, 1.0, 1.0, 1.0, 30, 3))
        self.assertEqual(a.theta.kappa, a.theta.omega)
        self.assertEqual(a.theta.kappa, 30)

        # If our parameters are correct, then we should see them in our Population object.
        a = Population(BaseParameters(1.0, 1.0, 1.0, 1.0, 3, 30))
        self.assertEqual(a.theta.n, 1.0)
        self.assertEqual(a.theta.f, 1.0)
        self.assertEqual(a.theta.c, 1.0)
        self.assertEqual(a.theta.d, 1.0)
        self.assertEqual(a.theta.kappa, 3)
        self.assertEqual(a.theta.omega, 30)

        # Our ancestor chain length should hold enough elements to store our tree.
        a = Population(BaseParameters(100, 1.0, 1.0, 1.0, 3, 30))
        self.assertEqual(a.coalescent_tree.size, triangle_n(2 * 100))

        # Our tree should be traced upon instantiation.
        c = list(range(triangle_n(198), triangle_n(199)))
        self.assertEqual(a.coalescent_tree[1], 0)
        self.assertEqual(a.coalescent_tree[2], 0)
        for b in a.coalescent_tree[-2 * 100:]:
            self.assertIn(b, c)

        # Our is_evolved flag should not be raised.
        self.assertFalse(a.is_evolved)

    def test_trace_tree(self):
        """ Verify that our tree tracing method for a Population instance is valid for several happy paths. These
        tests are similar to the tests in NumbaPopulationTest.test_coalesce_n.

        :return: None.
        """
        from numpy import array_equal

        a, b = list(range(0, triangle_n(25 * 2))), None
        for p in range(30):

            # Our tree should be populated upon instantiation, except for our initial element.
            c = Population(BaseParameters(25, 1.0, 1.0, 1.0, 3, 30))
            for d in c.coalescent_tree[1:]:
                self.assertIn(d, a)

            # We should not get the same tree for two different runs.
            if p != 0:
                self.assertFalse(array_equal(c.coalescent_tree[1:], b.coalescent_tree[1:]))
            b = c

    def test_evolve(self):
        """ Verify the evolution method for the Population class for several happy paths.

        :return: None.
        """
        from numpy import array, array_equal
        a = Population(BaseParameters(100, 1.0, 0.001, 0.001, 3, 30))
        z = array(a.coalescent_tree)

        # The returned array should be our current descendants.
        b = array(a.evolve(array([10])))
        self.assertEqual(b.size, 200)

        # Our is_evolved flag should be raised.
        self.assertTrue(a.is_evolved)

        # Calling our evolve method should return the same population.
        self.assertTrue(array_equal(b, a.evolve(array([10]))))

        # We should get two different populations, given the same chain and ancestors.
        a.coalescent_tree, a.is_evolved = z, False
        c = a.evolve(array([10]))
        self.assertFalse(array_equal(b, c))

        # Repeat for 10 different runs.
        for _ in range(100):
            a = Population(BaseParameters(100, 1.0, 0.001, 0.001, 3, 30))
            z = array(a.coalescent_tree)
            b = array(a.evolve(array([10])))

            # We should get two different populations, given the same chain and ancestors.
            a.coalescent_tree, a.is_evolved = z, False
            c = a.evolve(array([10]))
            self.assertFalse(array_equal(b, c))


if __name__ == '__main__':
    unittest.main()
