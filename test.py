import unittest
from individual import Individual
from objective_value import ObjectiveValue
from lotz import LOTZ
from mlotz import mLOTZ, mLOTZConstructor
from nsga_ii import NSGA_II

#####################################
# Tests for the Individual class
#####################################
class TestIndividual(unittest.TestCase):
    def test_initialization(self):
        """Test initialization and attribute correctness."""
        vec = [True, False, True]
        ind = Individual(vec, 3)
        self.assertEqual(ind.n, 3)
        self.assertEqual(ind.x, vec)

    def test_indexing(self):
        """Test single element and slice access."""
        vec = [True, False, True, False]
        ind = Individual(vec, 4)
        self.assertEqual(ind[0], True)
        self.assertEqual(ind[1:3], Individual(vec[1:3], 2))

    def test_iteration(self):
        """Test iteration over the individual."""
        vec = [True, False, True]
        ind = Individual(vec, 3)
        self.assertEqual(list(ind), vec)

    def test_equality(self):
        """Test equality operator."""
        ind1 = Individual([True, False, True], 3)
        ind2 = Individual([True, False, True], 3)
        ind3 = Individual([False, True, False], 3)
        self.assertEqual(ind1, ind2)
        self.assertNotEqual(ind1, ind3)

    def test_hashing(self):
        """Test correct hashing behavior."""
        ind1 = Individual([True, False, True], 3)
        ind2 = Individual([True, False, True], 3)
        ind3 = Individual([False, True, False], 3)
        self.assertEqual(hash(ind1), hash(ind2))
        self.assertNotEqual(hash(ind1), hash(ind3))

#####################################
# Tests for the ObjectiveValue class
#####################################
class TestObjectiveValue(unittest.TestCase):
    def test_initialization(self):
        """Test initialization and attribute correctness."""
        ind = Individual([True, False], 2)
        obj_val = ObjectiveValue(2, ind, [0.5, 1.0])
        self.assertEqual(obj_val.m, 2)
        self.assertEqual(obj_val.value(), [0.5, 1.0])

    def test_dominance(self):
        """Test weak and strict dominance."""
        ind = Individual([True, False], 2)
        obj1 = ObjectiveValue(2, ind, [1.0, 1.0])
        obj2 = ObjectiveValue(2, ind, [0.5, 0.5])
        obj3 = ObjectiveValue(2, ind, [1.0, 0.5])
        self.assertTrue(ObjectiveValue.strictly_dominates(obj1, obj2))
        self.assertFalse(ObjectiveValue.strictly_dominates(obj2, obj1))
        self.assertTrue(ObjectiveValue.weakly_dominates(obj1, obj3))

#####################################
# Tests for the LOTZ benchmark
#####################################
class TestLOTZ(unittest.TestCase):
    def test_lotz_calculation(self):
        """Test LOTZ leading ones and trailing zeros calculation."""
        ind = Individual([1, 1, 0, 0], 4)
        lotz = LOTZ(ind)
        self.assertEqual(lotz.calculate_leading_ones(), 2)
        self.assertEqual(lotz.calculate_trailing_zeros(), 2)

#####################################
# Tests for the mLOTZ benchmark
#####################################
class TestmLOTZ(unittest.TestCase):
    def test_mlotz_splitting(self):
        """
        Test that mLOTZ correctly partitions the individual.
        For mLOTZ with m = 4 and n = 8, the individual is partitioned into two chunks:
          - For keys 0 and 1 (first chunk), LOTZ on x[0:4] should yield (leading ones, trailing zeros) = (2,2)
          - For keys 2 and 3 (second chunk), LOTZ on x[4:8] should yield (2,2)
        """
        ind = Individual([1, 1, 0, 0, 1, 1, 0, 0], 8)
        mlotz = mLOTZ(4, 8, ind)
        self.assertEqual(mlotz[0], 2)  # key 0: leading ones on first chunk
        self.assertEqual(mlotz[1], 2)  # key 1: trailing zeros on first chunk
        self.assertEqual(mlotz[2], 2)  # key 2: leading ones on second chunk
        self.assertEqual(mlotz[3], 2)  # key 3: trailing zeros on second chunk

#####################################
# Tests for the NSGA-II algorithm
#####################################
class TestNSGAII(unittest.TestCase):
    def setUp(self):
        """
        Set up a small NSGA-II instance for testing.
        We use n = 4 and m = 2. For LOTZ (which is equivalent to 2LOTZ),
        the Pareto front is {(k, 4-k) | k=0,...,4} (i.e. 5 points).
        For theoretical guarantees, we use a population size of N = 4*(n+1) = 20.
        """
        self.n = 4
        self.m = 2
        self.population_size = 20  # 4*(n+1)
        self.nsga = NSGA_II(mLOTZConstructor(self.m, self.n))

    def test_population_generation(self):
        """Test that the population is generated with the correct size and type."""
        pop = self.nsga.generate_population(self.population_size, self.n, seed=42)
        self.assertEqual(len(pop), self.population_size)
        self.assertTrue(all(isinstance(ind, Individual) for ind in pop))

    def test_mutation(self):
        """Test that mutation produces an individual of the same length."""
        ind = Individual([True, False, True, False], 4)
        mutated = self.nsga.mutation(ind, seed=42)
        self.assertEqual(mutated.n, ind.n)
        self.assertEqual(len(mutated.x), 4)

    def test_non_dominated_sorting_all_equal(self):
        """
        Test non-dominated sorting when all individuals are identical.
        In this case, they should all be non-dominated and be in one front.
        """
        ind = Individual([True, True, False, False], 4)
        population = [ind, ind, ind]
        sorted_ranks = self.nsga.non_dominated_sorting(population)
        self.assertEqual(len(sorted_ranks), 1)
        self.assertEqual(len(sorted_ranks[0]), 3)

    def test_non_dominated_sorting_order(self):
        """
        Test non-dominated sorting with individuals that have different objective values.
        Using LOTZ via mLOTZConstructor (with m=2, n=4), we construct three individuals:
          - ind1: [True, True, False, False]  -> LOTZ -> (2,2)
          - ind2: [True, False, False, False]   -> LOTZ -> (1,3)
          - ind3: [False, False, True, True]    -> LOTZ -> (0,0)
        In a maximization setting, (2,2) and (1,3) are non-dominated relative to each other,
        while (0,0) is dominated by both.
        Hence, we expect two fronts.
        """
        ind1 = Individual([True, True, False, False], 4)    # (2,2)
        ind2 = Individual([True, False, False, False], 4)     # (1,3)
        ind3 = Individual([False, False, True, True], 4)      # (0,0)
        population = [ind1, ind2, ind3]
        sorted_ranks = self.nsga.non_dominated_sorting(population)
        self.assertGreaterEqual(len(sorted_ranks), 2)
        self.assertNotIn(ind3, sorted_ranks[0])
        self.assertIn(ind1, sorted_ranks[0])
        self.assertIn(ind2, sorted_ranks[0])

    def test_crowding_distance(self):
        """
        Test the crowding distance calculation.
        Verify that the returned dictionary has one entry per individual.
        """
        ind1 = Individual([True, True, False, False], 4)
        ind2 = Individual([False, False, True, True], 4)
        ind3 = Individual([True, False, True, False], 4)
        population = [ind1, ind2, ind3]
        distances = self.nsga.crowding_distance(population)
        self.assertEqual(len(distances), len(population))

    def test_nsga_ii_execution(self):
        """
        Test a complete NSGA-II execution on a small LOTZ instance.
        With n=4 and m=2, the Pareto front for LOTZ is {(k, 4-k) | k=0,...,4} (5 points).
        We run the algorithm until the termination criterion is met or the maximum iterations are reached.
        """
        iterations, final_pop = self.nsga.run(self.population_size, self.n, self.m, seed=42)
        self.assertGreater(iterations, 0)
        self.assertEqual(len(final_pop), self.population_size)
        # Check that the final population covers the full Pareto front.
        pareto_values = set()
        for ind in final_pop:
            val = tuple(self.nsga.f(ind).value())
            pareto_values.add(val)
        self.assertGreaterEqual(len(pareto_values), 5)

    def test_nsga_ii_progress_monitoring(self):
        """
        Test that NSGA-II eventually reaches full Pareto front coverage.
        For our small instance (n=4, m=2, population size=20), the final population should cover 5 Pareto front points.
        """
        iterations, final_pop = self.nsga.run(self.population_size, self.n, self.m, seed=42)
        ratio = self.nsga.population_covers_pareto_front(final_pop, self.m, self.n)
        self.assertGreaterEqual(ratio, 0.999)

if __name__ == "__main__":
    unittest.main()