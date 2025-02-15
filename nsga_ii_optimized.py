import numpy as np
import random
from objective_value import ObjectiveValueConstructor
from individual import Individual
from typing import Dict, List, Optional, Tuple

class NSGA_II_Optimized:
    """
    An optimized implementation of the NSGA-II algorithm for multi-objective optimization.
    
    This class implements NSGA-II using NumPy vectorized operations and caching of objective values to reduce redundant 
    computations. In particular, it performs vectorized non-dominated sorting, computes crowding distances in a 
    vectorized manner, and applies mutation using vectorized bit-flip operations. The algorithm uses a fixed population 
    size and stops either when the termination criterion is met (i.e. the population covers the entire Pareto front) or 
    when a maximum number of iterations is reached.
    
    Parameters:
        f (ObjectiveValueConstructor): An instance that, given an Individual, produces an ObjectiveValue representing 
                                       the multi-objective evaluation of that individual.
        seed (Optional[int]): A random seed to be used for reproducibility. If provided, it is used to initialize the 
                              random number generators for both Python's random module and NumPy.
    
    Methods:
        _compute_objectives(population): Computes and caches objective values for the entire population.
        update_seed(seed): updates the seed used for randomness 
        non_dominated_sorting(population, cached_obj): Performs vectorized non-dominated sorting on a population.
        crowding_distance(population, cached_obj): Computes the crowding distances for all individuals in a population.
        mutation(original): Applies vectorized bit-flip mutation to an individual.
        generate_population(N, n, seed): Generates an initial population of N individuals, each with n bits.
        run(population_size, problem_size, number_objectives, seed): Runs NSGA-II until the termination criterion is met or a 
            maximum number of iterations is reached.
        population_covers_pareto_front(population, m, n, cached_obj): Computes the ratio of the Pareto front points covered by 
            the population.
    """
    name = "Task's 6 original NSGA-II"
    def __init__(self, f: ObjectiveValueConstructor, seed: Optional[int] = None) -> None:
        self.f = f
        self._seed = None
        if seed:
            self.update_seed(seed)

    def update_seed(self, seed: int) -> None:
        """
        Update the random seed if the provided seed differs from the current seed.
        
        This method initializes (or re-initializes) the random number generators for both Python’s built-in random module 
        and NumPy's random generator. This ensures that the results of stochastic operations (such as mutation or 
        population initialization) are reproducible. If the provided seed is None or the same as the current seed, 
        no update is performed.
        
        Parameters:
            seed (int): The new seed value to be set.
        """
        if (seed is not None) and (seed != self._seed):
            self._seed = seed
            random.seed(seed)
            np.random.seed(seed)


    def _compute_objectives(self, population: List[Individual]):
        """
        Compute and cache the objective values for the entire population.
        
        Parameters:
            population (List[Individual]): A list of individuals for which the objective values will be computed.
        
        Returns:
            A tuple (obj_vals, obj_list), where:
                - obj_vals is a NumPy array of shape (N, m) containing the objective values for each individual, with 
                  N being the population size and m = self.f.m the number of objectives.
                - obj_list is a list of objective vectors (each a list of floats) in the same order as the individuals.
        """
        N = len(population)
        m = self.f.m
        obj_vals = np.zeros((N, m))
        obj_list = []
        for i, ind in enumerate(population):
            obj = self.f(ind).value()
            obj_vals[i, :] = obj
            obj_list.append(obj)
        return obj_vals, obj_list

    def non_dominated_sorting(self,  population: List[Individual], cached_obj: Optional[np.ndarray] = None) -> List[List[Individual]]:
        """
        Perform vectorized non-dominated sorting on the given population.
        
        Parameters:
            population (List[Individual]): The list of individuals to be sorted into fronts.
            cached_obj (Optional[np.ndarray]): If provided, a NumPy array of shape (N, m) containing precomputed objective 
                                               values for the population. If None, objective values are computed.
        
        Returns:
            A list of fronts, where each front is a list of Individuals. The first front contains all non-dominated individuals,
            the second front contains individuals dominated only by those in the first front, and so on.
        """
        N = len(population)
        m = self.f.m
        if cached_obj is None:
            obj_vals, _ = self._compute_objectives(population)
        else:
            obj_vals = cached_obj  # shape (N, m)
        
        # Create two arrays A and B so that:
        # A[i, j, :] = obj_vals[i, :] and B[i, j, :] = obj_vals[j, :]
        A = obj_vals[:, np.newaxis, :]   # shape (N, 1, m)
        B = obj_vals[np.newaxis, :, :]   # shape (1, N, m)
        # For each pair (i,j), check if A[i] > B[j] for all objectives:
        dominates = np.all(A > B, axis=2)  # shape (N, N)
        # Count how many individuals dominate each j:
        domination_count = np.sum(dominates, axis=0)  # shape (N,)
        
        sorted_ranks = []
        remaining = set(range(N))
        while remaining:
            # Individuals with zero domination count among those remaining:
            current_front = [i for i in remaining if domination_count[i] == 0]
            if not current_front:
                break
            # Append the corresponding individuals to the front list.
            sorted_ranks.append([population[i] for i in current_front])
            # Remove these indices from remaining.
            for i in current_front:
                remaining.remove(i)
            # For every individual still remaining, subtract the domination from individuals in current_front.
            for i in current_front:
                for j in remaining:
                    if dominates[i, j]:
                        domination_count[j] -= 1
        return sorted_ranks

    def crowding_distance(self, population: List[Individual], cached_obj: Optional[np.ndarray] = None) -> Dict[Individual, float]:
        """
        Compute crowding distances for a given population in a vectorized manner.
        
        Parameters:
            population (List[Individual]): The list of individuals for which the crowding distance is to be computed.
            cached_obj (Optional[np.ndarray]): A precomputed NumPy array of shape (N, m) of objective values for the population.
                                               If not provided, objective values will be computed.
        
        Returns:
            A dictionary mapping each Individual in the population to its computed crowding distance (a float).
        """
        N = len(population)
        m = self.f.m
        if cached_obj is None:
            obj_vals, _ = self._compute_objectives(population)
        else:
            obj_vals = cached_obj
        
        distances = np.zeros(N)
        for k in range(m):
            sorted_indices = np.argsort(obj_vals[:, k])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            denom = obj_vals[sorted_indices[-1], k] - obj_vals[sorted_indices[0], k]
            if denom == 0:
                # If no spread, assign infinite distance for interior individuals.
                distances[sorted_indices[1:-1]] = float('inf')
            else:
                # Compute difference between the neighbor values.
                diff = obj_vals[sorted_indices[2:], k] - obj_vals[sorted_indices[:-2], k]
                distances[sorted_indices[1:-1]] += diff / denom
        # Build a dictionary mapping individual to its computed distance.
        result: dict = {}
        for idx, ind in enumerate(population):
            result[ind] = distances[idx]
        return result

    def mutation(self, original: Individual) -> Individual:
        """
        Vectorized bit-flip mutation.
        Each bit is flipped with probability 1/n.
        :param original: The original individual to mutate.
        :return: A new mutated individual.
        """
        n = original.n
        arr = np.array(original.x, dtype=int)
        random_probs = np.random.rand(n)
        flip_mask = random_probs < (1.0 / n)
        new_arr = np.where(flip_mask, 1 - arr, arr)
        new_vec = new_arr.tolist()
        return Individual(new_vec, n)

    def generate_population(self, N: int, n: int, seed: Optional[int] = None) -> List[Individual]:
        """
        Generate a population of N individuals with n-bit sequences using numpy.
    
        :param N: Number of individuals
        :param n: Length of each individual's bit sequence
        :param seed: Random seed for reproducibility (default: None)
        :return: List(population), where population is a list of N individuals (each a list of n bits)
        """
        self.update_seed(seed)
        # Use a probability vector to generate bits (for each bit, probability uniformly in [0,1])
        p_vector = np.random.uniform(0, 1, n)
        pop_arr = (np.random.rand(N, n) < p_vector).astype(int)
        population = [Individual(list(row), n) for row in pop_arr]
        return population

    def run(self, population_size: int, problem_size: int, number_objectives: int, seed: Optional[int]=None, remove_duplicates: bool = False) -> Tuple[int, List[Individual]]:
        """
        Run NSGA-II until the termination criterion is met or a maximum number of iterations is reached.
        This method caches objective values per generation to avoid redundant computations.
        
        :param population_size: (N) The number of individuals in the population.
        :param problem_size: (n) The length of each individual's bit sequence.
        :param number_objectives: (m) The number of objectives to optimize. For the LOTZ benchmark,
                                  m must be an even number. In this context, the bit string is partitioned
                                  into m/2 equal-length disjoint chunks. For each chunk, two objectives are computed:
                                  the first objective returns the number of consecutive ones (leading ones) starting from the beginning
                                  of the chunk, and the second objective returns the number of consecutive zeros (trailing zeros) starting
                                  from the end of the chunk.
        :param seed: Random seed for reproducibility (default: None).
        :param remove_duplicates: Determines if duplicates are going to be removed in each iteration (default: False).
        :return: Tuple(counter, P_t), where:
                 - counter (int): The total number of iterations executed.
                 - P_t (List[Individual]): The final population (last generation) of size N, where each individual is represented
                                           as a list of n bits.

        """

        N = population_size
        n = problem_size
        m = number_objectives
        P_t = self.generate_population(N, n, seed)
        counter = 0
        max_iterations = 9 * n**2
        epsilon = 0.001
        
        while counter < max_iterations:
            # Generate offspring population via vectorized mutation (applied individually).
            y_ti = [self.mutation(x) for x in P_t]
            Q_t = P_t + y_ti
            if remove_duplicates:
                Q_t = list(set(Q_t))
            cached_obj_Q, _ = self._compute_objectives(Q_t)
            # Perform non-dominated sorting using cached objective values.
            fronts = self.non_dominated_sorting(Q_t, cached_obj=cached_obj_Q)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= N:
                    selected.extend(front)
                else:
                    remaining_needed = N - len(selected)
                    if len(front) > remaining_needed:
                        indices = [Q_t.index(ind) for ind in front]
                        front_obj = cached_obj_Q[indices, :]
                        cd = self.crowding_distance(front, cached_obj=front_obj)
                        front_with_distance = [(ind, cd[ind]) for ind in front]
                        front_with_distance.sort(key=lambda pair: pair[1], reverse=True)
                        threshold = front_with_distance[remaining_needed - 1][1]
                        selected_from_front = [pair[0] for pair in front_with_distance if pair[1] > threshold]
                        tied = [pair[0] for pair in front_with_distance if pair[1] == threshold]
                        remaining_to_pick = remaining_needed - len(selected_from_front)
                        if remaining_to_pick > 0:
                            selected_from_front.extend(random.sample(tied, remaining_to_pick))
                        else:
                            selected_from_front = selected_from_front[:remaining_needed]
                    else:
                        selected_from_front = front
                    selected.extend(selected_from_front)
                    break
            P_t = selected
            cached_obj, _ = self._compute_objectives(P_t)
            ratio = self.population_covers_pareto_front(P_t, m, n, cached_obj=cached_obj)
            counter += 1
            if abs(ratio-1) <= epsilon:
                break
        return counter, P_t

    def population_covers_pareto_front(self, population: List[Individual], m: int, n: int, cached_obj: Optional[np.ndarray] = None) -> float:
        """
        Returns the ratio of Pareto front points covered.
        For LOTZ (or mLOTZ) the Pareto front is determined by chunk length n_prime = 2n/m.
        For each chunk, valid Pareto points are those of the form (i, n_prime-i).
        Thus, total front size = (n_prime+1)^(m/2).

        Parameters:
            population (List[Individual]): A list of individuals for which the objective values will be computed.
            m (int): The number of objectives to optimize.
            n (int): Length of each individual's bit sequence
            cached_obj (Optional[np.ndarray]): A precomputed NumPy array of shape (N, m) of objective values for the population.
                                               If not provided, objective values will be computed.

        Returns:
            A float len(population_pareto_values) / total_pareto_optima that represents the ratio of pareto optimas that are in our current population, where:
                - len(population_pareto_values) returns the number of pareto optimas in our current population
                - total_pareto_optima is the total number of pareto optimas for the given n and m

        """
        n_prime = int(2 * n / m)
        total_pareto_optima = (n_prime + 1) ** (m // 2)
        if cached_obj is None:
            _, obj_list = self._compute_objectives(population)
        else:
            obj_list = [list(row) for row in cached_obj]
        population_pareto_values = set()
        for obj_val in obj_list:
            is_pareto = True
            for k in range(m // 2):
                if obj_val[2*k] + obj_val[2*k+1] != n_prime:
                    is_pareto = False
                    break
            if is_pareto:
                population_pareto_values.add(tuple(obj_val))
        return len(population_pareto_values) / total_pareto_optima