from typing import Dict, List, Optional

import numpy as np
from objective_value import ObjectiveValue, ObjectiveValueConstructor
from individual import Individual
import random



class NSGA_II:
    def __init__(self, f: ObjectiveValueConstructor):
        self.f = f
    
    # non dominated sorting criterion for breaking ties
    def non_dominated_sorting(self, population: List[Individual]) -> List[List[Individual]]:
        n = len(population)
        dominance_matrix = [[0] * n for _ in range(n)]
        population_obj = [self.f(x) for x in population]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dominance_matrix[i][j] = int(ObjectiveValue.strictly_dominates(population_obj[i], population_obj[j]))
                    # Column j will store the elements that strictly dominate the jth element from the population_obj

        remaining_pop = {i: sum(row[i] for row in dominance_matrix) for i in range(n)}
        sorted_ranks = []
        while remaining_pop:
            cur_rank = []
            removed_list = []
            for obj_val_ix, count_dominations in list(remaining_pop.items()):
                if count_dominations == 0:
                    cur_rank.append(population[obj_val_ix])
                    removed_list.append(obj_val_ix)
                    del remaining_pop[obj_val_ix]
            sorted_ranks.append(cur_rank)
            for i in removed_list:
                for j in remaining_pop.keys():
                    remaining_pop[j] -= dominance_matrix[i][j]

        return sorted_ranks

    def crowding_distance(self, population: List[Individual]) -> Dict[Individual, float]:
       def sort_key(individual, k):
           obj_val = self.f(individual)
           return obj_val[k]
       inf = float(10**10)
       n = len(population)
       distance_dict = {individual: 0 for individual in population}
       for k in range(self.f.m):
           sorted_pop = sorted(population, key=lambda ind: sort_key(ind, k))
           distances = [0]*n
           distances[0] = distances[n-1]  = inf 
           distance_dict[sorted_pop[0]] = distance_dict[sorted_pop[n-1]] = inf
           for i in range(1, n-1):
               if (self.f(sorted_pop[n-1])[k] == self.f(sorted_pop[0])[k]) or (distance_dict[sorted_pop[i]]==inf):
                   distance_dict[sorted_pop[i]] = inf
               else:
                   distance_dict[sorted_pop[i]] += (self.f(sorted_pop[i+1])[k] - self.f(sorted_pop[i-1])[k])/(self.f(sorted_pop[n-1])[k] - self.f(sorted_pop[0])[k])
       return distance_dict
    
    
    #mutation operation used in the NSGA_II
    def mutation(self, original: Individual, seed: Optional[int] = None) -> Individual:
        """
        Performs bit-flip mutation on an individual with probability 1/n per bit.

        :param original: The original individual to mutate.
        :param seed: Optional seed for reproducibility.
        :return: A new mutated individual.
        """
        n = original.n
        y = Individual(original.x, n)  # Copy of the original

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Generate mutation probabilities for all bits at once
        mutation_probs = np.random.rand(n)

        # Mutate bits where probability < 1/n
        y.x = list(np.where(mutation_probs < (1 / n), 1 - np.array(y.x), np.array(y.x)).tolist())

        return y
    

    def generate_population(N:int, n:int, seed: Optional[int] = None) -> List[Individual]:
        """
        Generates a population of N individuals with n-bit sequences,
        using a deterministic random seed for reproducibility.
    
        :param N: Number of individuals
        :param n: Length of each individual's bit sequence
        :param seed: Random seed for reproducibility (default: None)
        :return: Tuple (population, p_vector), where:
                 - population is a list of N individuals (each a list of n bits)
                 - p_vector is the list of predefined probabilities
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)  # Set NumPy seed for consistency
    
        # Generate probability vector using NumPy (faster and cleaner)
        p_vector = np.random.uniform(0, 1, n)
    
        # Generate population using NumPy for efficiency
        population = (np.random.rand(N, n) < p_vector).astype(int).tolist()
        population = [Individual(population[i],n) for i in range(N)]
        
        #population = [Individual(ind, n) for ind in (np.random.rand(N, n) < p_vector).astype(int)]
        return population
    
    # the NSGA_II algorithm itself
    def run(self, population_size: int, problem_size: int, number_objectives: int, seed: Optional[int] = None) -> List[Individual]:
        N = population_size
        n = problem_size
        m = number_objectives
        P_t = self.generate_population(N,n, seed)
        counter = 0
        n_prime = int(2*n/m)
        total_pareto_optima = (n_prime + 1) ** (m // 2)
        
        while counter < 9*n**2: 
            
            y_ti = [0]*N #population of the mutated individuals
            for i in range(N):
                x_ti = P_t[random.randint(0,N-1)]
                y_ti[i] = self.mutation(x_ti, seed)
            Q_t = P_t + y_ti # the union of the original population and its mutated copy
            fronts = self.non_dominated_sorting(Q_t)
            
            # After computing the fronts by non-dominated sorting:
            selected = []  # will hold the chosen individuals for the new population

            # First, add all fronts as long as doing so does not exceed the desired population size N
            for front in fronts:
                if len(selected) + len(front) <= N:
                    selected.extend(front)
                else:
                    # We are in the critical front, call it front_i_star
                    remaining_needed = N - len(selected)

                    # Compute the crowding distances for the individuals in this front
                    crowding_distances = self.crowding_distance(front)
                    # Create list of (individual, distance) pairs
                    front_with_distance = [(ind, crowding_distances[ind]) for ind in front]
                    # Sort descending by crowding distance
                    front_with_distance.sort(key=lambda pair: pair[1], reverse=True)

                    # If the number of individuals in this front exceeds remaining_needed,
                    # we have to break ties at the boundary.
                    if len(front_with_distance) > remaining_needed:
                        # Identify the crowding distance of the last candidate in the top remaining_needed
                        threshold = front_with_distance[remaining_needed - 1][1]
                        # Collect individuals with crowding distance strictly greater than threshold:
                        selected_from_front = [pair[0] for pair in front_with_distance if pair[1] > threshold]
                        # Collect individuals with exactly the threshold distance:
                        tied = [pair[0] for pair in front_with_distance if pair[1] == threshold]

                        # Determine how many more individuals we need:
                        remaining_to_pick = remaining_needed - len(selected_from_front)
                        if remaining_to_pick > 0:
                            # Randomly pick the required number among those that are tied
                            selected_from_front.extend(random.sample(tied, remaining_to_pick))
                        else:
                            # (In the unlikely event that remaining_to_pick is 0)
                            selected_from_front = selected_from_front[:remaining_needed]
                    else:
                        # If the front’s size is exactly equal to the remaining slot count, take all.
                        selected_from_front = [pair[0] for pair in front_with_distance]

                    # Add the selected individuals from the critical front
                    selected.extend(selected_from_front)
                    # We have now selected N individuals, so break out.
                    break
                
            # Now update the new population with exactly N individuals
            P_t = selected
            
            
            counter += 1
                
            
            
            
            
            
                    
                
            
                
            
        
        
        
