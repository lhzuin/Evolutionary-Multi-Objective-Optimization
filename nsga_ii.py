from typing import Dict, List
from objective_value import ObjectiveValue, ObjectiveValueConstructor
from individual import Individual
import random



class NSGA_II:
    def __init__(self, f: ObjectiveValueConstructor):
        self.f = f
    
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
    
    
    def mutation(self, original: Individual) -> Individual:
        n = original.n
        y = Individual(original.x, n) # firstly y is a brute copy of the original 
        
        for i in range(n):
            p = random.random()
            if p < 1/n:
                y.x[i] = 0 if y.x[i]== 1 else 1
                
        return y  


