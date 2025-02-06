from typing import List
from objective_value import ObjectiveValue


class NSGA_II:
    def __init__(self):
        pass
    @staticmethod
    def non_dominated_sorting(population: List[ObjectiveValue]) -> List[List[ObjectiveValue]]:
        n = len(population)
        dominance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dominance_matrix[i][j] = int(ObjectiveValue.strictly_dominates(population[i], population[j]))
                    # Column j will store the elements that strictly dominate the jth element from the population

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



                


    def crowding_distance(self):
        pass