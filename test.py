from individual import Individual
from objective_value import ObjectiveValueConstructor, ObjectiveValueConstructorFromFunction
from nsga_ii import NSGA_II
from mlotz import LOTZ, mLOTZ, mLOTZConstructor
from typing import List
x = Individual([1, 1, 0,  0, 1, 0, 0, 0], 8)
lotz = LOTZ(x)
print(lotz[1])
print(lotz[2])

mlotz = mLOTZ(4, len(x), x)
print(mlotz[1])
print(mlotz[2])
print(mlotz[3])
print(mlotz[4])



def binary_list_to_int(binary_list: List[int]) -> int:
    """
    Convert a list of zeros and ones into its integer value.

    Parameters:
        binary_list (List[int]): A list containing only 0 and 1.

    Returns:
        int: The integer value represented by the binary list.

    Raises:
        ValueError: If the list contains elements other than 0 or 1.
    """
    result = 0
    for bit in binary_list:
        if bit not in (0, 1):
            raise ValueError("List must contain only 0 and 1.")
        result = result * 2 + bit
    result = float(result)
    return [result, result]

f = ObjectiveValueConstructorFromFunction(2, binary_list_to_int)
population = [Individual([1, 1, 0,  0, 1, 0, 0, 0], 8), Individual([1, 0, 0,  0, 1, 0, 0, 0], 8), Individual([1, 1, 0,  1, 1, 0, 0, 1], 8), Individual([0, 1, 1,  0, 0, 0, 0, 1], 8)]
nsga = NSGA_II(f)
ordered = nsga.non_dominated_sorting(population)
for rank in ordered:
    print([obj for obj in rank])


print(nsga.crowding_distance(population))

print(nsga.mutation(population[0]))
print("****** MLOTZ**********")
mlotz_const = mLOTZConstructor(4, 8)
for x in population:
    print(mlotz_const(x))
nsga2 = NSGA_II(mlotz_const)
ordered = nsga2.non_dominated_sorting(population)
for rank in ordered:
    print([obj for obj in rank])


print(nsga2.crowding_distance(population))
