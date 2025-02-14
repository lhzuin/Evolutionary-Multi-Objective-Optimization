from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import statistics
import time

from nsga_ii_modified import NSGA_II_Modified
from nsga_ii_optimized import NSGA_II_Optimized
from mlotz import mLOTZConstructor


def run_nsgaii_experiment(n: int, m: int, runs: int, population_size: int, model, seed: Optional[int] = None) -> Tuple[float, float, int, int]:
    """
    Run NSGA-II for a given number of independent experiments for a problem of size n with m objectives.
    
    For each of the 'runs' experiments, the function instantiates an NSGA-II model (using the provided model class and 
    mLOTZConstructor with parameters m and n), runs the algorithm until the termination criterion is met (i.e. the final 
    population covers the Pareto front), and records the number of iterations, runtime (in seconds), and the final ratio 
    of Pareto front points covered. Different runs use seed + run_index if a seed is provided.
    
    Parameters:
        n (int): The problem size (i.e. the length of each individual's bit string).
        m (int): The number of objectives (for the LOTZ benchmark, m must be an even number).
        runs (int): The number of independent experiments to run.
        population_size (int): The number of individuals in the population for each run.
        model: A class or callable that instantiates the NSGA-II algorithm; it is expected to accept an ObjectiveValueConstructor 
               (e.g. mLOTZConstructor) and an optional seed.
        seed (Optional[int]): An optional random seed for reproducibility. If provided, each run will use seed + run_index.
    
    Returns:
        Tuple[float, float, float]: A tuple containing:
            - avg_iterations (float): The average number of iterations executed over all runs.
            - avg_time (float): The average runtime in seconds over all runs.
            - avg_pareto_front_ratio (float): The average ratio of the Pareto front covered in the final populations.
    """
    iteration_counts = []
    durations = []
    pareto_ratios = []
    print(f"\n===== Testing LOTZ (m={m}) with n = {n} =====")
    for r in range(runs):
        # Use a different seed per run if desired.
        run_seed = seed + r if seed is not None else None

        # Instantiate NSGA-II with the proper objective function.
        nsgaii = model(mLOTZConstructor(m, n), seed=run_seed)
        start = time.time()
        iterations, final_pop = nsgaii.run(population_size, n, m)
        end = time.time()
        duration = end - start
        iteration_counts.append(iterations)
        durations.append(duration)
        pareto_ratio = nsgaii.population_covers_pareto_front(final_pop, m, n)
        pareto_ratios.append(pareto_ratio)
        print(f"Run {r+1}: iterations = {iterations}, time = {duration:.2f} sec, success = {pareto_ratio:.3f}")
    avg_iterations = statistics.mean(iteration_counts)
    avg_time = statistics.mean(durations)
    avg_pareto_front_ratio = statistics.mean(pareto_ratios)
    return avg_iterations, avg_time, avg_pareto_front_ratio


def plot_pareto_coverage(n_runs: int, m: int, n_values: List[int], models: List[object], seed: Optional[int] = None):
    """
    Plot the average Pareto front coverage achieved by NSGA-II for a range of problem sizes.
    
    For each problem size in n_values, this function runs n_runs independent experiments for each model in models 
    (using the LOTZ or 4LOTZ benchmark with m objectives) and computes the average ratio of Pareto front points 
    covered by the final populations. It then plots the average Pareto front coverage versus problem size, with a separate 
    curve for each model.
    
    Parameters:
        n_runs (int): The number of independent experiments to perform per problem size.
        m (int): The number of objectives. For LOTZ, m must be an even number.
        n_values (List[int]): A list of problem sizes (bit-string lengths) to test.
        models (List[object]): A list of NSGA-II model classes (or callables) to evaluate. Each model is expected to have 
                               an attribute 'name' to be used in the legend.
        seed (Optional[int]): An optional random seed for reproducibility. If provided, each experiment will use a modified seed.
    
    Returns:
        None. This function displays a plot of the average Pareto front coverage versus problem size for each model.
    """
    title = f"Average Coverage of Pareto Front for {m}LOTZ ({n_runs} runs)"
    for model in models:
        pareto_ratio_avgs = []
        for n in n_values:
            n_prime = int(2 * n / m)
            M = (n_prime + 1) ** (m // 2)
            N = 4 * M
            avg_iterations, avg_time, avg_pareto_front_ratio = run_nsgaii_experiment(n, m, n_runs, N, model, seed)
            pareto_ratio_avgs.append(avg_pareto_front_ratio)
        plt.plot(n_values, pareto_ratio_avgs, "--o", label=model.name)
    plt.legend()
    plt.title(title)
    plt.xlabel("Problem size (n)")
    plt.ylabel("Average Pareto Front Coverage")
    plt.show(block=False)


            
if __name__ == "__main__":
    n_runs = 20
    m_values = [2, 4]
    n_values = [[4, 8, 16, 24, 32, 40, 50, 60, 70, 80, 90, 100], [4, 8, 16, 24, 32]]
    n_values = [[4, 8, 16, 24], [4, 8, 16]]
    models = [NSGA_II_Optimized, NSGA_II_Modified]
    
    for i in range(2):
        plot_pareto_coverage(n_runs, m_values[i], n_values[i], models)