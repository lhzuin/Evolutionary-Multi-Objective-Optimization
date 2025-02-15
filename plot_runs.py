from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import statistics
import time
import datetime
import os


from nsga_ii_modified import NSGA_II_Modified
from nsga_ii_optimized import NSGA_II_Optimized
from mlotz import mLOTZConstructor


def run_nsgaii_experiment(n: int, m: int, n_runs: int, population_size: int, model, seed: Optional[int] = None) -> Tuple[float, float, int, int]:
    """
    Run NSGA-II for a given number of independent experiments for a problem of size n with m objectives.
    
    For each of the 'n_runs' experiments, the function instantiates an NSGA-II model (using the provided model class and 
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
    for r in range(n_runs):
        # Use a different seed per run
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


def plot_pareto_coverage(n_runs: int, m: int, n_values: List[int], models: List[object], seed: Optional[int] = None) -> None:
    """
    Plot the average Pareto front coverage and running time achieved by NSGA-II for a range of problem sizes.
    
    For each problem size in n_values, this function runs n_runs independent experiments for each model in models 
    (using the LOTZ or 4LOTZ benchmark with m objectives) and computes:
      - the average ratio of Pareto front points covered by the final populations, and
      - the average running time of the algorithm.
      
    It then produces two plots:
      1. Average Pareto front coverage vs problem size, with a separate curve for each model.
      2. Average running time vs problem size, with a separate curve for each model.
    
    Parameters:
        n_runs (int): The number of independent experiments to perform per problem size.
        m (int): The number of objectives. For LOTZ, m must be an even number.
        n_values (List[int]): A list of problem sizes (bit-string lengths) to test.
        models (List[object]): A list of NSGA-II model classes (or callables) to evaluate. Each model is expected to have 
                               an attribute 'name' to be used in the legend.
        seed (Optional[int]): An optional random seed for reproducibility. If provided, each experiment will use a modified seed.
    
    Returns:
        None. This function displays two plots.
    """
    # Get a timestamp for saving files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dictionaries to store results for each model
    coverage_results = {}
    time_results = {}
    
    # Loop over each model and problem size, compute metrics, and store the results.
    for model in models:
        pareto_ratio_avgs = []
        avg_time_avgs = []
        for n in n_values:
            n_prime = int(2 * n / m)
            M = (n_prime + 1) ** (m // 2)
            N = 4 * M
            avg_iterations, avg_time, avg_pareto_front_ratio = run_nsgaii_experiment(n, m, n_runs, N, model, seed)
            print(f"LOTZ (n={n}): Average iterations = {avg_iterations:.2f}, Average time = {avg_time:.2f} sec, Pareto Cover Mean Rate = {100*avg_pareto_front_ratio:.2f}%")
            pareto_ratio_avgs.append(avg_pareto_front_ratio)
            avg_time_avgs.append(avg_time)
        coverage_results[model.name] = pareto_ratio_avgs
        time_results[model.name] = avg_time_avgs

    folder = "images"
    os.makedirs(folder, exist_ok=True)

    # ------------------------ Plot Average Pareto Front Coverage ------------------------
    plt.figure(figsize=(8, 6))
    title_coverage = f"Average Coverage of Pareto Front for {m}LOTZ ({n_runs} runs)"
    for model_name, coverage in coverage_results.items():
        plt.plot(n_values, coverage, "--o", label=model_name)
    plt.legend()
    plt.title(title_coverage)
    plt.xlabel("Problem size (n)")
    plt.ylabel("Average Pareto Front Coverage")
    coverage_filename = folder + "/" + f"pareto_coverage_m{m}_{timestamp}.png"
    plt.savefig(coverage_filename, format="png", dpi=300)
    plt.show(block=False)
    plt.close()
    
    # ------------------------ Plot Average Running Time ------------------------
    plt.figure(figsize=(8, 6))
    title_time = f"Average Running Time for {m}LOTZ ({n_runs} runs)"
    for model_name, times in time_results.items():
        plt.plot(n_values, times, "--o", label=model_name)
    plt.legend()
    plt.title(title_time)
    plt.xlabel("Problem size (n)")
    plt.ylabel("Average Running Time (s)")
    time_filename = folder + "/" + f"average_running_time_m{m}_{timestamp}.png"
    plt.savefig(time_filename, format="png", dpi=300)
    plt.show(block=False)
    plt.close()


            
if __name__ == "__main__":
    n_runs = 50
    m_values = [2, 4]
    n_values = [[4, 8, 12, 16, 20, 24, 32, 40], [4, 8, 12, 16, 20]]

    models = [NSGA_II_Optimized, NSGA_II_Modified]
    
    for i in range(len(m_values)):
        plot_pareto_coverage(n_runs, m_values[i], n_values[i], models, seed=42)