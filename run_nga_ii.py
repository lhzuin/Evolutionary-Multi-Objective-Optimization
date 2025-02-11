import statistics
import time
from typing import Optional, Tuple
from mlotz import mLOTZConstructor
from nsga_ii import NSGA_II


def run_nsgaii_experiment(n: int, m: int, runs: int, population_size: int, seed: Optional[int] = None) -> Tuple[float, float, int, int]:
    """
    Runs NSGA-II for 'runs' independent experiments for problem size n and m objectives.
    Returns a tuple:
      (average iterations, average time (sec), number of successful runs, total runs)
    """
    iteration_counts = []
    durations = []
    successes = 0
    for r in range(runs):
        # Use a different seed per run if desired.
        run_seed = seed + r if seed is not None else None

        # Instantiate NSGA-II with the proper objective function.
        nsgaii = NSGA_II(mLOTZConstructor(m, n))
        start = time.time()
        iterations, final_pop = nsgaii.run(population_size, n, m, seed=run_seed)
        end = time.time()
        duration = end - start
        iteration_counts.append(iterations)
        durations.append(duration)
        success = nsgaii.population_covers_pareto_front(final_pop, m, n)
        successes += int(success)
        print(f"Run {r+1}: iterations = {iterations}, time = {duration:.2f} sec, success = {success}")
    avg_iterations = statistics.mean(iteration_counts)
    avg_time = statistics.mean(durations)
    return avg_iterations, avg_time, successes, runs

def test_nsgaii():
    runs = 5  # perform 5 runs per setting 
    
    n_values = [4, 8, 16, 40, 80]
    
    for n in n_values:
        print(f"\n===== Testing LOTZ (m=2) with n = {n} =====")
        m = 2
        n_prime = int(2 * n / m)
        M = (n_prime + 1) ** (m // 2)
        N = 4 * M
        avg_it, avg_time, succ, tot = run_nsgaii_experiment(n, m, runs, N, seed=42)
        print(f"LOTZ (n={n}): Average iterations = {avg_it:.2f}, Average time = {avg_time:.2f} sec, success = {succ}/{tot}")
        
        print(f"\n===== Testing 4LOTZ (m=4) with n = {n} =====")
        m = 4
        n_prime = int(2 * n / m)
        M = (n_prime + 1) ** (m // 2)
        N = 4 * M
        avg_it, avg_time, succ, tot = run_nsgaii_experiment(n, m, runs, M, seed=42)
        print(f"4LOTZ (n={n}): Average iterations = {avg_it:.2f}, Average time = {avg_time:.2f} sec, success = {succ}/{tot}")

if __name__ == "__main__":
    test_nsgaii()