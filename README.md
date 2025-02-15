# Evolutionary Multi-Objective Optimization

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## Overview

This repository contains implementations of various evolutionary multi-objective optimization algorithms, focusing on the NSGA-II algorithm and its variants. The code is designed to solve benchmark problems like LOTZ and 4LOTZ, providing tools to analyze algorithm performance and visualize results.

## Repository Structure

Here's an overview of the key files and their functionalities:

- **`lotz.py`**: Defines the Leading Ones Trailing Zeros (LOTZ) problem, a bi-objective optimization benchmark.
- **`mlotz.py`**: Extends LOTZ to the multi-objective case, allowing for more than two objectives.
- **`individual.py`**: Represents an individual in the population, encapsulating its genetic representation and associated objective values.
- **`objective_value.py`**: Contains classes and methods to evaluate and store objective values for individuals.
- **`nsga_ii.py`**: Implements the classic NSGA-II algorithm, including selection, crossover, and mutation operators.
- **`nsga_ii_modified.py`**: Introduces a modified version of NSGA-II with dynamic crowding distance updates upon the removal of individuals.
- **`nsga_ii_optimized.py`**: Offers an optimized implementation of NSGA-II, focusing on computational efficiency and scalability.
- **`binary_heap.py`**: Provides a binary heap data structure, utilized for efficient priority queue operations within the algorithms.
- **`plot_runs.py`**: Contains functions to visualize the performance of algorithms over multiple runs, including Pareto front coverage plots.
- **`run_nsga_ii.py`**: Serves as the main script to execute the NSGA-II algorithm on specified problems with predefined parameters.
- **`test.py`**: Includes unit tests to verify the correctness of various components in the codebase.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed. It's recommended to use a virtual environment to manage dependencies.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/lhzuin/Evolutionary-Multi-Objective-Optimization.git
   cd Evolutionary-Multi-Objective-Optimization
   ```

2. **Install required packages**:

   While there's no `requirements.txt` provided, the primary dependencies include:

   - `numpy`
   - `matplotlib`

   Install them using pip:

   ```bash
   pip install numpy matplotlib
   ```

## Running the Code

### Executing NSGA-II on LOTZ

To run the classic NSGA-II algorithm on the LOTZ problem:

```bash
python run_nsga_ii.py
```

This script uses default parameters defined within. To customize parameters such as population size, number of generations, or problem size, modify the corresponding variables in `run_nsga_ii.py`.

### Running Modified NSGA-II

For the modified version with dynamic crowding distance updates:

```bash
python nsga_ii_modified.py
```

Ensure that the problem definition and parameters align with your experimental setup.

## Reproducing Results

To reproduce the results presented in our study:

1. **Run Multiple Trials**:

   Execute the desired algorithm script multiple times with different random seeds to gather sufficient data for statistical analysis.

2. **Plot Results**:

   Use the `plot_runs.py` script to visualize the performance:

   ```bash
   python plot_runs.py
   ```

   This will generate plots illustrating metrics like Pareto front coverage over generations.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Note: For detailed explanations of the algorithms and theoretical background, refer to the accompanying report or relevant literature.*
