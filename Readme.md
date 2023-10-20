# SPREAD

This repository includes a CLI program designed to run experiments and generate evaluation plots for an academic paper entitled "SPREAD: Towards Optimal GPU Sharing by Job Spreading for Lightweight Deep Learning Training Jobs."

The tool facilitates placement solving, partitioning and distribution, as well as trace experiments, and other related functions. With this program, executing these tasks is easy and straightforward. Furthermore, it offers multiple options for plotting evaluation figures based on the collected data.



## Prerequisite

Before running SPREAD, make sure you have the following prerequisites installed:

1. Linux/MacOS-based computer.
1. [Gurobi](https://www.gurobi.com/) - a mathematical optimization solver, and a valid license to use it.
2. Python environment - We recommend using [Anaconda](https://www.anaconda.com/products/individual) as it is a popular
   distribution for scientific computing and data analysis.

To create a conda environment, use the following command:

``` shell
conda env create -n SPREAD -f environment.yml
```

Note that we do not provide a docker image to run SPREAD since the academic license of gurobi is not available in container-based environment.

## Step-by-step Deployment

1. Prepare a Linux/MacOS-based server/PC.
2. Install Gurobi optimizer v10.0.1 and prepare a license at the default location.
3. Install Anaconda.
4. Clone the repository at https://github.com/dos-lab/SPREAD.
5. Enter the root directory of the repository.
6. Create a conda environment named "SPREAD" using the provided "environment.yml" file. (`conda env create -n SPREAD -f environment.yml`)
7. Activate the "SPREAD" environment. (`conda activate SPREAD`)
8. Run the command `python main.py --help` to see further instructions on how to run experiments or draw plots.

## Directory Structure

The directory structure of the repository is as follows:

```
├── results
│   ├── data        # experiment data
│   └── figures     # evaluation figures
├── src             # source code
├── environment.yml # conda environment file
└── main.py         # CLI entrance
```

## Usage

To use SPREAD, navigate to the root folder of the repository and run the following command:

``` shell
python main.py [--FLAG] 
```

Use the appropriate flag to run an experiment or generate an evaluation plot.

## Flags

The following flags are available to run experiments and generate evaluation plots:

- `--run-placement-experiment`: Run solver, partitioning, and distributing experiment and collect data.
- `--run-trace-experiment`: Run trace end-to-end experiment and collect data.
- `--plot-job-profiling-data`: Plot job profiling data figures.
- `--plot-solver-part-dist-eval`: Plot solver, partitioning, and distributing evaluation figures.
- `--plot-trace-eval`: Plot trace end-to-end evaluation figures.
- `--plot-preemption-eval`: Plot preemption evaluation figures.
- `--plot-scalability-eval`: Plot placement latency evaluation figures.
- `--plot-simulator-eval`: Plot simulator evaluation figures.
- `--plot-workloads`: Plot workloads figures.

## Examples

Here are some examples of how to use SPREAD:

To run the solver, partitioning, and distributing experiment and collect data:

``` python
python main.py --run-placement-experiment 
```

To run the trace-based end-to-end experiment and collect data:

``` python
python main.py --run-trace-experiment 
```

To draw job profiling data figures:

``` python
python main.py --plot-job-profiling-data 
```

To draw solver, partitioning, and distributing evaluation figures:

``` python
python main.py --plot-solver-part-dist-eval 
```

To draw trace-based end-to-end evaluation figures:

``` python
python main.py --plot-trace-eval 
```

To draw preemption evaluation figures:

``` python
python main.py --plot-preemption-eval 
```

To draw placement latency figures:

``` python
python main.py --plot-scalability-eval 
```

To draw simulator validation figures:

``` python
python main.py --plot-simulator-eval 
```

To draw workloads figures:

``` python
python main.py --plot-workloads 
```
