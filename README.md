# Evolutionary Computing

## Installation

Install in a virtual environment (miniconda/anaconda is advised.)

Create and environment.

```
conda create --name evocomp python=3
```

Install relevant packages.

```
pip install tmx numpy pygame
```

Clone repository. The `evoman/` folder is borrowed from @karinemiras' [evoman_framework](https://github.com/karinemiras/evoman_framework).

```
git clone https://github.com/theairbend3r/evocomp.git
```

## Usage

### Run a single experiment

```
python execute_experiment.py
```

### Tune hyperparameters (grid search)

```
python tune_params.py
```

This will run a grid search based on parameters in the `params.py` file. The grid search uses multiprocessing.

## Steps To Get Results

1. `tune_params.py`
2. `find best params list for each enemy group`
3. `run_training.py` (repeat experiment with best params 10x4 for each combination)
4. `run_testing.py` (test the best solution 5x)
5. `test_the_best_against_the_rest.py` (test the best solution against all enemies one by one)
6. `create_plots.py`
