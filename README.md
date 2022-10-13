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

1. `tune_params.py`: tunes different combinations of parameters in the `params.py` file. Creates dirs with `_run_x/` in the name.
2. `find_best_params_after_tuning.py`: stores the best combination of parameters based on average of best fitness across generations (there is just 1 run for tuning for each combination of parameters). Creates `best_tuned_params.txt`.
3. `run_training.py`: for each combination of run, group of enemies, and crossover, this runs the experiment 10 times. Creates dirs with `_run_{NUM}/` in the name.

> Update `tuned_params` dictionary inside `params.py` manually. The values from the experiment name are stored in `best_tuned_params.txt`.

4. `run_testing.py`: for each combination of run, group of enemies, and crossover, this runs the testing for all enemies 5 times. Creates `test_results_enemy_all.txt`.
5. `find_and_test_the_best_against_all.py`: finds the best solution after testing and runs it against all enemies one by one (in a loop) 5 times. Creates `test_best_against_all.txt`.
6. `create_plots.py`: creates .png plots.
