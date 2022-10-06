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
