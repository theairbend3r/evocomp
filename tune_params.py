import itertools
from functools import partial
from multiprocessing import Pool
from collections import namedtuple

from evolution import (
    mutate,
    crossover,
    select_for_next_generation,
    fitness,
)
from params import params
from execute_experiment import execute_experiment


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="Tag for experiment e.g. ea1")
    args = parser.parse_args()

    params_combo = list(itertools.product(*[v for v in params.values()]))

    ParamRow = namedtuple(
        "ParamRow",
        [
            "enemies",
            "population_size",
            "num_hidden_neurons",
            "num_generations",
            "mutation",
            "crossover",
            "selection",
            "fitness",
        ],
    )

    params_combo = [ParamRow(*pc) for pc in params_combo]

    params_combo_list = []
    for pc in params_combo:
        params_combo_list.append(
            (
                args.tag,
                pc.enemies,
                pc.population_size,
                pc.num_hidden_neurons,
                pc.num_generations,
                partial(mutate, method=pc.mutation),
                partial(crossover, method=pc.crossover),
                partial(select_for_next_generation, method=pc.selection),
                partial(fitness, method=pc.fitness),
            )
        )

    with Pool(processes=4) as pool:
        pool.starmap(execute_experiment, params_combo_list)
