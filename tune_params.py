import itertools
from functools import partial
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from evolution import (
    mutation,
    crossover,
    selection,
    fitness,
)
from params import params
from execute_experiment import execute_experiment


if __name__ == "__main__":

    params_combo = list(itertools.product(*[v for v in params.values()]))

    ParamRow = namedtuple(
        "ParamRow",
        [
            "enemies",
            "population_size",
            "num_hidden_neurons",
            "representation_size",
            "representation_size_upper_limit",
            "representation_size_lower_limit",
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
                pc.enemies,
                pc.population_size,
                pc.num_hidden_neurons,
                pc.representation_size,
                pc.representation_size_upper_limit,
                pc.representation_size_lower_limit,
                pc.num_generations,
                partial(mutation, method=pc.mutation),
                partial(crossover, method=pc.crossover),
                partial(selection, method=pc.selection),
                partial(fitness, method=pc.fitness),
            )
        )

    with Pool(processes=4) as pool:
        pool.starmap(execute_experiment, params_combo_list)

    # with ProcessPoolExecutor() as executor:
    #     for r in executor.map(execute_experiment, params_combo_list):
    #         print(r)

    # ParamRow = namedtuple(
    #     "ParamRow",
    #     [
    #         "enemies",
    #         "population_size",
    #         "num_hidden_neurons",
    #         "representation_size",
    #         "representation_size_upper_limit",
    #         "representation_size_lower_limit",
    #         "num_generations",
    #         "mutation",
    #         "crossover",
    #         "selection",
    #         "fitness",
    #     ],
    # )

    # params_combo = [ParamRow(*pc) for pc in params_combo]

    # for pc in params_combo:
    #     execute_experiment(
    #         enemies=pc.enemies,
    #         population_size=pc.population_size,
    #         num_hidden_neurons=pc.num_hidden_neurons,
    #         representation_size=pc.representation_size,
    #         representation_size_upper_limit=pc.representation_size_upper_limit,
    #         representation_size_lower_limit=pc.representation_size_upper_limit,
    #         num_generations=pc.num_generations,
    #         mutation=partial(mutation, method=pc.mutation),
    #         crossover=partial(crossover, method=pc.crossover),
    #         selection=partial(selection, method=pc.selection),
    #         fitness=partial(fitness, method=pc.fitness),
    #     )
