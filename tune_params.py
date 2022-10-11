import os
import itertools
from functools import partial
from collections import namedtuple

from evolution import (
    create_offspring,
    select_individuals_for_next_generation,
    fitness,
)
from params import params


if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("tag", help="Tag for experiment e.g. ea1")
    # args = parser.parse_args()

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
        # name experiment/run to save logs.
        experiment_name = f"enemy_{''.join([str(e) for e in pc.enemies])}-population_size_{pc.population_size}-num_hidden_neurons_{pc.num_hidden_neurons}-num_generations_{pc.num_generations}-mutation_{pc.mutation}-crossover_{pc.crossover}-selection_{pc.selection}-fitness_{pc.fitness}"
        # experiment_name = f"enemy_{''.join([str(e) for e in pc.enemies])}"
        params_combo_list.append(
            (
                pc.enemies,
                pc.population_size,
                pc.num_hidden_neurons,
                pc.num_generations,
                partial(
                    create_offspring,
                    mutation_method=pc.mutation,
                    crossover_method=pc.crossover,
                ),
                partial(select_individuals_for_next_generation, method=pc.selection),
                partial(fitness, method=pc.fitness),
            )
        )

        os.system(
            f"python execute_experiment.py -m train -c {pc.crossover} -r run -e {' '.join([str(e) for e in pc.enemies])}",
        )
