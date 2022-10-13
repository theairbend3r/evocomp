import os
import time
import itertools
from collections import namedtuple

from params import params


if __name__ == "__main__":
    start_time = time.time()
    params_combo = list(itertools.product(*[v for v in params.values()]))

    ParamRow = namedtuple(
        "ParamRow",
        [
            "enemies",
            "population_size",
            "num_generations",
            "mutation",
            "crossover",
        ],
    )

    params_combo = [ParamRow(*pc) for pc in params_combo]

    params_combo_list = []
    for pc in params_combo:
        # call execute_experiment
        os.system(
            f"python execute_experiment.py --run x --mode train -m {pc.mutation} -c {pc.crossover} -e {' '.join([str(e) for e in pc.enemies])} -g {str(pc.num_generations)} -p {str(pc.population_size)}"
        )

    end_time = time.time()
    print(f"\n\nTime taken to tune all params = {end_time - start_time}")
