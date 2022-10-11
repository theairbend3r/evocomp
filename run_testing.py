import os
import time
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--crossover", help="Crossover method e.g. uniform80")
args = parser.parse_args()

crossover = args.crossover
enemies = [[2, 8], [5, 6, 7]]
mutation = 0.2
num_generations = 3
population_size = 5

start_time = time.time()
num_runs = 5

paths = pathlib.Path("./")
for d in paths.iterdir():
    if (
        d.is_dir()
        and d.name.find("crossover") != -1
        and d.name.split("__")[-1][-1].isdigit()
    ):
        loop_run = d.name.split("__")[-1][-1]

        for run in range(num_runs):
            for enemy_group in enemies:
                os.system(
                    f"python execute_experiment.py --run {loop_run} --mode test -m {mutation} -c {crossover} -e {' '.join([str(e) for e in enemy_group])} -g {str(num_generations)} -p {str(population_size)}"
                )
end_time = time.time()

print(
    f"Time taken for testing {num_runs * len(enemies)} ({num_runs} runs for {len(enemies)} enemies) = {end_time - start_time} seconds"
)
