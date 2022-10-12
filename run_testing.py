import os
import time
import argparse
import pathlib
from params import tuned_params

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--crossover", help="Crossover method e.g. uniform80")
args = parser.parse_args()

crossover = args.crossover
mutation = tuned_params["mutation"]
num_generations = tuned_params["num_generations"]
population_size = tuned_params["population_size"]

start_time = time.time()
num_runs = 5

with open("./test_results_boxplots_groups_enemies.txt", "a") as f:
    f.write("\nfitness crossover run enemies")

paths = pathlib.Path("./")
for d in paths.iterdir():
    if (
        d.is_dir()
        and d.name.find("crossover") != -1
        and d.name.split("__")[-1][-1].isdigit()
    ):
        loop_run = d.name.split("__")[-1][-1]

        for run in range(num_runs):
            os.system(
                f"python execute_experiment.py --run {loop_run} --mode test -m {mutation} -c {crossover} -e {' '.join(d.name.split('__')[2].split('_')[-1])} -g {str(num_generations)} -p {str(population_size)}"
            )
end_time = time.time()

print(f"Time taken for testing = {end_time - start_time} seconds")
