import os
import time
import pathlib
from params import params, tuned_params


crossover = tuned_params["crossover"]
enemies = params["enemies"]
mutation = tuned_params["mutation"]
num_generations = tuned_params["num_generations"]
population_size = tuned_params["population_size"]

start_time = time.time()
num_runs = 5

with open("./test_results_enemy_all.txt", "a") as f:
    f.write("\nfitness gain experiment_name")

paths = pathlib.Path("./")
for d in paths.iterdir():
    if (
        d.is_dir()
        and d.name.find("crossover") != -1
        and d.name.split("__")[-1][-1].isdigit()
    ):
        loop_run = d.name.split("__")[-1][-1]

        for run in range(num_runs):
            for c in crossover:
                os.system(
                    f"python execute_experiment.py --run {loop_run} --mode test -m {mutation} -c {c} -e {' '.join(d.name.split('__')[2].split('_')[-1])} -g {str(num_generations)} -p {str(population_size)}"
                )
end_time = time.time()

print(f"Time taken for testing = {end_time - start_time} seconds")
