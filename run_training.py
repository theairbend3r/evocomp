import os
import time
from params import params, tuned_params


crossover = tuned_params["crossover"]
enemies = params["enemies"]
mutation = tuned_params["mutation"]
num_generations = tuned_params["num_generations"]
population_size = tuned_params["population_size"]

start_time = time.time()
num_runs = 10
for run in range(num_runs):
    for enemy_group in enemies:
        os.system(
            f"python execute_experiment.py --run {run} --mode train -m {mutation} -c {crossover} -e {' '.join([str(e) for e in enemy_group])} -g {str(num_generations)} -p {str(population_size)}"
        )
end_time = time.time()

print(
    f"Time taken for {num_runs * len(enemies)} run ({num_runs} runs for {len(enemies)} enemies) = {end_time - start_time} seconds"
)
