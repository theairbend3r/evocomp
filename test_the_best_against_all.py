import pandas as pd
import os
import numpy as np

mutation = 0.2
num_generations = 3
population_size = 5

test_results_df = pd.read_csv('./test_results.txt', delimiter=' ')

max_idx_per_enemy = test_results_df.groupby('enemies')['fitness'].idxmax()
best_results_df = test_results_df.loc[max_idx_per_enemy]
# print(best_results_df.tolist())

for l in list:
    os.system(
        f"python execute_experiment.py --run {l[2]} --mode test -m {mutation} -c {l[1]} -e {' '.join(l[3])} -g {str(num_generations)} -p {str(population_size)}"
    )
    