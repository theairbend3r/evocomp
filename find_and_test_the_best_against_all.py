import sys
import os
import pathlib
import numpy as np
import pandas as pd

from params import tuned_params
from demo_controller import player_controller

sys.path.insert(0, "evoman")
from environment import Environment

# run experiement headless.
os.environ["SDL_VIDEODRIVER"] = "dummy"

log_folder = pathlib.Path("./results_of_best_against_all")
if not log_folder.is_dir():
    log_folder.mkdir(parents=True, exist_ok=True)

# create evoman environment
EVOMAN_ENV_TEST = Environment(
    experiment_name="results_of_best_against_all",
    playermode="ai",
    player_controller=player_controller(10),
    enemymode="static",
    level=2,
    speed="fastest",
)


# mutation = tuned_params["mutation"]
# num_generations = tuned_params["num_generations"]
# population_size = tuned_params["population_size"]

# test_results_df = pd.read_csv("./test_results.txt", delimiter=" ")
# max_idx_per_enemy = test_results_df.groupby(["crossover", "enemies"])[
#     "fitness"
# ].idxmax()
# best_results_df = test_results_df.loc[max_idx_per_enemy]

with open(
    "./results_of_best_against_all/test_results_all_enemies.txt", "a"
) as f:
    f.write("\nexperiment_name fitness gain game_time enemy")

# best_experiment_name = find_best_experiment()
# run_best_solution_5times()
# "./results_of_best_against_all/test_results_boxplots_all_enemies_BEST_solutions.txt",
# print(test_df)
# test_df["is_gain_positive"] = test_df["gain"] > 0
# experiment_name_list = list(test_df["experiment_name"].unique())
# for exp in experiment_name_list:
#     exp_df = test_df[test_df["experiment_name"] == exp]
#     num_gain_positive = exp_df["is_gain_positive"].sum()
#     print(exp, num_gain_positive)

# for i in range(len(best_results_df)):
    # experiment_run = best_results_df.iloc[i, :]["run"]
    # experiment_crossover = best_results_df.iloc[i, :]["crossover"]
    # enemies = best_results_df.iloc[i, :]["enemies"]

best_experiment_name = "crossover_onepoint__mutation_0.2__enemies_28__population_100__generations_30__run_3/"
run = best_experiment_name.split('__')[-1].split('_')[-1]
mutation = best_experiment_name.split('__')[1].split('_')[-1]
crossover = best_experiment_name.split('__')[0].split('_')[-1]
enemies = best_experiment_name.split('__')[2].split('_')[-1]
num_generations = best_experiment_name.split('__')[4].split('_')[-1]
population_size = best_experiment_name.split('__')[3].split('_')[-1]

# experiment_name = f"crossover_{crossover}__mutation_{mutation}__enemies_{enemies}__population_{population_size}__generations_{num_generations}__run_{experiment_run}"
# print(experiment_name)

best_sol = np.loadtxt(best_experiment_name + "/best.txt")

# iterate over enemies 1, 2, ..., 8.
for e in range(1, 9):
    EVOMAN_ENV_TEST.update_parameter("enemies", [e])
    fitness, player_life, enemy_life, game_time = EVOMAN_ENV_TEST.play(best_sol)

    with open(
        "./results_of_best_against_all/test_results_all_enemies.txt",
        "a",
    ) as f:
        f.write(
            f"\n{best_experiment_name} {fitness} {player_life - enemy_life} {game_time} {e}"
        )



# best_experiment_name = "crossover_onepoint__mutation_0.2__enemies_28__population_100__generations_30__run_3/"
# run = best_experiment_name.split('__')[-1].split('_')[-1]
# mutation = best_experiment_name.split('__')[1].split('_')[-1]
# crossover = best_experiment_name.split('__')[0].split('_')[-1]
# enemies = best_experiment_name.split('__')[2].split('_')[-1]
# num_generations = best_experiment_name.split('__')[4].split('_')[-1]
# population_size = best_experiment_name.split('__')[3].split('_')[-1]
#
# with open("./test_results.txt", "a") as f:
#     f.write(
#         "\n\n\n" +
#         "BEST SOLUTION FOR ALL ENEMIES"
#     )
#
# num_runs = 5
#
# for run in range(num_runs):
#     os.system(
#         f"python execute_experiment.py --run {run} --mode test -m {mutation} -c {crossover} -e {' '.join(enemies)} -g {str(num_generations)} -p {str(population_size)}"
#     )
