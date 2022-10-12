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


mutation = tuned_params["mutation"]
num_generations = tuned_params["num_generations"]
population_size = tuned_params["population_size"]

test_results_df = pd.read_csv("./test_results.txt", delimiter=" ")
max_idx_per_enemy = test_results_df.groupby(["crossover", "enemies"])[
    "fitness"
].idxmax()
best_results_df = test_results_df.loc[max_idx_per_enemy]
print(best_results_df)

with open("./test_results_boxplots_all_enemies.txt", "a") as f:
    f.write("\nexperiment_name fitness player_life enemy_life game_time enemy")

for i in range(len(best_results_df)):
    experiment_run = best_results_df.iloc[i, :]["run"]
    experiment_crossover = best_results_df.iloc[i, :]["crossover"]
    enemies = best_results_df.iloc[i, :]["enemies"]

    experiment_name = f"crossover_{experiment_crossover}__mutation_{mutation}__enemies_{enemies}__population_{population_size}__generations_{num_generations}__run_{experiment_run}"
    print(experiment_name)

    best_sol = np.loadtxt(experiment_name + "/best.txt")

    # iterate over enemies 1, 2, ..., 8.
    for e in range(1, 9):
        EVOMAN_ENV_TEST.update_parameter("enemies", [e])
        fitness, player_life, enemy_life, game_time = EVOMAN_ENV_TEST.play(best_sol)

        with open("./test_results_boxplots_all_enemies.txt", "a") as f:
            f.write(
                f"\n{experiment_name} {fitness} {player_life} {enemy_life} {game_time} {e}"
            )
