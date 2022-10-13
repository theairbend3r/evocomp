import sys
import os
import pathlib
import numpy as np
import pandas as pd
from pprint import pprint

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


with open("./test_best_against_all.txt", "a") as f:
    f.write("\nexperiment_name fitness gain game_time enemy iteration")


test_df = pd.read_csv("./test_results_enemy_all.txt", delimiter=" ")
test_df["is_gain_positive"] = test_df["gain"] > 0
experiment_name_list = list(test_df["experiment_name"].unique())

compare_dict = {
    exp: {"num_gain_positive": 0, "avg_gain_value": 0} for exp in experiment_name_list
}

for exp in experiment_name_list:
    exp_df = test_df[test_df["experiment_name"] == exp]
    num_gain_positive = exp_df["is_gain_positive"].sum()
    avg_gain_value = exp_df["gain"].mean()
    compare_dict[exp]["num_gain_positive"] = num_gain_positive
    compare_dict[exp]["avg_gain_value"] = avg_gain_value

highest_num_gain_positive = -1
count_num_gain_positive = 0

highest_avg_gain_value = 0
best_experiment_name = None

for exp in compare_dict:
    if compare_dict[exp]["num_gain_positive"] == highest_num_gain_positive:
        count_num_gain_positive += 1

    if compare_dict[exp]["num_gain_positive"] > highest_num_gain_positive:
        highest_num_gain_positive = compare_dict[exp]["num_gain_positive"]
        best_experiment_name = exp

if count_num_gain_positive > 0:
    for exp in compare_dict:
        if compare_dict[exp]["avg_gain_value"] > highest_avg_gain_value:
            highest_avg_gain_value = compare_dict[exp]["avg_gain_value"]
            best_experiment_name = exp

print(best_experiment_name)

if best_experiment_name is None:
    raise ValueError("No best experiment found.")

run = best_experiment_name.split("__")[-1].split("_")[-1]
mutation = best_experiment_name.split("__")[1].split("_")[-1]
crossover = best_experiment_name.split("__")[0].split("_")[-1]
enemies = best_experiment_name.split("__")[2].split("_")[-1]
num_generations = best_experiment_name.split("__")[4].split("_")[-1]
population_size = best_experiment_name.split("__")[3].split("_")[-1]


best_sol = np.loadtxt(best_experiment_name + "/best.txt")

# iterate over enemies 1, 2, ..., 8.
for e in range(1, 9):
    for i in range(5):
        EVOMAN_ENV_TEST.update_parameter("enemies", [e])
        fitness, player_life, enemy_life, game_time = EVOMAN_ENV_TEST.play(best_sol)

        with open(
            "./test_best_against_all.txt",
            "a",
        ) as f:
            f.write(
                f"\n{best_experiment_name} {fitness} {player_life - enemy_life} {game_time} {e} {i}"
            )
