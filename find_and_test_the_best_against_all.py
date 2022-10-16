from math import inf
import sys
import os
import pathlib
import numpy as np
import pandas as pd

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
test_df["ea"] = [
    x.split("__")[0].split("_")[1].split("-")[0] for x in test_df["experiment_name"]
]
ea_list = list(test_df["ea"].unique())
experiment_name_list = list(test_df["experiment_name"].unique())


# dictionary to compare the EAs
compare_dict_ea_sar = {
    exp: {"num_gain_positive": 0, "avg_gain_value": 0}
    for exp in experiment_name_list
    if exp.split("__")[0].split("_")[1].split("-")[0] == "sar"
}

compare_dict_ea_blend = {
    exp: {"num_gain_positive": 0, "avg_gain_value": 0}
    for exp in experiment_name_list
    if exp.split("__")[0].split("_")[1].split("-")[0] == "blend"
}

compare_dict = {"sar": compare_dict_ea_sar, "blend": compare_dict_ea_blend}

# generate stats to find the best individual per EA
for ea in ea_list:
    for exp in experiment_name_list:
        if exp.split("__")[0].split("_")[1].split("-")[0] == ea:
            exp_df = test_df[test_df["experiment_name"] == exp]
            num_gain_positive = exp_df["is_gain_positive"].sum()
            avg_gain_value = exp_df["gain"].mean()
            compare_dict[ea][exp]["num_gain_positive"] = num_gain_positive
            compare_dict[ea][exp]["avg_gain_value"] = avg_gain_value


# find the best stats and therefore the best per EA
highest_num_gain_positive = {"sar": -inf, "blend": -inf}
count_num_gain_positive = {"sar": 0, "blend": 0}
highest_avg_gain_value = {"sar": -inf, "blend": -inf}
best_experiment_name = {"sar": "", "blend": ""}

for ea in ea_list:
    for exp in compare_dict[ea]:
        if compare_dict[ea][exp]["num_gain_positive"] == highest_num_gain_positive[ea]:
            count_num_gain_positive[ea] += 1

        if compare_dict[ea][exp]["num_gain_positive"] > highest_num_gain_positive[ea]:
            highest_num_gain_positive[ea] = compare_dict[ea][exp]["num_gain_positive"]
            best_experiment_name[ea] = exp

    if count_num_gain_positive[ea] > 0:
        for exp in compare_dict[ea]:
            if compare_dict[ea][exp]["avg_gain_value"] > highest_avg_gain_value[ea]:
                highest_avg_gain_value[ea] = compare_dict[ea][exp]["avg_gain_value"]
                best_experiment_name[ea] = exp

# best individual per EA
print(best_experiment_name)

for ea in ea_list:
    if best_experiment_name[ea] is None:
        raise ValueError(f"No best experiment found for {ea}.")

# test best EA against all enemies 5 times
for ea in ea_list:
    run = best_experiment_name[ea].split("__")[-1].split("_")[-1]
    mutation = best_experiment_name[ea].split("__")[1].split("_")[-1]
    crossover = best_experiment_name[ea].split("__")[0].split("_")[-1]
    enemies = best_experiment_name[ea].split("__")[2].split("_")[-1]
    num_generations = best_experiment_name[ea].split("__")[4].split("_")[-1]
    population_size = best_experiment_name[ea].split("__")[3].split("_")[-1]

    best_sol = np.loadtxt(best_experiment_name[ea] + "/best.txt")

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
                    f"\n{best_experiment_name[ea]} {fitness} {player_life - enemy_life} {game_time} {e} {i}"
                )
