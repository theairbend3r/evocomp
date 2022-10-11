import os
import sys
import time
import pathlib
import numpy as np
from functools import partial
from typing import Callable
from multiprocessing import Pool
import argparse

from demo_controller import player_controller
from evolution import (
    add_offspring_to_population,
    doomsday_protocol,
    generate_population,
    normalise_array,
    save_results,
    track_solution_improvement,
)

sys.path.insert(0, "evoman")
from environment import Environment

np.random.seed(69)

# ===========
# run experiement headless.
os.environ["SDL_VIDEODRIVER"] = "dummy"

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--experiment_name", help="Name for experiment e.g. ea1")
parser.add_argument("-e", "--enemies", nargs="+", help="Enemies (e.g. [1, 2])")
args = parser.parse_args()

experiment_name = args.experiment_name
enemies = args.enemies

# create directory, if it does not already exist, to store runs.
log_folder = pathlib.Path(f"./{experiment_name}")
if not log_folder.is_dir():
    log_folder.mkdir(parents=True, exist_ok=True)

# create evoman environment
EVOMAN_ENV = Environment(
    experiment_name=experiment_name,
    enemies=enemies,
    multiplemode="yes",
    playermode="ai",
    player_controller=player_controller(10),
    enemymode="static",
    level=2,
    speed="fastest",
)


def simulate_game(population_row: np.ndarray) -> float:
    """Simulate game between bot and enemy. Returns fitness scores
    for each individual in the population array.

    Parameters
    ----------
    env : Evoman environment

    population_row : np.ndarray


    Returns
    -------
    float

    """
    # env.play() returns: fitness, player_life, enemy_life, time_elapsed
    fitness, _, _, _ = EVOMAN_ENV.play(pcont=population_row)
    return fitness


def evaluate(population: np.ndarray) -> np.ndarray:
    """Evaluate each row of the population array. Returns an (n, ) array
    with fitness scores for each of n generations.

    Parameters
    ----------
    env : Evoman environment

    population : np.ndarray


    Returns
    -------
    np.ndarray

    """
    params_combo_list = [row for row in population]

    with Pool(processes=4) as pool:
        result = pool.map(simulate_game, params_combo_list)

    return np.array(result)


# =========


def execute_experiment(
    experiment_name: str,
    enemies: list,
    population_size: int,
    num_hidden_neurons: int,
    num_generations: int,
    create_offspring: Callable,
    select_individuals_for_next_generation: Callable,
    fitness: Callable,
):
    # # run experiement headless.
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    # name experiment/run to save logs.
    # experiment_name = f"{experiment_tag.upper()}-enemy_{''.join([str(e) for e in enemies])}-population_size_{population_size}-num_hidden_neurons_{num_hidden_neurons}-num_generations_{num_generations}-mutation_{mutate.keywords['method']}-crossover_{crossover.keywords['method']}-selection_{select_individuals_for_next_generation.keywords['method']}-fitness_{fitness.keywords['method']}"

    # print(experiment_name)

    # # create directory, if it does not already exist, to store runs.
    # log_folder = pathlib.Path(f"./{experiment_name}")
    # if not log_folder.is_dir():
    #     log_folder.mkdir(parents=True, exist_ok=True)
    #
    # # create evoman environment
    # EVOMAN_ENV = Environment(
    #     experiment_name=experiment_name,
    #     enemies=enemies,
    #     multiplemode="yes",
    #     playermode="ai",
    #     player_controller=player_controller(num_hidden_neurons),
    #     enemymode="static",
    #     level=2,
    #     speed="fastest",
    # )

    # check environment state
    EVOMAN_ENV.state_to_log()

    # initialise population
    population = generate_population(
        population_size=population_size,
        num_sensors=EVOMAN_ENV.get_num_sensors(),
        num_hidden_neurons=num_hidden_neurons,
    )

    # get fitness of the 0th generation of the population
    population_fitness = evaluate(population=population)

    # track solution stats
    alltime_best_individual = np.argmax(population_fitness)
    alltime_best_solution = population_fitness[alltime_best_individual]
    population_fitness_mean = np.mean(population_fitness)
    population_fitness_std = np.std(population_fitness)

    # update solution (population fitness)
    EVOMAN_ENV.update_solutions([population, population_fitness])

    # write initial solution to file
    save_results(
        num_gen=0,
        experiment_name=experiment_name,
        alltime_best_solution=alltime_best_solution,
        population_fitness_mean=population_fitness_mean,
        population_fitness_std=population_fitness_std,
    )

    # track solution improvement
    solution_not_improved_count = 0

    experiment_start_time = time.time()

    # start evolution
    for gen in range(1, num_generations + 1):
        generation_start_time = time.time()

        # generate offsprings via crossover+mutation between parents
        offspring = create_offspring(population=population)
        print(population.shape)
        print(offspring.shape)

        # evaluate the fitness of the offsprings
        offspring_fitness = evaluate(population=offspring)

        # add offsprings to the population
        population, population_fitness = add_offspring_to_population(
            population=population,
            population_fitness=population_fitness,
            offspring=offspring,
            offspring_fitness=offspring_fitness,
        )

        current_best_individual = np.argmax(population_fitness)
        current_best_solution = population_fitness[current_best_individual]
        current_best_solution_normalised = normalise_array(current_best_solution)

        # select a subset from the new population
        population, population_fitness = select_individuals_for_next_generation(
            population=population,
            population_fitness=population_fitness,
            population_size=population_size,
            method="idk",
        )

        # track solution for doomsday protocol
        solution_not_improved_count = track_solution_improvement(
            alltime_best_solution=alltime_best_solution,
            current_best_solution=current_best_solution,
            solution_not_improved_count=solution_not_improved_count,
        )

        if solution_not_improved_count == 10:
            with open(experiment_name + "/results.txt", "a") as f:
                f.write("\ndoomsday")

            population, population_fitness = doomsday_protocol(
                population=population, population_fitness=population_fitness
            )

        # save scores
        save_results(
            num_gen=gen,
            experiment_name=experiment_name,
            alltime_best_solution=alltime_best_solution,
            population_fitness_mean=population_fitness_mean,
            population_fitness_std=population_fitness_std,
        )

        # saves generation number
        with open(experiment_name + "/gen.txt", "w") as f:
            f.write(str(gen))

        # saves file with the best solution
        np.savetxt(experiment_name + "/best.txt", population[alltime_best_individual])

        # saves simulation state
        EVOMAN_ENV.update_solutions([population, population_fitness])
        EVOMAN_ENV.save_state()

        generation_end_time = time.time()
        print(
            f"\n\nTime taken for generation {gen} :{generation_end_time - generation_start_time}"
        )

    experiment_end_time = time.time()
    print(
        f"\n\n\n\nTime taken for all generations:{experiment_end_time - experiment_start_time}"
    )
    print("=" * 50)


if __name__ == "__main__":
    from evolution import (
        fitness,
        create_offspring,
        select_individuals_for_next_generation,
    )

    execute_experiment(
        experiment_name=experiment_name,
        enemies=[7, 8],
        num_generations=3,
        population_size=5,
        num_hidden_neurons=10,
        create_offspring=partial(
            create_offspring, mutation_method="random", crossover_method="onepoint"
        ),
        select_individuals_for_next_generation=partial(
            select_individuals_for_next_generation, method="selection3"
        ),
        fitness=partial(fitness, method="fitness4"),
    )
