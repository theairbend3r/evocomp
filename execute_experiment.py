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
    save_results,
    track_solution_improvement,
    select_individuals_for_next_generation,
)

sys.path.insert(0, "evoman")
from environment import Environment

np.random.seed(69)

# run experiement headless.
os.environ["SDL_VIDEODRIVER"] = "dummy"

parser = argparse.ArgumentParser()
parser.add_argument("-run", "--run", help="Current run, from 1 to 10")
parser.add_argument("-mode", "--mode", help="Run mode e.g. train")
parser.add_argument("-m", "--mutation", help="Mutation (e.g. 70)")
parser.add_argument("-c", "--crossover", help="Crossover method e.g. uniform80")
parser.add_argument("-e", "--enemies", nargs="+", help="Enemies (e.g. [1, 2])")
parser.add_argument("-g", "--num_generations", help="Number of generations (e.g. 40)")
parser.add_argument("-p", "--population_size", help="Population size (e.g. 40)")
args = parser.parse_args()

run = args.run
mode = args.mode
mutation = float(args.mutation)
crossover = args.crossover
enemies = args.enemies
num_generations = int(args.num_generations)
population_size = int(args.population_size)

experiment_name = f"crossover_{crossover}__mutation_{mutation}__enemies_{''.join(enemies)}__population_{population_size}__generations_{num_generations}__run_{run}"
print("=" * 50)
print("\n " + experiment_name)

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


def test_experiment(experiment_name):

    start_time = time.time()
    best_sol = np.loadtxt(experiment_name + "/best.txt")
    EVOMAN_ENV.update_parameter("speed", "fastest")
    EVOMAN_ENV.update_parameter("enemies", [1, 2, 3, 4, 5, 6, 7, 8])
    fitness, player_life, enemy_life, game_time = EVOMAN_ENV.play(best_sol)
    print("\n RUNNING THE BEST SOLUTION \n")

    with open("./test_results_enemy_all.txt", "a") as f:
        f.write(
            "\n"
            + str(fitness)
            + " "
            + str(player_life - enemy_life)
            + " "
            + str(experiment_name)
        )
    end_time = time.time()

    print(f"Time taken for testing a single run: {end_time - start_time}")


def execute_experiment(
    population_size: int,
    num_generations: int,
    create_offspring: Callable,
):
    # check environment state
    EVOMAN_ENV.state_to_log()

    # initialise population
    population = generate_population(
        population_size=population_size,
        num_sensors=EVOMAN_ENV.get_num_sensors(),
        num_hidden_neurons=10,
    )

    # get fitness of the 0th generation of the population
    population_fitness = evaluate(population=population)

    # track solution stats
    alltime_best_individual = np.argmax(population_fitness)
    # alltime_best_fitness = population_fitness[alltime_best_individual]
    alltime_best_solution = population_fitness[alltime_best_individual]
    population_fitness_mean = np.mean(population_fitness)
    population_fitness_std = np.std(population_fitness)

    # update solution (population fitness)
    EVOMAN_ENV.update_solutions([population, population_fitness])

    with open(experiment_name + "/results.txt", "a") as f:
        f.write("\n\ngen best mean std")

    # write initial solution to file
    save_results(
        num_gen=0,
        experiment_name=experiment_name,
        current_best_solution=alltime_best_solution,
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
        offspring = create_offspring(
            population=population, population_fitness=population_fitness
        )

        # evaluate the fitness of the offsprings
        offspring_fitness = evaluate(population=offspring)

        # add offsprings to the population
        population, population_fitness = add_offspring_to_population(
            population=population,
            population_fitness=population_fitness,
            offspring=offspring,
            offspring_fitness=offspring_fitness,
        )

        # when np.random.uniform(0,1) > percentage
        unique_individuals_idx = np.unique(population, axis=0, return_index=True)[1]
        population = population[unique_individuals_idx]
        population_fitness = population_fitness[unique_individuals_idx]

        # select a subset from the new population
        population, population_fitness = select_individuals_for_next_generation(
            population=population,
            population_fitness=population_fitness,
            population_size=population_size,
        )

        # track results
        current_best_individual = np.argmax(population_fitness)
        # current_best_fitness = population_fitness[current_best_individual]
        current_best_solution = population_fitness[current_best_individual]
        population_fitness_mean = np.mean(population_fitness)
        population_fitness_std = np.std(population_fitness)

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

        if current_best_solution > alltime_best_solution:
            alltime_best_individual = current_best_individual

        # save scores
        save_results(
            num_gen=gen,
            experiment_name=experiment_name,
            current_best_solution=current_best_solution,
            population_fitness_mean=population_fitness_mean,
            population_fitness_std=population_fitness_std,
        )

        # saves generation number
        with open(experiment_name + "/gen.txt", "w") as f:
            f.write(str(gen))

        # saves file with the best solution/weights
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

    if mode == "train":
        from evolution import (
            create_offspring,
            select_individuals_for_next_generation,
        )

        execute_experiment(
            num_generations=num_generations,
            population_size=population_size,
            create_offspring=partial(
                create_offspring,
                mutation_percentage=mutation,
                crossover_method=crossover,
            ),
        )

    if mode == "test":
        test_experiment(experiment_name)
