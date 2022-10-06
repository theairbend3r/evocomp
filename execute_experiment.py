import os
import sys
import pathlib
import numpy as np
from functools import partial
from typing import Callable


from demo_controller import player_controller
from evolution import (
    add_offspring_to_population,
    doomsday_protocol,
    generate_population,
    evaluate,
    normalise_array,
    save_results,
    track_solution_improvement,
)

sys.path.insert(0, "evoman")
from environment import Environment

np.random.seed(69)


def execute_experiment(
    enemies: list,
    population_size: int,
    num_hidden_neurons: int,
    representation_size: int,
    representation_size_upper_limit: float,
    representation_size_lower_limit: float,
    num_generations: int,
    mutate: Callable,
    crossover: Callable,
    select_for_next_generation: Callable,
    fitness: Callable,
):
    # run experiement headless.
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    # name experiment/run to save logs.
    experiment_name = f"enemy_{''.join([str(e) for e in enemies])}-population_size_{population_size}-num_hidden_neurons_{num_hidden_neurons}-representation_size_{representation_size}-representation_size_upper_limit_{representation_size_upper_limit}-representation_size_lower_limit_{representation_size_lower_limit}-num_generations_{num_generations}-mutation_{mutate.keywords['method']}-crossover_{crossover.keywords['method']}-selection_{select_for_next_generation.keywords['method']}-fitness_{fitness.keywords['method']}"
    print(experiment_name)

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
        player_controller=player_controller(num_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
    )

    # check environment state
    EVOMAN_ENV.state_to_log()

    # initialise population
    population = generate_population(
        lower_limit=representation_size_lower_limit,
        upper_limit=representation_size_upper_limit,
        population_size=population_size,
        num_sensors=EVOMAN_ENV.get_num_sensors(),
        num_hidden_neurons=num_hidden_neurons,
    )

    # get fitness of the 0th generation of the population
    population_fitness = evaluate(env=EVOMAN_ENV, population=population)

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

    # start evolution
    for gen in range(1, num_generations + 1):
        # generate offsprings via crossover between parents
        offspring = crossover(
            population=population, num_offspring=population.shape[0], method="uniform"
        )

        # mutate genotype
        offspring = mutate(population=offspring)

        # evaluate the fitness of the offsprings
        offspring_fitness = evaluate(env=EVOMAN_ENV, population=offspring)

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

        population, population_fitness = select_for_next_generation(
            population=population, population_fitness=population_fitness, method="idk"
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


if __name__ == "__main__":
    from evolution import fitness, crossover, mutate, select_for_next_generation

    execute_experiment(
        enemies=[7, 8],
        num_generations=2,
        population_size=3,
        num_hidden_neurons=4,
        representation_size=5,
        representation_size_upper_limit=1.0,
        representation_size_lower_limit=-1.0,
        crossover=partial(crossover, method="crossover2"),
        mutate=partial(mutate, method="mutation1"),
        select_for_next_generation=partial(
            select_for_next_generation, method="selection3"
        ),
        fitness=partial(fitness, method="fitness4"),
    )
