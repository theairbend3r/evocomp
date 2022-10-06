import os
import sys
import pathlib
from functools import partial
from typing import Callable

from evolution import generate_population
from demo_controller import player_controller

sys.path.insert(0, "evoman")
from environment import Environment


def execute_experiment(
    enemies: list,
    population_size: int,
    num_hidden_neurons: int,
    representation_size: int,
    representation_size_upper_limit: float,
    representation_size_lower_limit: float,
    num_generations: int,
    mutation: Callable,
    crossover: Callable,
    selection: Callable,
    fitness: Callable,
):
    print(
        enemies,
        population_size,
        num_hidden_neurons,
        representation_size,
        representation_size_upper_limit,
        representation_size_lower_limit,
        num_generations,
        mutation(),
        crossover(),
        selection(),
        fitness(),
    )

    # run experiement headless.
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    # name experiment/run to save logs.
    experiment_name = f"enemy_{''.join([str(e) for e in enemies])}-population_size_{population_size}-num_hidden_neurons_{num_hidden_neurons}-representation_size_{representation_size}-representation_size_upper_limit_{representation_size_upper_limit}-representation_size_lower_limit_{representation_size_lower_limit}-num_generations_{num_generations}-mutation_{mutation.keywords['method']}-crossover_{crossover.keywords['method']}-selection_{selection.keywords['method']}-fitness_{fitness.keywords['method']}"
    print(experiment_name)

    # create directory, if it does not already exist, to store runs.
    log_folder = pathlib.Path(f"./{experiment_name}")
    if not log_folder.is_dir():
        log_folder.mkdir(parents=True, exist_ok=True)

    # create evoman environment
    evoman_env = Environment(
        experiment_name=experiment_name,
        enemies=enemies,
        multiplemode="yes",
        playermode="ai",
        player_controller=player_controller(num_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
    )

    # initialise population
    population = generate_population(
        lower_limit=representation_size_lower_limit,
        upper_limit=representation_size_upper_limit,
        population_size=population_size,
        representation_size=representation_size,
    )

    # start evolution
    for gen in range(1, num_generations + 1):
        print(gen)


if __name__ == "__main__":
    from evolution import (
        mutation,
        crossover,
        selection,
        fitness,
    )

    execute_experiment(
        enemies=[1],
        num_generations=5,
        population_size=3,
        num_hidden_neurons=4,
        representation_size=6,
        representation_size_upper_limit=1.0,
        representation_size_lower_limit=-1.0,
        mutation=partial(mutation, method="mutation1"),
        crossover=partial(crossover, method="crossover2"),
        selection=partial(selection, method="selection3"),
        fitness=partial(fitness, method="fitness4"),
    )
