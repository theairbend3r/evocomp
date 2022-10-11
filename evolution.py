import numpy as np
from typing import Any


def generate_population(
    population_size: int,
    num_sensors: int,
    num_hidden_neurons: int,
):

    representation_size = (num_sensors + 1) * num_hidden_neurons + (
        num_hidden_neurons + 1
    ) * 5

    return np.random.uniform(-1.0, 1.0, (population_size, representation_size))


def simulate_game(env, population_row: np.ndarray) -> float:
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
    fitness, _, _, _ = env.play(pcont=population_row)
    return fitness


def evaluate(env, population: np.ndarray) -> np.ndarray:
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
    return np.array([simulate_game(env, row) for row in population])


def select_parents_for_reproduction(population: np.ndarray) -> list:
    """

    Parameters
    ----------
    population : np.ndarray


    Returns
    -------
    list

    """
    # get indices for parents from population and shuffle them.
    parent_idx = np.arange(population.shape[0])
    np.random.shuffle(parent_idx)

    # if odd number of parents, remove last parent.
    if parent_idx.shape[0] % 2 != 0:
        parent_idx = parent_idx[:-1]

    # create combos of 2 parents.
    parent_combos = np.split(parent_idx, np.arange(2, parent_idx.shape[0], 2))

    return parent_combos


def crossover(parent_1: np.ndarray, parent_2: np.ndarray, method: str) -> tuple:
    """

    Parameters
    ----------
    parent_1 : np.ndarray

    parent_2 : np.ndarray

    method : str


    Returns
    -------
    tuple

    """
    if parent_1.shape != parent_2.shape:
        raise ValueError("Both parents should have same array shape.")

    if method == "onepoint":
        swap_point = np.random.randint(0, parent_1.shape[0])
        offspring_1 = np.concatenate((parent_1[:swap_point], parent_2[swap_point:]))
        offspring_2 = np.concatenate((parent_1[swap_point:], parent_2[:swap_point]))

        return offspring_1, offspring_2


def mutate(individual: np.ndarray, mutation_percentage: str) -> np.ndarray:
    """

    Parameters
    ----------
    individual : np.ndarray

    method : str


    Returns
    -------
    np.ndarray

    """
    # flip_threshold = 0.5
    child_mutated = np.zeros(len(individual))
    for i in range(0, len(individual)):
        child_mutated[i] = individual[i]
        if (
            np.random.uniform(0, 1) <= mutation_percentage
        ):  # 0.3 will differ based on tuning
            child_mutated[i] += np.random.normal(0, 1)
            if child_mutated[i] < -1:
                child_mutated[i] = -1
            elif child_mutated[i] > 1:
                child_mutated[i] = 1
    return child_mutated


def create_offspring(
    population: np.ndarray, crossover_method: str, mutation_percentage: str
):
    """

    Parameters
    ----------
    population : np.ndarray

    crossover_method : str

    mutation_percentage : str

    Returns
    -------


    """

    parent_combos = select_parents_for_reproduction(population=population)

    offspring_list = []
    for i in range(population[parent_combos].shape[0]):
        parent_1, parent_2 = (
            population[parent_combos][i][0],
            population[parent_combos][i][1],
        )

        offspring_1, offspring_2 = crossover(
            parent_1=parent_1, parent_2=parent_2, method=crossover_method
        )

        offspring_1 = mutate(
            individual=offspring_1, mutation_percentage=mutation_percentage
        )
        offspring_2 = mutate(
            individual=offspring_2, mutation_percentage=mutation_percentage
        )

        offspring_list.append(offspring_1)
        offspring_list.append(offspring_2)

    offspring_array = np.vstack(offspring_list)

    return offspring_array


def add_offspring_to_population(
    population: np.ndarray,
    population_fitness: np.ndarray,
    offspring: np.ndarray,
    offspring_fitness: np.ndarray,
) -> tuple:

    """

    Parameters
    ----------
    population : np.ndarray

    population_fitness : np.ndarray

    offspring : np.ndarray

    offspring_fitness : np.ndarray


    Returns
    -------
    tuple

    """
    combined_population = np.vstack([population, offspring])
    combined_population_fitness = np.append(population_fitness, offspring_fitness)

    return combined_population, combined_population_fitness


def normalise_array(arr: np.ndarray) -> float:
    """Normalise array.

    Parameters
    ----------
    arr : np.ndarray


    Returns
    -------
    np.ndarray

    """
    if np.max(arr) - np.min(arr) > 0:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    else:
        return 0.0000000001


def select_individuals_for_next_generation(
    population: np.ndarray,
    population_fitness: np.ndarray,
    population_size: int,
    method: str,
) -> tuple:
    """

    Parameters
    ----------
    population : np.ndarray

    population_fitness : np.ndarray

    population_size : int

    method : str


    Returns
    -------
    tuple

    """
    # subset population rows based on top-n fitness values.
    top_n_fitness_idx = np.argpartition(population_fitness, -population_size)[
        -population_size:
    ]
    top_n_population = population[top_n_fitness_idx]
    top_n_population_fitness = population_fitness[top_n_fitness_idx]

    return top_n_population, top_n_population_fitness


def track_solution_improvement(
    alltime_best_solution: float,
    current_best_solution: float,
    solution_not_improved_count: int,
) -> int:
    """Track solution improvement over generations.


    If solution does not improve for n generations,
    implement doomsday protocol.

    Parameters
    ----------
    alltime_best_solution : float

    current_best_solution : float

    solution_not_improved_count : int


    Returns
    -------
    int

    """
    # doomsday if fitness score doesn't improve for n generations.
    if alltime_best_solution >= current_best_solution:
        solution_not_improved_count += 1
    else:
        alltime_best_solution = current_best_solution
        solution_not_improved_count = 0

    return solution_not_improved_count


def doomsday_protocol(population: np.ndarray, population_fitness: np.ndarray) -> tuple:
    """Implement the doomsday protocol.

    Parameters
    ----------
    population : np.ndarray

    population_fitness : np.ndarray


    Returns
    -------
    tuple

    """
    return population, population_fitness


def save_results(
    num_gen: int,
    experiment_name: str,
    alltime_best_solution: float,
    population_fitness_mean: np.floating[Any],
    population_fitness_std: np.floating[Any],
):
    """Saves results to file.

    Parameters
    ----------
    num_gen : int

    experiment_name : str

    alltime_best_solution : float

    population_fitness_mean : float

    population_fitness_std : float

    """
    with open(experiment_name + "/results.txt", "a") as f:
        # f.write("\n\ngen best mean std")
        print(
            "\n GENERATION "
            + str(num_gen)
            + " "
            + str(round(alltime_best_solution, 6))
            + " "
            + str(round(population_fitness_mean, 6))
            + " "
            + str(round(population_fitness_std, 6))
        )
        f.write(
            "\n"
            + str(num_gen)
            + " "
            + str(round(alltime_best_solution, 6))
            + " "
            + str(round(population_fitness_mean, 6))
            + " "
            + str(round(population_fitness_std, 6))
        )
