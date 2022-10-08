import numpy as np
from typing import Any


def generate_population(
    upper_limit: float,
    lower_limit: float,
    population_size: int,
    num_sensors: int,
    num_hidden_neurons: int,
):

    representation_size = (num_sensors + 1) * num_hidden_neurons + (
        num_hidden_neurons + 1
    ) * 5

    return np.random.uniform(
        lower_limit, upper_limit, (population_size, representation_size)
    )


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


def crossover(population: np.ndarray, num_offspring: int, method: str) -> np.ndarray:
    """Create offspring.

    Parameters
    ----------
    population : np.ndarray

    num_offspring : int

    method : str



    Returns
    -------
    np.ndarray

    """
    return population + 0.1


def mutate(population: np.ndarray, method: str) -> np.ndarray:
    """Mutate offspring genotype.

    Parameters
    ----------
    population : np.ndarray

    method : str

    Returns
    -------
    np.ndarray

    """
    # child_mutated = np.zeros(len(population))
    child_mutated = np.zeros((0, population.shape[1]))
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            temp = np.zeros(len(population[i]))
            temp[i] = population[i][j]
            if np.random.uniform(0, 1) <= 0.3:  # 0.3 will differ based on tuning
                temp[i] += np.random.normal(0, 1)
                if temp[i] < -1:
                    temp[i] = -1
                elif temp[i] > 1:
                    temp[i] = 1
        child_mutated = np.vstack((child_mutated, temp))
    return child_mutated


def add_offspring_to_population(
    population: np.ndarray,
    population_fitness: np.ndarray,
    offspring: np.ndarray,
    offspring_fitness: np.ndarray,
) -> tuple:

    """Add offspring to population.

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


def select_for_next_generation(
    population: np.ndarray, population_fitness: np.ndarray, method: str
) -> tuple:
    """Selecy individuals for the next generation.

    Parameters
    ----------
    population : np.ndarray

    population_fitness : np.ndarray

    method : str


    Returns
    -------
    np.ndarray

    """
    population_fitness_normalised = normalise_array(population_fitness)
    population_fitness_probability = population_fitness_normalised / np.sum(
        population_fitness_normalised, axis=0
    )
    return population, population_fitness


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
        f.write("\n\ngen best mean std")
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


def fitness(method):
    return f"fitness_{method}"
