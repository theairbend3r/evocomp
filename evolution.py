import numpy as np


def generate_population(
    upper_limit: float,
    lower_limit: float,
    population_size: int,
    representation_size: int,
):
    return np.random.uniform(
        lower_limit, upper_limit, (population_size, representation_size)
    )


def mutation(method):
    return f"mutation_{method}"


def crossover(method):
    return f"crossover_{method}"


def selection(method):
    return f"selection_{method}"


def fitness(method):
    return f"fitness_{method}"
