params = {
    "enemies": [[7, 8], [2, 5]],
    "population_size": [150, 180],
    "num_generations": [30, 40],
    "mutation": [0.2, 0.4],
    "crossover": ["blend-0.5", "blend-0.75", "blend-1.0"],
}

# params after tuning, needed for testing
tuned_params = {
    "crossover": ["blend-0.75"],
    "mutation": 0.2,
    "population_size": 120,
    "num_generations": 40,
}
