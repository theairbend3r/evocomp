params = {
    "enemies": [[7, 8], [2, 5]],
    "population_size": [2, 3],
    "num_generations": [2, 3],
    "mutation": [0.2, 0.4],
    "crossover": ["blend-0.7", "sar-0.7"],
}

# params after tuning, needed for testing
tuned_params = {
    "crossover": "blend-0.7",
    "mutation": 0.4,
    "population_size": 2,
    "num_generations": 3,
}
