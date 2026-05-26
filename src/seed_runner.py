from src.seed import set_seed


def run_for_seeds(seeds, experiment_function):
    results = []

    for seed in seeds:
        set_seed(seed)

        result = experiment_function(seed)
        results.append(result)

    return results
