from src.seed_runner import run_for_seeds


def test_run_for_seeds_returns_all_results():
    seeds = [42, 123, 2026]

    def dummy_experiment(seed):
        return {
            "seed": seed,
            "accuracy": 1.0
        }

    results = run_for_seeds(
        seeds,
        dummy_experiment
    )

    assert len(results) == 3
    assert results[0]["seed"] == 42
