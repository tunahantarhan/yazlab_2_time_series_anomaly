from src.results_summary import summarize_metric


def test_summarize_metric_returns_mean_and_std():
    results = [
        {"seed": 42, "f1_score": 0.80},
        {"seed": 123, "f1_score": 0.90},
        {"seed": 2026, "f1_score": 1.00}
    ]

    summary = summarize_metric(
        results,
        metric_name="f1_score"
    )

    assert round(summary["mean"], 2) == 0.90
    assert round(summary["std"], 2) > 0
