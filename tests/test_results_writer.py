import json

from src.results_writer import save_results


def test_save_results_creates_json_file(tmp_path):
    results = {
        "model": "GRU",
        "f1_score": 0.85,
        "runtime": 1.23
    }

    output_path = tmp_path / "results.json"

    save_results(
        results,
        output_path
    )

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded["model"] == "GRU"
    assert loaded["f1_score"] == 0.85
