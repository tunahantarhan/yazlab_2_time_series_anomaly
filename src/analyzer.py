import json

from src.automata_core import AutomataPreprocessor
from src.data_loader import load_dataset


class ParameterAnalyzer:
    def __init__(self):
        self.preprocessor = AutomataPreprocessor()

    def run_grid_search(
        self,
        raw_data,
        sizes=[3, 4, 5, 6]
    ):
        results = {}

        print("\n-> Parametre Analiz Dongusu basliyor...")

        for w in sizes:
            for a in sizes:
                discrete_data = self.preprocessor.extract_patterns(
                    time_series=raw_data,
                    subsequence_length=w,
                    alphabet_size=a
                )

                transition_probs = self.preprocessor.calculate_transition_probabilities(
                    discrete_data
                )

                state_count = len(
                    list(set(discrete_data))
                )

                key = f"w:{w}_a:{a}"

                results[key] = {
                    "subsequence_length": w,
                    "alphabet_size": a,
                    "state_count": state_count,
                    "transition_matrix_size": len(transition_probs)
                }

                print(
                    f"   -> Analiz tamamlandi: {key} | State sayisi: {state_count}"
                )

        return results


if __name__ == "__main__":
    analyzer = ParameterAnalyzer()

    print("-> Analiz icin SWAT verisi yukleniyor...")

    raw_data = load_dataset(
        "data/swat/merged.csv"
    )

    raw_data.columns = raw_data.columns.str.strip()
    raw_data = raw_data.ffill()
    raw_data = raw_data.bfill()

    if "Timestamp" in raw_data.columns:
        raw_data = raw_data.drop(
            columns=["Timestamp"]
        )

    if "Normal/Attack" in raw_data.columns:
        raw_data["Normal/Attack"] = (
            raw_data["Normal/Attack"]
            .map({
                "Normal": 0,
                "Attack": 1
            })
        )

    raw_data = raw_data.head(50000)

    final_results = analyzer.run_grid_search(
        raw_data,
        sizes=[3, 4, 5, 6]
    )

    print("\n ==== TUM ANALIZ SONUCLARI ====")

    print(
        json.dumps(
            final_results,
            indent=4,
            ensure_ascii=False
        )
    )
