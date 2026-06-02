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

    print("-> Analiz icin BATADAL verisi yukleniyor...")

    raw_data_df = load_dataset(
        "data/batadal/training_dataset_1.csv"
    )

    raw_data_df.columns = raw_data_df.columns.str.strip()
    raw_data_df = raw_data_df.ffill().bfill()

    if "DATETIME" in raw_data_df.columns:
        raw_data_df = raw_data_df.drop(
            columns=["DATETIME"]
        )

    if "ATT_FLAG" in raw_data_df.columns:
        raw_data_df = raw_data_df.drop(
            columns=["ATT_FLAG"]
        )

    # 1D sensör verisi ayrıştırılarak otomat matrisi oluşturulur
    sensor_column = raw_data_df.columns[0]
    print(f"-> Analiz icin secilen sensor sutunu: {sensor_column}")

    raw_data_series = raw_data_df[sensor_column].head(5000).values

    final_results = analyzer.run_grid_search(
        raw_data_series,
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