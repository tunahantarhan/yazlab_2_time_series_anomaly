# -*- coding: utf-8 -*-

import time
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from src.seed_runner import run_for_seeds
from src.preprocessing_pipeline import prepare_dataset
from src.sequence_generator import create_labeled_sequences
from src.models.gru_model import build_gru_model
from src.models.cnn_model import build_cnn_model
from src.trainer import train_model
from src.statistical_tests import run_wilcoxon_test, run_mcnemar_test


SEEDS = [42, 123, 2026, 7, 999]


def prepare_swat_data():
    df = pd.read_csv(
        "data/swat/merged.csv"
    )

    df.columns = df.columns.str.strip()
    df = df.ffill()
    df = df.bfill()

    df["Normal/Attack"] = (
        df["Normal/Attack"]
        .map({
            "Normal": 0,
            "Attack": 1
        })
    )

    normal_df = df[
        df["Normal/Attack"] == 0
    ].head(25000)

    attack_df = df[
        df["Normal/Attack"] == 1
    ].head(25000)

    df = pd.concat(
        [normal_df, attack_df],
        ignore_index=True
    )

    dataset = prepare_dataset(
        df,
        target_column="Normal/Attack",
        columns_to_drop=["Timestamp"]
    )

    X_train_seq, y_train_seq = create_labeled_sequences(
        dataset["X_train"],
        dataset["y_train"],
        window_size=4
    )

    X_val_seq, y_val_seq = create_labeled_sequences(
        dataset["X_val"],
        dataset["y_val"],
        window_size=4
    )

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq


def run_single_model(seed, model_type):
    X_train, y_train, X_val, y_val = prepare_swat_data()

    if model_type == "gru":
        model = build_gru_model(
            input_shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        )
    else:
        model = build_cnn_model(
            input_shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        )

    start_time = time.time()

    train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    training_time = time.time() - start_time

    predictions = model.predict(
        X_val,
        verbose=0
    )

    predictions = (
        predictions > 0.5
    ).astype(int).flatten()

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    return {
        "seed": seed,
        "model": model_type,
        "f1_score": f1,
        "training_time": training_time,
        "predictions": predictions,
        "y_true": y_val
    }


def summarize_results(results, model_name):
    f1_scores = [
        result["f1_score"]
        for result in results
    ]

    times = [
        result["training_time"]
        for result in results
    ]

    print()
    print(f"=== {model_name.upper()} SEED RESULTS ===")

    for result in results:
        print(
            f"Seed {result['seed']} | "
            f"F1: {result['f1_score']:.4f} | "
            f"Training Time: {result['training_time']:.4f} sn"
        )

    print()
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    print(f"Std F1 : {np.std(f1_scores):.4f}")
    print(f"Mean Training Time: {np.mean(times):.4f} sn")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # output kısmında uyarı kalabalığı engellenir
    
    print("SWAT 5 seed deneyleri baslatiliyor...")

    gru_results = run_for_seeds(
        SEEDS,
        lambda seed: run_single_model(
            seed,
            "gru"
        )
    )

    cnn_results = run_for_seeds(
        SEEDS,
        lambda seed: run_single_model(
            seed,
            "cnn"
        )
    )

    summarize_results(
        gru_results,
        "gru"
    )

    summarize_results(
        cnn_results,
        "cnn"
    )

    # ==== GERCEK VERİLERLE İSTATİSTİKSEL TESTLERİN UYGULANMASI ====
    print("\n" + "="*55)
    print("==== GERÇEK VERİLERLE İSTATİSTİKSEL TEST SONUÇLARI ====")
    print("="*55)

    # Wilcoxon Testi (5 Seed F1 Skorları üzerinden)
    gru_f1_scores = [res["f1_score"] for res in gru_results]
    cnn_f1_scores = [res["f1_score"] for res in cnn_results]
    
    try:
        w_stat, w_pval = run_wilcoxon_test(gru_f1_scores, cnn_f1_scores)
        print("\n[Wilcoxon Isaretli Rutbe Testi]")
        print(f"Statistic: {w_stat:.4f} | P-value: {w_pval:.4f}")
        if w_pval < 0.05:
            print("Yorum: GRU ve CNN F1 skorları arasında istatistiksel olarak anlamlı fark vardır.")
        else:
            print("Yorum: GRU ve CNN F1 skorları arasında istatistiksel olarak anlamlı fark YOKTUR.")
    except Exception as e:
        print(f"\n[Wilcoxon Testi]: Skorlar arasında varyans yok. Hata: {e}")

    # McNemar Testi (Seed 42'nin Tahminleri üzerinden)
    y_true_seed42 = gru_results[0]["y_true"]
    gru_preds_seed42 = gru_results[0]["predictions"]
    cnn_preds_seed42 = cnn_results[0]["predictions"]

    try:
        m_stat, m_pval = run_mcnemar_test(y_true_seed42, gru_preds_seed42, cnn_preds_seed42)
        print("\n[McNemar Testi] (Seed 42 Tahminleri)")
        print(f"Statistic: {m_stat:.4f} | P-value: {m_pval:.4f}")
        if m_pval < 0.05:
            print("Yorum: Modellerin hata yaptıkları örüntüler birbirinden FARKLIDIR.")
        else:
            print("Yorum: Modeller benzer noktalarda hata yapmıştır, tahminler BENZERDİR.")
    except Exception as e:
        print(f"\n[McNemar Testi]: Hesaplanamadı. Hata: {e}")
