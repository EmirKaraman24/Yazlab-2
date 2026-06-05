"""
Cross-Dataset Genellenebilirlik Analizi.

Bu script, bir veri setinde eğitilen modellerin (LSTM, GRU, 1D-CNN, Automata)
diğer veri setindeki performansını (F1-score) ölçer.
Sonuçları results/cross_dataset_results.csv dosyasına kaydeder.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import time

# Proje kök dizinini ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.dl_models import load_config, ModelWeightsManager
from src.sax import SAXTransformer
from src.automata import ProbabilisticAutomata
from src.run_inference import _load_numpy_split, _evaluate_automata_per_step
from src.metrics import compute_binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    config_path = "config.yaml"
    config = load_config(config_path)
    w = config["automata_params"]["window_size_fixed"]
    a = config["automata_params"]["alphabet_size_fixed"]

    processed_dir = os.path.join("data", "processed")
    manager = ModelWeightsManager(config_path=config_path)
    seed = 42
    suffix = f"seed_{seed}"

    # Verileri yükle
    skab_dir = os.path.join(processed_dir, "skab", "fold_1")
    batadal_dir = os.path.join(processed_dir, "batadal")

    # Numpy test dizileri (DL modelleri için)
    X_test_skab, y_test_skab = _load_numpy_split(skab_dir, "test")
    X_test_bat, y_test_bat = _load_numpy_split(batadal_dir, "test")

    # Numpy train dizileri (Automata eğitimi için)
    X_train_skab, y_train_skab = _load_numpy_split(skab_dir, "train")
    X_train_bat, y_train_bat = _load_numpy_split(batadal_dir, "train")

    if X_test_skab is None or X_test_bat is None:
        logging.error("Test verileri yüklenemedi. Önce run_preprocessing.py ve main.py çalıştırılmalıdır.")
        return

    # 1. Automata modellerini eğit (Cross-dataset için SAX ve Automata fit edilmeli)
    logging.info("Otomata modelleri eğitiliyor...")
    # SKAB Automata
    skab_sax = SAXTransformer(num_segments=w, alphabet_size=a)
    skab_sax.fit(X_train_skab)
    train_skab_sax = skab_sax.transform(X_train_skab)
    skab_automata = ProbabilisticAutomata()
    skab_automata.fit(train_skab_sax)

    # BATADAL Automata
    batadal_sax = SAXTransformer(num_segments=w, alphabet_size=a)
    batadal_sax.fit(X_train_bat)
    train_batadal_sax = batadal_sax.transform(X_train_bat)
    batadal_automata = ProbabilisticAutomata()
    batadal_automata.fit(train_batadal_sax)

    results = []

    models_list = ["LSTM", "1D-CNN", "GRU", "Automata"]

    # ----------------------------------------------------
    # SENARYO 1: Train SKAB -> Test BATADAL & SKAB
    # ----------------------------------------------------
    logging.info("--- Train: SKAB testleri yapılıyor ---")
    for model_name in models_list:
        # Test on SKAB (In-Domain)
        if model_name == "Automata":
            test_sax = skab_sax.transform(X_test_skab)
            y_pred = _evaluate_automata_per_step(skab_automata, test_sax, anomaly_threshold=0.05)
            metrics = compute_binary_metrics(y_test_skab, y_pred)
        else:
            try:
                model = manager.load(model_name, sequence_length=X_test_skab.shape[1], num_features=X_test_skab.shape[2], suffix=suffix)
                res = model.predict_model(X_test_skab, y_true=y_test_skab)
                metrics = compute_binary_metrics(y_test_skab, res["predictions"])
            except Exception as e:
                logging.warning(f"SKAB {model_name} yüklenemedi: {e}")
                metrics = {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        results.append({
            "Train_Dataset": "SKAB",
            "Test_Dataset": "SKAB",
            "Model": model_name,
            "F1": metrics["f1"]
        })

        # Test on BATADAL (Out-of-Domain)
        if model_name == "Automata":
            test_sax = skab_sax.transform(X_test_bat)
            y_pred = _evaluate_automata_per_step(skab_automata, test_sax, anomaly_threshold=0.05)
            metrics = compute_binary_metrics(y_test_bat, y_pred)
        else:
            try:
                model = manager.load(model_name, sequence_length=X_test_bat.shape[1], num_features=X_test_bat.shape[2], suffix=suffix)
                res = model.predict_model(X_test_bat, y_true=y_test_bat)
                metrics = compute_binary_metrics(y_test_bat, res["predictions"])
            except Exception as e:
                logging.warning(f"SKAB {model_name} on BATADAL test fail: {e}")
                metrics = {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        results.append({
            "Train_Dataset": "SKAB",
            "Test_Dataset": "BATADAL",
            "Model": model_name,
            "F1": metrics["f1"]
        })

    # ----------------------------------------------------
    # SENARYO 2: Train BATADAL -> Test SKAB & BATADAL
    # ----------------------------------------------------
    logging.info("--- Train: BATADAL testleri yapılıyor ---")
    for model_name in models_list:
        # Test on BATADAL (In-Domain)
        if model_name == "Automata":
            test_sax = batadal_sax.transform(X_test_bat)
            y_pred = _evaluate_automata_per_step(batadal_automata, test_sax, anomaly_threshold=0.05)
            metrics = compute_binary_metrics(y_test_bat, y_pred)
        else:
            try:
                # BATADAL models are saved with suffix, but let's make sure they load correctly
                model = manager.load(model_name, sequence_length=X_test_bat.shape[1], num_features=X_test_bat.shape[2], suffix=suffix)
                res = model.predict_model(X_test_bat, y_true=y_test_bat)
                metrics = compute_binary_metrics(y_test_bat, res["predictions"])
            except Exception as e:
                logging.warning(f"BATADAL {model_name} yüklenemedi: {e}")
                metrics = {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        results.append({
            "Train_Dataset": "BATADAL",
            "Test_Dataset": "BATADAL",
            "Model": model_name,
            "F1": metrics["f1"]
        })

        # Test on SKAB (Out-of-Domain)
        if model_name == "Automata":
            test_sax = batadal_sax.transform(X_test_skab)
            y_pred = _evaluate_automata_per_step(batadal_automata, test_sax, anomaly_threshold=0.05)
            metrics = compute_binary_metrics(y_test_skab, y_pred)
        else:
            try:
                model = manager.load(model_name, sequence_length=X_test_skab.shape[1], num_features=X_test_skab.shape[2], suffix=suffix)
                res = model.predict_model(X_test_skab, y_true=y_test_skab)
                metrics = compute_binary_metrics(y_test_skab, res["predictions"])
            except Exception as e:
                logging.warning(f"BATADAL {model_name} on SKAB test fail: {e}")
                metrics = {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        results.append({
            "Train_Dataset": "BATADAL",
            "Test_Dataset": "SKAB",
            "Model": model_name,
            "F1": metrics["f1"]
        })

    # Sonuçları DataFrame'e dönüştür ve kaydet
    df_results = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/cross_dataset_results.csv", index=False)
    
    print("\n=== CROSS-DATASET SONUÇLARI (F1-score) ===")
    print(df_results.pivot(index=["Train_Dataset", "Model"], columns="Test_Dataset", values="F1"))
    logging.info("Cross-dataset sonuçları kaydedildi: results/cross_dataset_results.csv")

if __name__ == "__main__":
    main()
