"""
Otomata Parametre Varyasyon Deneyleri.

Bu script, config.yaml içindeki window_sizes (3,4,5,6) ve 
alphabet_sizes (3,4,5,6) listelerindeki tüm kombinasyonları deneyerek 
Probabilistic Automata'nın performansını ölçer ve kaydeder.
"""

import os
import sys
import numpy as np
import logging
import pandas as pd

# Proje kökünü yola ekleyelim
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_config
from src.preprocessing import create_sliding_windows
from src.sax import SAXTransformer
from src.automata import ProbabilisticAutomata
from src.experiment_logger import ExperimentLogger
from src.metrics import compute_binary_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def _evaluate_automata_per_step(automata, sax_sequence, anomaly_threshold=0.05):
    """
    Otomata için adım adım anomali tespiti yapar.
    Görülmemiş (unseen) durumlar veya geçiş olasılığı threshold'un altında
    kalan durumlar anomali (1) olarak etiketlenir.
    """
    n = len(sax_sequence)
    y_pred = np.zeros(n)
    if n < 2:
        return y_pred
        
    mapped_data = automata.map_sequence_to_states(sax_sequence)
    resolved_states = mapped_data["resolved_states"]
    transition_rows = mapped_data["transition_rows"]
    
    for i in range(n - 1):
        raw_state = sax_sequence[i+1]
        is_unseen = raw_state not in automata.state_to_id
        
        if is_unseen:
            # Görülmemiş (unseen) durumlar anomali olarak kabul edilir
            y_pred[i+1] = 1
        else:
            next_state = resolved_states[i+1]
            next_id = automata.state_to_id[next_state]
            prob = transition_rows[i][next_id]
            
            # Geçiş olasılığı çok düşükse anomali
            if prob < anomaly_threshold:
                y_pred[i+1] = 1
                
    return y_pred

def run_automata_grid_search(config_path="config.yaml"):
    config = load_config(config_path)
    window_sizes = config.get("automata_params", {}).get("window_sizes", [3, 4, 5, 6])
    alphabet_sizes = config.get("automata_params", {}).get("alphabet_sizes", [3, 4, 5, 6])
    
    # Otomata anomali tespit eşiği (0.05 olasılıktan düşükse anomali)
    anomaly_threshold = 0.05
    
    processed_dir = os.path.join("data", "processed")
    logger = ExperimentLogger()
    
    # Hızlı analiz için sadece SKAB Fold 1 ve BATADAL kullanıyoruz
    datasets = [
        ("SKAB", os.path.join(processed_dir, "skab", "fold_1")),
        ("BATADAL", os.path.join(processed_dir, "batadal"))
    ]
    
    for dataset_name, data_dir in datasets:
        train_csv = os.path.join(data_dir, "train.csv")
        test_csv = os.path.join(data_dir, "test.csv")
        
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            logging.warning(f"{data_dir} dizininde CSV dosyaları eksik, atlanıyor.")
            continue
            
        train_df = pd.read_csv(train_csv, index_col=0)
        test_df = pd.read_csv(test_csv, index_col=0)
        
        label_col = "anomaly" if dataset_name == "SKAB" else "ATT_FLAG"
        if label_col not in train_df.columns:
            logging.warning(f"{label_col} bulunamadı, {dataset_name} atlanıyor.")
            continue
        
        # Grid Search Döngüleri
        for w in window_sizes:
            logging.info(f"==> Veriler sliding window'a çevriliyor (Window: {w})")
            X_train_w, y_train_w = create_sliding_windows(train_df, window_size=w, feature_cols=["PC1"], label_col=label_col)
            X_test_w, y_test_w = create_sliding_windows(test_df, window_size=w, feature_cols=["PC1"], label_col=label_col)
            
            for a in alphabet_sizes:
                logging.info(f"--- {dataset_name} | Window: {w} | Alphabet: {a} ---")
                
                # 1. SAX Dönüşümü
                sax = SAXTransformer(num_segments=w, alphabet_size=a)
                sax.fit(X_train_w)
                train_sax = sax.transform(X_train_w)
                test_sax = sax.transform(X_test_w)
                
                # 2. Otomata Eğitimi
                automata = ProbabilisticAutomata()
                automata.fit(train_sax)
                
                # 3. Anomali Çıkarımı
                y_pred = _evaluate_automata_per_step(automata, test_sax, anomaly_threshold=anomaly_threshold)
                
                # 4. Performans ve Loglama
                metrics = compute_binary_metrics(y_test_w, y_pred)
                
                logger.log(
                    model_type="ProbabilisticAutomata",
                    dataset=f"{dataset_name}_fold_1" if dataset_name == "SKAB" else "BATADAL_test",
                    metrics=metrics,
                    window_size=w,
                    alphabet_size=a,
                    threshold=anomaly_threshold,
                    scenario="original",
                    n_samples=len(y_test_w),
                    n_anomalies_true=int(np.sum(y_test_w)),
                    n_anomalies_pred=int(np.sum(y_pred)),
                    notes="Grid search for Automata parameters"
                )

if __name__ == "__main__":
    run_automata_grid_search()
