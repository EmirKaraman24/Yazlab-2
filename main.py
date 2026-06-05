"""
Ana Script (main.py)

Bu script, 5 farklı seed (42, 123, 2026, 7, 999) üzerinden projenin tüm eğitim
ve çıkarım (inference) süreçlerini çalıştırır. Sonuçları loglar.
"""

import os
import random
import numpy as np
import tensorflow as tf
import logging

from src.dl_models import load_config, train_all_models, ModelWeightsManager
from src.run_inference import (
    run_skab_inference,
    run_batadal_inference,
    _print_summary_table,
    _save_results_csv,
    _load_numpy_split,
    run_skab_automata_inference,
    run_batadal_automata_inference,
)
from src.experiment_logger import ExperimentLogger
from src.sax import SAXTransformer
from src.automata import ProbabilisticAutomata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def set_seed(seed: int):
    """Tüm kütüphaneler için rastgelelik tohumunu (seed) sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"=== SEED AYARLANDI: {seed} ===")

def main():
    config_path = "config.yaml"
    config = load_config(config_path)
    
    seeds = [42, 123, 2026, 7, 999]
    processed_data_dir = os.path.join("data", "processed")
    manager = ModelWeightsManager(config_path=config_path)
    logger = ExperimentLogger()
    
    all_rows = []
    
    for seed in seeds:
        set_seed(seed)
        suffix = f"seed_{seed}"
        
        # ---------------------------------------------------------
        # 1. SKAB EĞİTİM VE INFERENCE
        # ---------------------------------------------------------
        # SKAB için fold_1 üzerinden genel bir model eğitiyoruz (run_inference tüm test foldlarına uygulayacak)
        skab_dir = os.path.join(processed_data_dir, "skab", "fold_1")
        logging.info("SKAB Eğitimine başlanıyor...")
        X_train_skab, y_train_skab = _load_numpy_split(skab_dir, "train")
        X_val_skab, y_val_skab = _load_numpy_split(skab_dir, "val")
        
        skab_models = {}
        if X_train_skab is not None:
            skab_models = train_all_models(X_train_skab, y_train_skab, X_val_skab, y_val_skab, config_path)
            manager.save_all(skab_models, suffix=suffix)
            
        # SKAB Automata eğitimi
        w = config["automata_params"]["window_size_fixed"]
        a = config["automata_params"]["alphabet_size_fixed"]
        logging.info("SKAB Automata Eğitimine başlanıyor...")
        import time
        start_time = time.time()
        skab_sax = SAXTransformer(num_segments=w, alphabet_size=a)
        skab_sax.fit(X_train_skab)
        train_skab_sax = skab_sax.transform(X_train_skab)
        skab_automata = ProbabilisticAutomata()
        skab_automata.fit(train_skab_sax)
        skab_auto_train_time = time.time() - start_time
        
        # ---------------------------------------------------------
        # 2. BATADAL EĞİTİM VE INFERENCE
        # ---------------------------------------------------------
        batadal_dir = os.path.join(processed_data_dir, "batadal")
        logging.info("BATADAL Eğitimine başlanıyor...")
        X_train_bat, y_train_bat = _load_numpy_split(batadal_dir, "train")
        X_val_bat, y_val_bat = _load_numpy_split(batadal_dir, "val")
        
        batadal_models = {}
        if X_train_bat is not None:
            batadal_models = train_all_models(X_train_bat, y_train_bat, X_val_bat, y_val_bat, config_path)
            manager.save_all(batadal_models, suffix=suffix)
            
        # BATADAL Automata eğitimi
        logging.info("BATADAL Automata Eğitimine başlanıyor...")
        start_time = time.time()
        batadal_sax = SAXTransformer(num_segments=w, alphabet_size=a)
        batadal_sax.fit(X_train_bat)
        train_batadal_sax = batadal_sax.transform(X_train_bat)
        batadal_automata = ProbabilisticAutomata()
        batadal_automata.fit(train_batadal_sax)
        batadal_auto_train_time = time.time() - start_time
        
        # ---------------------------------------------------------
        # 3. ÇIKARIM (INFERENCE) - 3 FARKLI SENARYO
        # ---------------------------------------------------------
        scenarios = ["original", "noisy", "unseen"]
        
        for scenario in scenarios:
            logging.info(f"Seed {seed} | Senaryo: {scenario} için Inference başlatılıyor...")
            skab_rows = run_skab_inference(config, manager, processed_data_dir, suffix=suffix, scenario=scenario)
            batadal_rows = run_batadal_inference(config, manager, processed_data_dir, suffix=suffix, scenario=scenario)
            
            skab_auto_rows = run_skab_automata_inference(
                config, skab_sax, skab_automata, processed_data_dir,
                suffix=suffix, scenario=scenario, training_time=skab_auto_train_time
            )
            batadal_auto_rows = run_batadal_automata_inference(
                config, batadal_sax, batadal_automata, processed_data_dir,
                suffix=suffix, scenario=scenario, training_time=batadal_auto_train_time
            )
            
            combined_rows = skab_rows + batadal_rows + skab_auto_rows + batadal_auto_rows
            
            for row in combined_rows:
                row["seed"] = seed
                all_rows.append(row)
                
                dataset_full = row["dataset"]
                is_skab = "SKAB" in dataset_full
                dataset_name = "SKAB" if is_skab else "BATADAL"
                fold_id = dataset_full if is_skab else None
                
                row_threshold = row.get("threshold", 0.5)
                if row["model"] == "ProbabilisticAutomata":
                    row_threshold = 0.05
                
                logger.log(
                    model_type=row["model"],
                    dataset=dataset_name,
                    metrics=row,
                    fold_id=fold_id,
                    seed=seed,
                    scenario=scenario,
                    threshold=row_threshold,
                    n_samples=row.get("n_samples"),
                    n_anomalies_true=row.get("n_anomalies_true"),
                    n_anomalies_pred=row.get("n_anomalies_pred")
                )
            
    # Tüm seed'lerin toplu sonucunu kaydet
    output_csv = os.path.join("results", "inference_results_all_seeds.csv")
    _print_summary_table(all_rows)
    _save_results_csv(all_rows, output_path=output_csv)
    
    logging.info("Tüm seed'ler için uçtan uca pipeline tamamlandı.")

if __name__ == "__main__":
    main()
