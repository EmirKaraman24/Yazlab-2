"""
Parametre Duyarlılık (Sensitivity) Görselleştirmeleri.

Bu script, Experiment Logger'ın oluşturduğu CSV dosyasını okuyarak
Olasılıksal Otomata'nın hiperparametre değişimlerine (window_size ve alphabet_size)
karşı performansındaki (F1-score) değişimi görselleştirir (Heatmap).
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Headless
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def plot_automata_sensitivity(csv_path="results/experiments.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        logging.error(f"{csv_path} bulunamadı! Lütfen önce run_automata_experiments.py'yi çalıştırın.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Sadece Otomata deneylerini filtrele
    automata_df = df[df["model_type"] == "ProbabilisticAutomata"]
    if automata_df.empty:
        logging.warning("CSV dosyasında ProbabilisticAutomata logu bulunamadı.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset bazında ayrım yap
    datasets = automata_df["dataset"].unique()
    
    for dataset in datasets:
        ds_df = automata_df[automata_df["dataset"] == dataset].copy()
        
        # Sayısal değerlere dönüştür
        ds_df["window_size"] = pd.to_numeric(ds_df["window_size"], errors='coerce')
        ds_df["alphabet_size"] = pd.to_numeric(ds_df["alphabet_size"], errors='coerce')
        ds_df["f1"] = pd.to_numeric(ds_df["f1"], errors='coerce')
        
        ds_df = ds_df.dropna(subset=["window_size", "alphabet_size", "f1"])
        
        # Ortalamayı hesapla
        pivot_df = ds_df.pivot_table(index="window_size", columns="alphabet_size", values="f1", aggfunc="mean")
        
        if pivot_df.empty:
            continue
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".4f", cbar_kws={'label': 'F1 Score'})
        plt.title(f"Automata Parameter Sensitivity (F1-Score)\nDataset: {dataset}", pad=20, fontsize=14, fontweight='bold')
        plt.xlabel("Alphabet Size (a)")
        plt.ylabel("Window Size (w)")
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"sensitivity_automata_{dataset}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        logging.info(f"Parametre duyarlılık haritası kaydedildi: {out_path}")

if __name__ == "__main__":
    # Proje köküne geç
    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)
    plot_automata_sensitivity()
