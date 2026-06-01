"""
Orijinal ve Gürültülü Sonuçların Karşılaştırılması.

Bu script, modellerin "original" ve "noisy" senaryolardaki performansını (F1-score)
karşılaştırıp aradaki farkı raporlar ve görselleştirir.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Headless
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def compare_noisy_scenarios(csv_path="results/experiments.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        logging.error(f"{csv_path} bulunamadı!")
        return
        
    df = pd.read_csv(csv_path)
    
    # Senaryosu original veya noisy olanları filtrele
    df_filtered = df[df["scenario"].isin(["original", "noisy"])].copy()
    if df_filtered.empty:
        logging.warning("CSV dosyasında original/noisy senaryo verisi bulunamadı.")
        return
        
    df_filtered["f1"] = pd.to_numeric(df_filtered["f1"], errors='coerce')
    df_filtered = df_filtered.dropna(subset=["f1"])
        
    # Basitlik için: model, dataset bazında senaryoların ortalama F1 skorunu alalım
    grouped = df_filtered.groupby(["model_type", "dataset", "scenario"])["f1"].mean().reset_index()
    
    if grouped.empty:
        return
        
    # Pivot ile yan yana getirelim
    pivot_df = grouped.pivot(index=["model_type", "dataset"], columns="scenario", values="f1").reset_index()
    
    # Eğer noisy veya original yoksa uyar
    if "original" not in pivot_df.columns or "noisy" not in pivot_df.columns:
        logging.warning("Original veya Noisy senaryolardan biri eksik!")
        return
        
    pivot_df["f1_diff"] = pivot_df["noisy"] - pivot_df["original"]
    pivot_df["f1_drop_pct"] = (pivot_df["f1_diff"] / pivot_df["original"]) * 100.0
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "noise_comparison_report.csv")
    pivot_df.to_csv(report_path, index=False)
    logging.info(f"Gürültü karşılaştırma raporu kaydedildi: {report_path}")
    
    # Grafik Çizimi
    pivot_df["Experiment"] = pivot_df["model_type"] + "\n(" + pivot_df["dataset"] + ")"
    
    x = range(len(pivot_df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar([i - width/2 for i in x], pivot_df["original"], width, label='Original', color='#3498db')
    rects2 = ax.bar([i + width/2 for i in x], pivot_df["noisy"], width, label='Noisy', color='#e74c3c')
    
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Modellerin Gürültü Dayanıklılığı (Original vs Noisy)', fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot_df["Experiment"], rotation=45, ha='center', fontsize=9)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "noise_comparison_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close('all')
    logging.info(f"Gürültü karşılaştırma grafiği kaydedildi: {plot_path}")

if __name__ == "__main__":
    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)
    compare_noisy_scenarios()
