"""
Deney Raporu Doldurma Scripti (fill_report.py).

Bu script, results/experiments.csv ve results/cross_dataset_results.csv dosyalarını okuyarak:
1. Rapor şablonundaki (yazlab2-EK-1.pdf) Tablo 1 - Tablo 5'i doldurmak için gerekli istatistikleri hesaplar.
2. Sonuçları results/report_tables.md dosyasına tablo olarak yazar (kolay kopyalama için).
3. pypdf ve reportlab kullanarak yazlab2-EK-1.pdf üzerine verileri yazıp yazlab2-EK-1_filled.pdf olarak kaydeder.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import io

# reportlab ve pypdf importları
try:
    import pypdf
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
except ImportError:
    logging.error("Lütfen pypdf ve reportlab kütüphanelerini kurun.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def safe_mean_std(series):
    """Ortalama ve standart sapmayı formatlı şekilde döner (örn. '0.8521 +- 0.0123')."""
    if series.empty:
        return "N/A"
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return "N/A"
    mean = series.mean()
    std = series.std()
    if pd.isna(std):
        std = 0.0
    return f"{mean:.4f} \u00b1 {std:.4f}"

def safe_val(val, decimals=4):
    """Değeri güvenli şekilde float'a çevirip yuvarlar."""
    try:
        fval = float(val)
        if pd.isna(fval):
            return "N/A"
        return f"{fval:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

def main():
    exp_csv = "results/experiments.csv"
    cross_csv = "results/cross_dataset_results.csv"
    pdf_template = r"c:\Users\EMIR\Downloads\yazlab2-EK-1.pdf"
    pdf_output = r"c:\Users\EMIR\Downloads\yazlab2-EK-1_filled.pdf"

    if not os.path.exists(exp_csv):
        logging.error(f"{exp_csv} bulunamadı! Önce main.py çalıştırılmalıdır.")
        return

    df = pd.read_csv(exp_csv)
    df["f1"] = pd.to_numeric(df["f1"], errors='coerce')
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors='coerce')
    df["precision"] = pd.to_numeric(df["precision"], errors='coerce')
    df["recall"] = pd.to_numeric(df["recall"], errors='coerce')
    df["training_time"] = pd.to_numeric(df["training_time"], errors='coerce')
    df["inference_time"] = pd.to_numeric(df["inference_time"], errors='coerce')

    # Model isimlendirmelerini standartlaştır (ProbabilisticAutomata -> Automata)
    df["model_type"] = df["model_type"].replace({"ProbabilisticAutomata": "Automata"})

    models = ["LSTM", "GRU", "1D-CNN", "Automata"]

    # ----------------------------------------------------
    # TABLO 1: Model Performansı ve Stabilitesi (F1-score)
    # ----------------------------------------------------
    # SKAB için seed başına 5 fold'un ortalama F1'ini alıp seed'ler üzerinden ortalama + std hesaplayalım
    tablo1_data = {}
    
    # 1. SWAT (SKAB)
    df_skab_orig = df[(df["dataset"] == "SKAB") & (df["scenario"] == "original")]
    
    for m in models:
        df_m = df_skab_orig[df_skab_orig["model_type"] == m]
        # Her seed için fold ortalamasını al
        seed_f1s = df_m.groupby("seed")["f1"].mean()
        tablo1_data[(m, "SWAT")] = safe_mean_std(seed_f1s)

    # 2. WADI (N/A)
    for m in models:
        tablo1_data[(m, "WADI")] = "N/A"

    # 3. BATADAL
    df_bat_orig = df[(df["dataset"] == "BATADAL") & (df["scenario"] == "original")]
    for m in models:
        df_m = df_bat_orig[df_bat_orig["model_type"] == m]
        seed_f1s = df_m.groupby("seed")["f1"].mean()
        tablo1_data[(m, "BATADAL")] = safe_mean_std(seed_f1s)

    # ----------------------------------------------------
    # TABLO 2: Gürültü Etkisi ve Unseen Senaryo Analizi (BATADAL üzerinde)
    # ----------------------------------------------------
    tablo2_data = {}
    df_bat = df[df["dataset"] == "BATADAL"]
    
    for m in models:
        # Original F1
        f1_orig = df_bat[(df_bat["model_type"] == m) & (df_bat["scenario"] == "original")]["f1"].mean()
        tablo2_data[(m, "Original")] = safe_val(f1_orig)
        
        # Noisy F1
        f1_noisy = df_bat[(df_bat["model_type"] == m) & (df_bat["scenario"] == "noisy")]["f1"].mean()
        tablo2_data[(m, "Noisy")] = safe_val(f1_noisy)
        
        # Unseen Recall (Det. Rate)
        rec_unseen = df_bat[(df_bat["model_type"] == m) & (df_bat["scenario"] == "unseen")]["recall"].mean()
        tablo2_data[(m, "Det_Rate")] = safe_val(rec_unseen)
        
        # Unseen Accuracy (Map. Acc.)
        acc_unseen = df_bat[(df_bat["model_type"] == m) & (df_bat["scenario"] == "unseen")]["accuracy"].mean()
        tablo2_data[(m, "Map_Acc")] = safe_val(acc_unseen)

    # ----------------------------------------------------
    # TABLO 3: Cross-Dataset Performans Karşılaştırması
    # ----------------------------------------------------
    tablo3_data = {}
    # Varsayılan değerler
    for m in models:
        for t_train in ["SKAB", "WADI", "BATADAL"]:
            for t_test in ["SWAT", "WADI", "BATADAL"]:
                tablo3_data[(m, t_train, t_test)] = "N/A"

    if os.path.exists(cross_csv):
        df_cross = pd.read_csv(cross_csv)
        df_cross["Model"] = df_cross["Model"].replace({"ProbabilisticAutomata": "Automata"})
        for _, r in df_cross.iterrows():
            m = r["Model"]
            train_d = "SKAB" if r["Train_Dataset"] == "SKAB" else "BATADAL"
            test_d = "SWAT" if r["Test_Dataset"] == "SKAB" else "BATADAL"
            tablo3_data[(m, train_d, test_d)] = safe_val(r["F1"])
    else:
        # Cross dataset çalıştırılmadıysa tablo 1'deki in-domain sonuçlarını kullanalım
        for m in models:
            tablo3_data[(m, "SKAB", "SWAT")] = safe_val(df_skab_orig[df_skab_orig["model_type"] == m]["f1"].mean())
            tablo3_data[(m, "BATADAL", "BATADAL")] = safe_val(df_bat_orig[df_bat_orig["model_type"] == m]["f1"].mean())

    # ----------------------------------------------------
    # TABLO 4: Automata Parametre Duyarlılık Analizi (F1-score)
    # ----------------------------------------------------
    # Grid search verilerini toplayalım
    tablo4_data = {
        "W_3": "N/A", "W_4": "N/A", "W_5": "N/A", "W_6": "N/A",
        "A_3": "N/A", "A_4": "N/A", "A_5": "N/A", "A_6": "N/A",
    }
    
    # Grid search notlarında 'Grid search for Automata parameters' yazar
    df_grid = df[(df["model_type"] == "Automata") & (df["notes"].str.contains("Grid", na=True))]
    # SKAB fold 1 üzerinden duyarlılık
    df_grid_skab = df_grid[df_grid["dataset"].str.contains("SKAB", na=False)]
    
    if not df_grid_skab.empty:
        # Window size varyasyonu (Alphabet size fixed at 3)
        df_w = df_grid_skab[df_grid_skab["alphabet_size"] == 3]
        for w_val in [3, 4, 5, 6]:
            val = df_w[df_w["window_size"] == w_val]["f1"].mean()
            tablo4_data[f"W_{w_val}"] = safe_val(val)
            
        # Alphabet size varyasyonu (Window size fixed at 4)
        df_a = df_grid_skab[df_grid_skab["window_size"] == 4]
        for a_val in [3, 4, 5, 6]:
            val = df_a[df_a["alphabet_size"] == a_val]["f1"].mean()
            tablo4_data[f"A_{a_val}"] = safe_val(val)
    else:
        # Eğer grid search çalıştırılmadıysa varsayılan parametre sonucunu koyalım
        f1_auto = df_skab_orig[df_skab_orig["model_type"] == "Automata"]["f1"].mean()
        tablo4_data["W_4"] = safe_val(f1_auto)
        tablo4_data["A_3"] = safe_val(f1_auto)

    # ----------------------------------------------------
    # TABLO 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması
    # ----------------------------------------------------
    # SKAB fold 1 original senaryo üzerindeki eğitim ve inference sürelerinin ortalamasını alalım
    tablo5_data = {}
    df_time = df[(df["scenario"] == "original")]
    
    for m in models:
        df_m = df_time[df_time["model_type"] == m]
        t_train = df_m["training_time"].mean()
        t_inf = df_m["inference_time"].mean()
        
        tablo5_data[(m, "Train")] = safe_val(t_train, decimals=2)
        tablo5_data[(m, "Inf")] = safe_val(t_inf, decimals=2)

    # ----------------------------------------------------
    # MARKDOWN ÇIKTISI ÜRET
    # ----------------------------------------------------
    md_content = f"""# Proje Deney Sonuçları Rapor Tabloları

## Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score \u00b1 Standart Sapma)

| Model | SWAT (SKAB) | WADI | BATADAL |
| :--- | :---: | :---: | :---: |
| **LSTM** | {tablo1_data[("LSTM", "SWAT")]} | N/A | {tablo1_data[("LSTM", "BATADAL")]} |
| **GRU** | {tablo1_data[("GRU", "SWAT")]} | N/A | {tablo1_data[("GRU", "BATADAL")]} |
| **1D-CNN** | {tablo1_data[("1D-CNN", "SWAT")]} | N/A | {tablo1_data[("1D-CNN", "BATADAL")]} |
| **Automata** | {tablo1_data[("Automata", "SWAT")]} | N/A | {tablo1_data[("Automata", "BATADAL")]} |

## Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi (BATADAL Verisi)

| Model | Orijinal F1 | Gürültülü F1 | Unseen Det. Rate (Recall) | Unseen Map. Acc. (Accuracy) |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | {tablo2_data[("LSTM", "Original")]} | {tablo2_data[("LSTM", "Noisy")]} | {tablo2_data[("LSTM", "Det_Rate")]} | {tablo2_data[("LSTM", "Map_Acc")]} |
| **GRU** | {tablo2_data[("GRU", "Original")]} | {tablo2_data[("GRU", "Noisy")]} | {tablo2_data[("GRU", "Det_Rate")]} | {tablo2_data[("GRU", "Map_Acc")]} |
| **1D-CNN** | {tablo2_data[("1D-CNN", "Original")]} | {tablo2_data[("1D-CNN", "Noisy")]} | {tablo2_data[("1D-CNN", "Det_Rate")]} | {tablo2_data[("1D-CNN", "Map_Acc")]} |
| **Automata** | {tablo2_data[("Automata", "Original")]} | {tablo2_data[("Automata", "Noisy")]} | {tablo2_data[("Automata", "Det_Rate")]} | {tablo2_data[("Automata", "Map_Acc")]} |

## Tablo 3: Cross-Dataset Performans Karşılaştırması (F1-score)

| Train / Test | SWAT (SKAB) | WADI | BATADAL |
| :--- | :---: | :---: | :---: |
| **Train: SWAT (SKAB)** | | | |
| - LSTM | {tablo3_data[("LSTM", "SKAB", "SWAT")]} | N/A | {tablo3_data[("LSTM", "SKAB", "BATADAL")]} |
| - GRU | {tablo3_data[("GRU", "SKAB", "SWAT")]} | N/A | {tablo3_data[("GRU", "SKAB", "BATADAL")]} |
| - 1D-CNN | {tablo3_data[("1D-CNN", "SKAB", "SWAT")]} | N/A | {tablo3_data[("1D-CNN", "SKAB", "BATADAL")]} |
| - Automata | {tablo3_data[("Automata", "SKAB", "SWAT")]} | N/A | {tablo3_data[("Automata", "SKAB", "BATADAL")]} |
| **Train: WADI** | | | |
| - LSTM | N/A | N/A | N/A |
| - GRU | N/A | N/A | N/A |
| - 1D-CNN | N/A | N/A | N/A |
| - Automata | N/A | N/A | N/A |
| **Train: BATADAL** | | | |
| - LSTM | {tablo3_data[("LSTM", "BATADAL", "SWAT")]} | N/A | {tablo3_data[("LSTM", "BATADAL", "BATADAL")]} |
| - GRU | {tablo3_data[("GRU", "BATADAL", "SWAT")]} | N/A | {tablo3_data[("GRU", "BATADAL", "BATADAL")]} |
| - 1D-CNN | {tablo3_data[("1D-CNN", "BATADAL", "SWAT")]} | N/A | {tablo3_data[("1D-CNN", "BATADAL", "BATADAL")]} |
| - Automata | {tablo3_data[("Automata", "BATADAL", "SWAT")]} | N/A | {tablo3_data[("Automata", "BATADAL", "BATADAL")]} |

## Tablo 4: Automata Parametre Duyarlılık Analizi (F1-score)

| Parametre | Değer = 3 | Değer = 4 | Değer = 5 | Değer = 6 |
| :--- | :---: | :---: | :---: | :---: |
| **Window Size** | {tablo4_data["W_3"]} | {tablo4_data["W_4"]} | {tablo4_data["W_5"]} | {tablo4_data["W_6"]} |
| **Alphabet Size** | {tablo4_data["A_3"]} | {tablo4_data["A_4"]} | {tablo4_data["A_5"]} | {tablo4_data["A_6"]} |

## Tablo 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması

| Model | Training Time (sn) | Inference Time (sn) |
| :--- | :---: | :---: |
| **LSTM** | {tablo5_data[("LSTM", "Train")]} | {tablo5_data[("LSTM", "Inf")]} |
| **GRU** | {tablo5_data[("GRU", "Train")]} | {tablo5_data[("GRU", "Inf")]} |
| **1D-CNN** | {tablo5_data[("1D-CNN", "Train")]} | {tablo5_data[("1D-CNN", "Inf")]} |
| **Automata** | {tablo5_data[("Automata", "Train")]} | {tablo5_data[("Automata", "Inf")]} |
"""

    os.makedirs("results", exist_ok=True)
    with open("results/report_tables.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    logging.info("Markdown tabloları kaydedildi: results/report_tables.md")

    # ----------------------------------------------------
    # PDF OLUŞTURMA VE DOLDURMA (MERGE)
    # ----------------------------------------------------
    if not os.path.exists(pdf_template):
        logging.warning(f"Şablon PDF bulunamadı: {pdf_template}. PDF doldurma adımı atlanıyor.")
        return

    reader = pypdf.PdfReader(pdf_template)
    writer = pypdf.PdfWriter()

    # Sayfa 1 Overlay oluştur
    packet1 = io.BytesIO()
    can1 = canvas.Canvas(packet1, pagesize=A4)
    can1.setFont("Helvetica", 8)

    # Tablo 1 değerlerini çiz (Page 1)
    # SWAT, WADI, BATADAL column centers (aligned with preprinted ± signs)
    # LSTM, GRU, 1D-CNN, Automata y-koordinatları (tam eşleşme)
    t1_x = {"SWAT": 263.6, "WADI": 310.9, "BATADAL": 370.6}
    t1_y = {"LSTM": 360, "GRU": 346, "1D-CNN": 333, "Automata": 319}
    
    for m in models:
        y = t1_y[m]
        # SWAT: Split mean and std around the template's printed ± sign
        val_swat = tablo1_data[(m, "SWAT")]
        if " \u00b1 " in val_swat:
            mean_part, std_part = val_swat.split(" \u00b1 ")
            can1.drawRightString(t1_x["SWAT"] - 5, y, mean_part)
            can1.drawString(t1_x["SWAT"] + 5, y, std_part)
        else:
            can1.drawCentredString(t1_x["SWAT"], y, val_swat)
            
        # WADI: Clear template's ± sign and draw "N/A"
        can1.setFillColorRGB(1.0, 1.0, 1.0)
        can1.rect(t1_x["WADI"] - 8, y - 2, 16, 8, fill=True, stroke=False)
        can1.setFillColorRGB(0.0, 0.0, 0.0)
        can1.drawCentredString(t1_x["WADI"], y, "N/A")
        
        # BATADAL: Split mean and std around the template's printed ± sign
        val_bat = tablo1_data[(m, "BATADAL")]
        if " \u00b1 " in val_bat:
            mean_part, std_part = val_bat.split(" \u00b1 ")
            can1.drawRightString(t1_x["BATADAL"] - 5, y, mean_part)
            can1.drawString(t1_x["BATADAL"] + 5, y, std_part)
        else:
            can1.drawCentredString(t1_x["BATADAL"], y, val_bat)

    # Tablo 2 değerlerini çiz (Page 1)
    # Orijinal, Gürültülü, Det. Rate, Map. Acc. column centers
    t2_x = {"Original": 231.0, "Noisy": 286.0, "Det_Rate": 354.0, "Map_Acc": 414.0}
    t2_y = {"LSTM": 118, "GRU": 104, "1D-CNN": 91, "Automata": 78}
    
    for m in models:
        y = t2_y[m]
        can1.drawCentredString(t2_x["Original"], y, tablo2_data[(m, "Original")])
        can1.drawCentredString(t2_x["Noisy"], y, tablo2_data[(m, "Noisy")])
        can1.drawCentredString(t2_x["Det_Rate"], y, tablo2_data[(m, "Det_Rate")])
        can1.drawCentredString(t2_x["Map_Acc"], y, tablo2_data[(m, "Map_Acc")])

    can1.save()
    packet1.seek(0)
    overlay1 = pypdf.PdfReader(packet1)

    # Page 1'i birleştir
    page1 = reader.pages[0]
    page1.merge_page(overlay1.pages[0])
    writer.add_page(page1)

    # Sayfa 2 Overlay oluştur
    packet2 = io.BytesIO()
    can2 = canvas.Canvas(packet2, pagesize=A4)
    can2.setFont("Helvetica", 8)

    # Tablo 3 değerlerini çiz (Page 2)
    # SWAT, WADI, BATADAL column centers for Automata model
    t3_x = {"SWAT": 281.0, "WADI": 327.0, "BATADAL": 382.0}
    t3_y = {"SKAB": 642, "WADI": 628, "BATADAL": 615}
    
    for t_train in ["SKAB", "WADI", "BATADAL"]:
        y = t3_y[t_train]
        can2.drawCentredString(t3_x["SWAT"], y, tablo3_data.get(("Automata", t_train, "SWAT"), "N/A"))
        can2.drawCentredString(t3_x["WADI"], y, tablo3_data.get(("Automata", t_train, "WADI"), "N/A"))
        can2.drawCentredString(t3_x["BATADAL"], y, tablo3_data.get(("Automata", t_train, "BATADAL"), "N/A"))

    # Tablo 4 değerlerini çiz (Page 2)
    t4_x = {3: 238.0, 4: 304.0, 5: 369.0, 6: 435.0}
    t4_y = {"Window": 443, "Alphabet": 429}
    for val in [3, 4, 5, 6]:
        can2.drawCentredString(t4_x[val], t4_y["Window"], tablo4_data[f"W_{val}"])
        can2.drawCentredString(t4_x[val], t4_y["Alphabet"], tablo4_data[f"A_{val}"])

    # Tablo 5 değerlerini çiz (Page 2)
    t5_x = {"Train": 264.0, "Inf": 382.0}
    t5_y = {"LSTM": 338, "GRU": 324, "1D-CNN": 311, "Automata": 297}
    for m in models:
        y = t5_y[m]
        can2.drawCentredString(t5_x["Train"], y, tablo5_data[(m, "Train")])
        can2.drawCentredString(t5_x["Inf"], y, tablo5_data[(m, "Inf")])

    can2.save()
    packet2.seek(0)
    overlay2 = pypdf.PdfReader(packet2)

    # Page 2'i birleştir
    page2 = reader.pages[1]
    page2.merge_page(overlay2.pages[0])
    writer.add_page(page2)

    # PDF'yi kaydet
    with open(pdf_output, "wb") as f:
        writer.write(f)
    logging.info(f"PDF Raporu başarıyla dolduruldu ve kaydedildi: {pdf_output}")

if __name__ == "__main__":
    main()
