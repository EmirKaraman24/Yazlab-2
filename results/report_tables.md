# Proje Deney Sonuçları Rapor Tabloları

## Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)

| Model | SWAT (SKAB) | WADI | BATADAL |
| :--- | :---: | :---: | :---: |
| **LSTM** | 0.0000 ± 0.0000 | N/A | 0.0000 ± 0.0000 |
| **GRU** | 0.0000 ± 0.0000 | N/A | 0.0000 ± 0.0000 |
| **1D-CNN** | 0.0000 ± 0.0000 | N/A | 0.0000 ± 0.0000 |
| **Automata** | 0.0173 ± 0.0000 | N/A | 0.0541 ± 0.0000 |

## Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi (BATADAL Verisi)

| Model | Orijinal F1 | Gürültülü F1 | Unseen Det. Rate (Recall) | Unseen Map. Acc. (Accuracy) |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | 0.0000 | 0.0000 | 0.0000 | 0.9040 |
| **GRU** | 0.0000 | 0.0000 | 0.0000 | 0.9040 |
| **1D-CNN** | 0.0000 | 0.0000 | 0.0000 | 0.9040 |
| **Automata** | 0.0541 | 0.0739 | 0.1250 | 0.8182 |

## Tablo 3: Cross-Dataset Performans Karşılaştırması (F1-score)

| Train / Test | SWAT (SKAB) | WADI | BATADAL |
| :--- | :---: | :---: | :---: |
| **Train: SWAT (SKAB)** | | | |
| - LSTM | 0.0000 | N/A | 0.0000 |
| - GRU | 0.0000 | N/A | 0.0000 |
| - 1D-CNN | 0.0000 | N/A | 0.0000 |
| - Automata | 0.0570 | N/A | 0.0973 |
| **Train: WADI** | | | |
| - LSTM | N/A | N/A | N/A |
| - GRU | N/A | N/A | N/A |
| - 1D-CNN | N/A | N/A | N/A |
| - Automata | N/A | N/A | N/A |
| **Train: BATADAL** | | | |
| - LSTM | 0.0000 | N/A | 0.0000 |
| - GRU | 0.0000 | N/A | 0.0000 |
| - 1D-CNN | 0.0000 | N/A | 0.0000 |
| - Automata | 0.1212 | N/A | 0.0541 |

## Tablo 4: Automata Parametre Duyarlılık Analizi (F1-score)

| Parametre | Değer = 3 | Değer = 4 | Değer = 5 | Değer = 6 |
| :--- | :---: | :---: | :---: | :---: |
| **Window Size** | 0.0655 | 0.0570 | 0.0757 | 0.1037 |
| **Alphabet Size** | 0.0570 | 0.1009 | 0.1593 | 0.2209 |

## Tablo 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması

| Model | Training Time (sn) | Inference Time (sn) |
| :--- | :---: | :---: |
| **LSTM** | 6.92 | 0.53 |
| **GRU** | 6.03 | 0.50 |
| **1D-CNN** | 4.45 | 0.35 |
| **Automata** | 0.05 | 0.02 |
