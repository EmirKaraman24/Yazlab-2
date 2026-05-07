# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

Bu proje, zaman serilerinde anomali tespiti için Derin Öğrenme modelleri (LSTM, 1D-CNN) ile yorumlanabilir (interpretable) Olasılıksal Otomata (Probabilistic Automata) modellerinin karşılaştırmalı analizini içermektedir.

## Veri Setleri
- **SKAB:** `data/valve1` ve `data/valve2` klasörlerinde yer almaktadır (Otomatik olarak indirilmiştir).
- **BATADAL:** `batadal.net` sunucuları şu an ulaşılamaz olduğundan `data/BATADAL/BATADAL_dataset04.csv` dosyası manuel olarak projeye dahil edilmelidir.

## Kurulum
```bash
pip install -r requirements.txt
```

## Yapılandırma
Projedeki tüm sabit değerler ve model hiperparametreleri `config.yaml` dosyası üzerinden yönetilmektedir.

## Kurallar
Proje geliştirilirken uyulması gereken katı kurallar `task.md` dosyasının en üstünde yer almaktadır. Lütfen kodlama yaparken düzenli olarak kontrol edin.
