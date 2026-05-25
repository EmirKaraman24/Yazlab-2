"""
Değerlendirme ve Metrik Hesaplama Modülü.

Modellerin (Derin Öğrenme veya Otomata) performansını değerlendirmek için
kullanılan standart metrik fonksiyonlarını içerir.
"""

import numpy as np

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    İkili sınıflandırma (anomali tespiti) için temel metrikleri hesaplar.

    Accuracy, Precision, Recall ve F1-score değerlerini döndürür.
    Sıfıra bölme durumunda (örneğin hiç pozitif tahmin yoksa)
    ilgili metrik 0.0 olarak raporlanır.

    Parameters
    ----------
    y_true : np.ndarray
        Gerçek etiketler (0 veya 1).
    y_pred : np.ndarray
        Tahmin edilen etiketler (0 veya 1).

    Returns
    -------
    dict
        'accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn'
        anahtarlarını içeren sonuç sözlüğü.
    """
    # Numpy dizilerine çevirip 1D (flatten) yapıyoruz (güvenlik amaçlı)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
