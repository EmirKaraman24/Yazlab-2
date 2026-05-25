"""
Modeller Arası Performans Çıkarımı (Inference) Scripti.

Bu script; diske kaydedilmiş tüm derin öğrenme modellerini yükler,
test veri seti üzerinde tahmin (inference) çalıştırır ve modellerin
performansını karşılaştırmalı bir tablo olarak raporlar.

Desteklenen veri setleri: SKAB (fold bazında), BATADAL (tek test seti).
Desteklenen modeller  : config.yaml -> deep_learning_params.models_to_run

Çalıştırmak için proje kökünde::

    python src/run_inference.py

Çıktı CSV dosyası şuraya yazılır::

    results/inference_results.csv
"""

import logging
import os
import sys

import numpy as np
import yaml

# Proje kökünü Python yoluna ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dl_models import ModelWeightsManager, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------


def _load_numpy_split(directory: str, split: str) -> tuple:
    """
    ``{directory}/{split}_X.npy`` ve ``{directory}/{split}_y.npy``
    dosyalarını yükler.

    Parameters
    ----------
    directory : str
        Numpy dosyalarının bulunduğu klasör.
    split : str
        ``"train"``, ``"val"`` veya ``"test"`` gibi bölüm adı.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or (None, None)
        ``(X, y)`` ikilisi; dosyalar eksikse ``(None, None)`` döner.
    """
    x_path = os.path.join(directory, f"{split}_X.npy")
    y_path = os.path.join(directory, f"{split}_y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        logging.warning("Veri dosyası bulunamadı: %s veya %s", x_path, y_path)
        return None, None

    X = np.load(x_path)
    y = np.load(y_path)
    logging.info("  Yüklendi: %s — X=%s, y=%s", directory, X.shape, y.shape)
    return X, y


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    İkili sınıflandırma için temel metrikleri hesaplar.

    Accuracy, Precision, Recall ve F1-score değerlerini döndürür.
    Sıfıra bölme durumunda ilgili metrik ``0.0`` olarak raporlanır.

    Parameters
    ----------
    y_true : np.ndarray
        Gerçek etiketler (0 veya 1).
    y_pred : np.ndarray
        Tahmin edilen etiketler (0 veya 1).

    Returns
    -------
    dict
        ``accuracy``, ``precision``, ``recall``, ``f1`` anahtarlarını içerir.
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

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


def _run_inference_on_split(
    manager: ModelWeightsManager,
    models_to_run: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_label: str,
    suffix: str = "",
    threshold: float = 0.5,
) -> list:
    """
    Verilen test bölümü üzerinde tüm modeller için inference çalıştırır.

    Parameters
    ----------
    manager : ModelWeightsManager
        Kaydedilmiş modelleri yükleyen yönetici nesnesi.
    models_to_run : list of str
        Çalıştırılacak model adları (örn. ``["LSTM", "1D-CNN"]``).
    X_test : np.ndarray
        Test özellikleri, şekil: ``(n_samples, sequence_length, num_features)``.
    y_test : np.ndarray
        Gerçek etiketler, şekil: ``(n_samples,)``.
    dataset_label : str
        Sonuç tablosuna eklenecek veri seti etiketi (örn. ``"SKAB_fold_1"``).
    suffix : str, optional
        ModelWeightsManager ile kaydedilirken kullanılan suffix.
    threshold : float, optional
        Sigmoid → ikili etiket dönüşümü eşiği. Varsayılan: 0.5.

    Returns
    -------
    list of dict
        Her model için bir satır içeren sonuç listesi.
    """
    sequence_length = X_test.shape[1]
    num_features    = X_test.shape[2]

    rows = []
    for model_name in models_to_run:
        logging.info(
            "  [%s] %s modeli yükleniyor (suffix='%s')…",
            dataset_label, model_name, suffix,
        )
        try:
            model = manager.load(
                model_name,
                sequence_length=sequence_length,
                num_features=num_features,
                suffix=suffix,
            )
        except FileNotFoundError as exc:
            logging.warning("  Model dosyası bulunamadı, atlanıyor: %s", exc)
            continue

        result = model.predict_model(X_test, y_true=y_test, threshold=threshold)
        metrics = _compute_binary_metrics(y_test, result["predictions"])

        row = {
            "dataset":   dataset_label,
            "model":     model_name,
            "suffix":    suffix,
            "n_samples": result["n_samples"],
            "n_anomalies_pred": result["n_anomalies"],
            "n_anomalies_true": int(y_test.sum()),
            **metrics,
        }
        rows.append(row)
        logging.info(
            "  [%s | %s] Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f",
            dataset_label, model_name,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"],
        )

    return rows


def _save_results_csv(rows: list, output_path: str) -> None:
    """
    Sonuç satırlarını CSV formatında diske yazar.

    Parameters
    ----------
    rows : list of dict
        Her biri bir model–veri seti çifti için metrikleri içeren satırlar.
    output_path : str
        Yazılacak CSV dosyasının tam yolu.
    """
    import csv

    if not rows:
        logging.warning("Kaydedilecek sonuç yok; CSV oluşturulmadı.")
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Sonuçlar kaydedildi: %s", output_path)


def _print_summary_table(rows: list) -> None:
    """
    Sonuçları konsola hizalanmış bir tablo olarak yazdırır.

    Parameters
    ----------
    rows : list of dict
        Inference sonuç satırları.
    """
    if not rows:
        return

    header = f"{'Dataset':<25} {'Model':<10} {'Suffix':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for row in rows:
        print(
            f"{row['dataset']:<25} {row['model']:<10} {row['suffix']:<10} "
            f"{row['accuracy']:>6.4f} {row['precision']:>6.4f} "
            f"{row['recall']:>6.4f} {row['f1']:>6.4f}"
        )
    print(separator + "\n")


# ---------------------------------------------------------------------------
# SKAB Inference
# ---------------------------------------------------------------------------


def run_skab_inference(
    config: dict,
    manager: ModelWeightsManager,
    processed_data_dir: str,
    suffix: str = "",
) -> list:
    """
    SKAB veri setinin tüm fold'ları üzerinde inference çalıştırır.

    Her fold için ``{processed_data_dir}/skab/fold_{n}/test_X.npy`` ve
    ``test_y.npy`` dosyaları okunur. Dosya bulunamazsa fold atlanır.

    Parameters
    ----------
    config : dict
        config.yaml içeriği.
    manager : ModelWeightsManager
        Kaydedilmiş modelleri yükleyen yönetici nesnesi.
    processed_data_dir : str
        İşlenmiş verilerin kök klasörü (ör. ``"data/processed"``).
    suffix : str, optional
        Model dosyalarında kullanılan suffix.

    Returns
    -------
    list of dict
        Tüm fold'lara ait sonuç satırları.
    """
    logging.info("=" * 60)
    logging.info("SKAB inference başlatıldı.")
    logging.info("=" * 60)

    n_splits      = config["preprocessing"].get("n_splits", 5)
    models_to_run = config.get("deep_learning_params", {}).get("models_to_run", ["LSTM", "1D-CNN"])
    threshold     = config.get("deep_learning_params", {}).get("prediction_threshold", 0.5)

    all_rows = []
    for fold_num in range(1, n_splits + 1):
        fold_dir = os.path.join(processed_data_dir, "skab", f"fold_{fold_num}")
        X_test, y_test = _load_numpy_split(fold_dir, "test")
        if X_test is None:
            continue

        dataset_label = f"SKAB_fold_{fold_num}"
        rows = _run_inference_on_split(
            manager, models_to_run, X_test, y_test,
            dataset_label=dataset_label, suffix=suffix, threshold=threshold,
        )
        all_rows.extend(rows)

    logging.info("SKAB inference tamamlandı. (%d satır)", len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# BATADAL Inference
# ---------------------------------------------------------------------------


def run_batadal_inference(
    config: dict,
    manager: ModelWeightsManager,
    processed_data_dir: str,
    suffix: str = "",
) -> list:
    """
    BATADAL test seti üzerinde inference çalıştırır.

    ``{processed_data_dir}/batadal/test_X.npy`` ve ``test_y.npy``
    dosyaları okunur.

    Parameters
    ----------
    config : dict
        config.yaml içeriği.
    manager : ModelWeightsManager
        Kaydedilmiş modelleri yükleyen yönetici nesnesi.
    processed_data_dir : str
        İşlenmiş verilerin kök klasörü (ör. ``"data/processed"``).
    suffix : str, optional
        Model dosyalarında kullanılan suffix.

    Returns
    -------
    list of dict
        BATADAL test seti için sonuç satırları.
    """
    logging.info("=" * 60)
    logging.info("BATADAL inference başlatıldı.")
    logging.info("=" * 60)

    models_to_run = config.get("deep_learning_params", {}).get("models_to_run", ["LSTM", "1D-CNN"])
    threshold     = config.get("deep_learning_params", {}).get("prediction_threshold", 0.5)

    batadal_dir = os.path.join(processed_data_dir, "batadal")
    X_test, y_test = _load_numpy_split(batadal_dir, "test")
    if X_test is None:
        logging.warning("BATADAL test verisi bulunamadı; atlanıyor.")
        return []

    rows = _run_inference_on_split(
        manager, models_to_run, X_test, y_test,
        dataset_label="BATADAL_test", suffix=suffix, threshold=threshold,
    )
    logging.info("BATADAL inference tamamlandı. (%d satır)", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Ana Giriş Noktası
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Çalışma dizinini proje kökü olarak ayarla
    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)

    config = load_config("config.yaml")

    processed_data_dir = os.path.join("data", "processed")
    results_dir        = "results"
    output_csv         = os.path.join(results_dir, "inference_results.csv")

    # Kaydedilen modelleri yönetecek nesne (model_save_dir config'den okunur)
    manager = ModelWeightsManager(config_path="config.yaml")

    # Suffix boş bırakılır; seed bazlı deneylerde buraya seed değeri yazılır
    # (örn. suffix="seed42")
    suffix = ""

    all_rows: list = []

    # SKAB — fold bazında inference
    skab_rows = run_skab_inference(
        config, manager, processed_data_dir=processed_data_dir, suffix=suffix
    )
    all_rows.extend(skab_rows)

    # BATADAL — tek test seti inference
    batadal_rows = run_batadal_inference(
        config, manager, processed_data_dir=processed_data_dir, suffix=suffix
    )
    all_rows.extend(batadal_rows)

    # Sonuçları konsola ve CSV'ye yaz
    _print_summary_table(all_rows)
    _save_results_csv(all_rows, output_path=output_csv)

    logging.info("=" * 60)
    logging.info("Inference pipeline tamamlandı.")
    logging.info("=" * 60)
