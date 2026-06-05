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
from metrics import compute_binary_metrics
from sax import SAXTransformer
from automata import ProbabilisticAutomata

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


def _inject_noise(X: np.ndarray, noise_level: float = 0.5) -> np.ndarray:
    """Veriye Gaussian gürültü ekler."""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def _inject_unseen(X: np.ndarray) -> np.ndarray:
    """Veriye eğitimde görülmemiş (unseen) büyük offsetler ekler."""
    X_unseen = X.copy()
    n_samples = X.shape[0]
    n_inject = max(1, int(n_samples * 0.05))
    indices = np.random.choice(n_samples, n_inject, replace=False)
    X_unseen[indices] += 10.0
    return X_unseen


def _run_inference_on_split(
    manager: ModelWeightsManager,
    models_to_run: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_label: str,
    suffix: str = "",
    threshold: float = 0.5,
    scenario: str = "original",
    config: dict = None,
) -> list:
    """
    Verilen test bölümü üzerinde tüm modeller için inference çalıştırır.
    """
    sequence_length = X_test.shape[1]
    num_features    = X_test.shape[2]

    if scenario == "noisy":
        noise_level = 0.1
        if config is not None:
            noise_level = config.get("preprocessing", {}).get("gaussian_noise_std", 0.1)
        X_test = _inject_noise(X_test, noise_level=noise_level)
        dataset_label = f"{dataset_label}_noisy"
    elif scenario == "unseen":
        X_test = _inject_unseen(X_test)
        dataset_label = f"{dataset_label}_unseen"

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

        import time
        start_time = time.time()
        result = model.predict_model(X_test, y_true=y_test, threshold=threshold)
        inference_time = time.time() - start_time
        training_time = getattr(model, "training_time", 0.0)

        metrics = compute_binary_metrics(y_test, result["predictions"])

        row = {
            "dataset":   dataset_label,
            "model":     model_name,
            "suffix":    suffix,
            "n_samples": result["n_samples"],
            "n_anomalies_pred": result["n_anomalies"],
            "n_anomalies_true": int(y_test.sum()),
            "training_time": training_time,
            "inference_time": inference_time,
            **metrics,
        }
        rows.append(row)
        logging.info(
            "  [%s | %s] Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f (train_time=%.2fs, inf_time=%.2fs)",
            dataset_label, model_name,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"],
            training_time, inference_time,
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
    scenario: str = "original",
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
    scenario : str, optional
        "original", "noisy" veya "unseen" senaryosu.

    Returns
    -------
    list of dict
        Tüm fold'lara ait sonuç satırları.
    """
    logging.info("=" * 60)
    logging.info(f"SKAB inference başlatıldı. Senaryo: {scenario}")
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
        split_rows = _run_inference_on_split(
            manager, models_to_run, X_test, y_test,
            dataset_label=dataset_label,
            suffix=suffix,
            threshold=threshold,
            scenario=scenario,
            config=config,
        )
        all_rows.extend(split_rows)

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
    scenario: str = "original",
) -> list:
    """
    BATADAL veri setinin test kısmı üzerinde inference çalıştırır.

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
    scenario : str, optional
        "original", "noisy" veya "unseen" senaryosu.

    Returns
    -------
    list of dict
        Sonuç satırları.
    """
    logging.info("=" * 60)
    logging.info(f"BATADAL inference başlatıldı. Senaryo: {scenario}")
    logging.info("=" * 60)

    models_to_run = config.get("deep_learning_params", {}).get("models_to_run", ["LSTM", "1D-CNN"])
    threshold     = config.get("deep_learning_params", {}).get("prediction_threshold", 0.5)

    batadal_dir = os.path.join(processed_data_dir, "batadal")
    X_test, y_test = _load_numpy_split(batadal_dir, "test")
    if X_test is None:
        logging.warning("BATADAL test verisi bulunamadı. Inference atlanıyor.")
        return []

    dataset_label = "BATADAL_test"
    rows = _run_inference_on_split(
        manager, models_to_run, X_test, y_test,
        dataset_label=dataset_label,
        suffix=suffix,
        threshold=threshold,
        scenario=scenario,
        config=config,
    )
    logging.info("BATADAL inference tamamlandı. (%d satır)", len(rows))
    return rows


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


def run_automata_inference_on_split(
    sax_transformer: SAXTransformer,
    automata: ProbabilisticAutomata,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_label: str,
    suffix: str = "",
    scenario: str = "original",
    config: dict = None,
    training_time: float = 0.0,
) -> list:
    """
    Olasılıksal Otomata için verilen test bölümü üzerinde inference çalıştırır.
    """
    if scenario == "noisy":
        noise_level = 0.1
        if config is not None:
            noise_level = config.get("preprocessing", {}).get("gaussian_noise_std", 0.1)
        X_test = _inject_noise(X_test, noise_level=noise_level)
        dataset_label = f"{dataset_label}_noisy"
    elif scenario == "unseen":
        X_test = _inject_unseen(X_test)
        dataset_label = f"{dataset_label}_unseen"

    import time
    start_time = time.time()
    # Test verisini SAX dizisine dönüştür
    test_sax = sax_transformer.transform(X_test)
    
    # Otomata anomali skoru hesapla
    anomaly_threshold = 0.05
    y_pred = _evaluate_automata_per_step(automata, test_sax, anomaly_threshold=anomaly_threshold)
    inference_time = time.time() - start_time

    # Metrikleri hesapla
    metrics = compute_binary_metrics(y_test, y_pred)

    row = {
        "dataset":   dataset_label,
        "model":     "ProbabilisticAutomata",
        "suffix":    suffix,
        "n_samples": len(y_test),
        "n_anomalies_pred": int(np.sum(y_pred)),
        "n_anomalies_true": int(y_test.sum()),
        "training_time": training_time,
        "inference_time": inference_time,
        **metrics,
    }
    
    logging.info(
        "  [%s | Automata] Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f (train_time=%.2fs, inf_time=%.2fs)",
        dataset_label,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
        training_time, inference_time,
    )
    return [row]


def run_skab_automata_inference(
    config: dict,
    sax_transformer: SAXTransformer,
    automata: ProbabilisticAutomata,
    processed_data_dir: str,
    suffix: str = "",
    scenario: str = "original",
    training_time: float = 0.0,
) -> list:
    """
    SKAB veri setinin tüm fold'ları üzerinde Automata inference çalıştırır.
    """
    logging.info("=" * 60)
    logging.info(f"SKAB Automata inference başlatıldı. Senaryo: {scenario}")
    logging.info("=" * 60)

    n_splits      = config["preprocessing"].get("n_splits", 5)
    all_rows = []
    
    for fold_num in range(1, n_splits + 1):
        fold_dir = os.path.join(processed_data_dir, "skab", f"fold_{fold_num}")
        X_test, y_test = _load_numpy_split(fold_dir, "test")
        if X_test is None:
            continue

        dataset_label = f"SKAB_fold_{fold_num}"
        split_rows = run_automata_inference_on_split(
            sax_transformer, automata, X_test, y_test,
            dataset_label=dataset_label,
            suffix=suffix,
            scenario=scenario,
            config=config,
            training_time=training_time,
        )
        all_rows.extend(split_rows)

    logging.info("SKAB Automata inference tamamlandı. (%d satır)", len(all_rows))
    return all_rows


def run_batadal_automata_inference(
    config: dict,
    sax_transformer: SAXTransformer,
    automata: ProbabilisticAutomata,
    processed_data_dir: str,
    suffix: str = "",
    scenario: str = "original",
    training_time: float = 0.0,
) -> list:
    """
    BATADAL veri setinin test kısmı üzerinde Automata inference çalıştırır.
    """
    logging.info("=" * 60)
    logging.info(f"BATADAL Automata inference başlatıldı. Senaryo: {scenario}")
    logging.info("=" * 60)

    batadal_dir = os.path.join(processed_data_dir, "batadal")
    X_test, y_test = _load_numpy_split(batadal_dir, "test")
    if X_test is None:
        logging.warning("BATADAL test verisi bulunamadı. Inference atlanıyor.")
        return []

    dataset_label = "BATADAL_test"
    rows = run_automata_inference_on_split(
        sax_transformer, automata, X_test, y_test,
        dataset_label=dataset_label,
        suffix=suffix,
        scenario=scenario,
        config=config,
        training_time=training_time,
    )
    logging.info("BATADAL Automata inference tamamlandı. (%d satır)", len(rows))
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
