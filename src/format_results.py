"""
Sonuç Formatlama ve Çıktı Scripti (format_results.py).

Bu script, deneylerin CSV kayıtlarından (results/experiments.csv veya
results/inference_results_all_seeds.csv) iki farklı formatta çıktı üretir:

1. **SKAB — Fold Bazlı Tablo**:
   Her fold için model bazında ortalama metrikler (Accuracy, Precision,
   Recall, F1) hesaplanır ve ``results/skab_fold_results.csv`` olarak kaydedilir.

2. **BATADAL — Test Veri Seti Tablosu**:
   BATADAL test seti üzerindeki model bazında metrikler hesaplanır ve
   ``results/batadal_test_results.csv`` olarak kaydedilir.

Ayrıca her iki tablo konsola okunabilir biçimde yazdırılır.

Çalıştırmak için proje kökünde::

    python src/format_results.py

Opsiyonel argümanlar::

    python src/format_results.py [experiments_csv] [inference_csv]
"""

import csv
import logging
import os
import sys
from collections import defaultdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# CSV Okuma
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> list:
    """CSV dosyasını okur, bulunamazsa boş liste döner."""
    if not os.path.exists(path):
        logging.warning("Dosya bulunamadı: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _safe_float(value: str, default: float = 0.0) -> float:
    """String değeri float'a güvenli şekilde çevirir."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# SKAB — Fold Bazlı Formatlama
# ---------------------------------------------------------------------------

def format_skab_fold_results(
    records: list,
    output_path: str = "results/skab_fold_results.csv",
) -> list:
    """
    SKAB kayıtlarını fold ve model bazında gruplar, ortalama metrikler hesaplar.

    Her (fold_id, model_type) çifti için şu metriklerin ortalaması alınır:
    accuracy, precision, recall, f1, tp, tn, fp, fn.

    Parameters
    ----------
    records : list of dict
        CSV'den okunan ham kayıtlar.
    output_path : str
        Çıktı CSV dosyasının yolu.

    Returns
    -------
    list of dict
        Formatlanmış satırlar (fold x model).
    """
    # Sadece SKAB kayıtlarını filtrele
    skab_records = [
        r for r in records
        if "SKAB" in r.get("dataset", "") or "skab" in r.get("dataset", "").lower()
    ]

    if not skab_records:
        logging.warning("SKAB kaydı bulunamadı.")
        return []

    # (fold_id, model_type) → metrik listeler
    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    metric_cols = ["accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]

    for rec in skab_records:
        fold_id    = rec.get("fold_id") or rec.get("dataset", "")
        model_type = rec.get("model_type") or rec.get("model", "")
        key = (fold_id, model_type)
        for col in metric_cols:
            val = rec.get(col, "")
            if val != "":
                grouped[key][col].append(_safe_float(val))

    # Ortalama al, satır oluştur
    rows = []
    for (fold_id, model_type), metrics in sorted(grouped.items()):
        row = {
            "fold_id":    fold_id,
            "model_type": model_type,
        }
        for col in metric_cols:
            vals = metrics.get(col, [])
            if vals:
                row[col] = round(sum(vals) / len(vals), 4)
                row[f"{col}_count"] = len(vals)
            else:
                row[col] = ""
                row[f"{col}_count"] = 0
        rows.append(row)

    _write_csv(rows, output_path)
    logging.info("SKAB fold tablosu kaydedildi: %s (%d satır)", output_path, len(rows))
    return rows


# ---------------------------------------------------------------------------
# BATADAL — Test Veri Seti Formatlama
# ---------------------------------------------------------------------------

def format_batadal_test_results(
    records: list,
    output_path: str = "results/batadal_test_results.csv",
) -> list:
    """
    BATADAL kayıtlarını model bazında gruplar, test seti metriklerini raporlar.

    Her model için tüm seed ve senaryo kombinasyonlarındaki ortalama metrikler
    hesaplanır.

    Parameters
    ----------
    records : list of dict
        CSV'den okunan ham kayıtlar.
    output_path : str
        Çıktı CSV dosyasının yolu.

    Returns
    -------
    list of dict
        Formatlanmış satırlar (model bazında).
    """
    batadal_records = [
        r for r in records
        if "BATADAL" in r.get("dataset", "") or "batadal" in r.get("dataset", "").lower()
    ]

    if not batadal_records:
        logging.warning("BATADAL kaydı bulunamadı.")
        return []

    # (model_type, scenario) → metrik listeler
    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    metric_cols = ["accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]

    for rec in batadal_records:
        model_type = rec.get("model_type") or rec.get("model", "")
        scenario   = rec.get("scenario", "original")
        key = (model_type, scenario)
        for col in metric_cols:
            val = rec.get(col, "")
            if val != "":
                grouped[key][col].append(_safe_float(val))

    rows = []
    for (model_type, scenario), metrics in sorted(grouped.items()):
        row = {
            "dataset":    "BATADAL_test",
            "model_type": model_type,
            "scenario":   scenario,
        }
        for col in metric_cols:
            vals = metrics.get(col, [])
            if vals:
                row[col] = round(sum(vals) / len(vals), 4)
                row[f"{col}_count"] = len(vals)
            else:
                row[col] = ""
                row[f"{col}_count"] = 0
        rows.append(row)

    _write_csv(rows, output_path)
    logging.info("BATADAL test tablosu kaydedildi: %s (%d satır)", output_path, len(rows))
    return rows


# ---------------------------------------------------------------------------
# CSV Yazma
# ---------------------------------------------------------------------------

def _write_csv(rows: list, output_path: str) -> None:
    """Satır listesini CSV formatında diske yazar."""
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Konsol Yazdırma
# ---------------------------------------------------------------------------

def print_skab_table(rows: list) -> None:
    """SKAB fold tablosunu konsola yazdırır."""
    if not rows:
        print("SKAB: Gösterilecek veri yok.")
        return

    header = f"{'Fold':<20} {'Model':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
    sep = "-" * len(header)
    print(f"\n{'=== SKAB FOLD BAZLI SONUÇLAR ==='}")
    print(sep)
    print(header)
    print(sep)

    current_fold = None
    for row in sorted(rows, key=lambda r: (r.get("fold_id", ""), r.get("model_type", ""))):
        fold = row.get("fold_id", "")
        if fold != current_fold:
            if current_fold is not None:
                print()
            current_fold = fold

        def fmt(key):
            v = row.get(key, "")
            if v == "":
                return f"{'N/A':>7}"
            try:
                return f"{float(v):>7.4f}"
            except (ValueError, TypeError):
                return f"{str(v):>7}"

        def fmt_int(key):
            v = row.get(key, "")
            if v == "":
                return f"{'N/A':>5}"
            try:
                return f"{int(float(v)):>5}"
            except (ValueError, TypeError):
                return f"{str(v):>5}"

        print(
            f"{fold:<20} {row.get('model_type', ''):<15} "
            f"{fmt('accuracy')} {fmt('precision')} {fmt('recall')} {fmt('f1')} "
            f"{fmt_int('tp')} {fmt_int('tn')} {fmt_int('fp')} {fmt_int('fn')}"
        )
    print(sep + "\n")


def print_batadal_table(rows: list) -> None:
    """BATADAL test seti tablosunu konsola yazdırır."""
    if not rows:
        print("BATADAL: Gösterilecek veri yok.")
        return

    header = f"{'Model':<15} {'Senaryo':<12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
    sep = "-" * len(header)
    print(f"\n{'=== BATADAL TEST VERİ SETİ SONUÇLARI ==='}")
    print(sep)
    print(header)
    print(sep)

    for row in sorted(rows, key=lambda r: (r.get("model_type", ""), r.get("scenario", ""))):
        def fmt(key):
            v = row.get(key, "")
            if v == "":
                return f"{'N/A':>7}"
            try:
                return f"{float(v):>7.4f}"
            except (ValueError, TypeError):
                return f"{str(v):>7}"

        def fmt_int(key):
            v = row.get(key, "")
            if v == "":
                return f"{'N/A':>5}"
            try:
                return f"{int(float(v)):>5}"
            except (ValueError, TypeError):
                return f"{str(v):>5}"

        print(
            f"{row.get('model_type', ''):<15} {row.get('scenario', ''):<12} "
            f"{fmt('accuracy')} {fmt('precision')} {fmt('recall')} {fmt('f1')} "
            f"{fmt_int('tp')} {fmt_int('tn')} {fmt_int('fp')} {fmt_int('fn')}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Ana Giriş Noktası
# ---------------------------------------------------------------------------

def run_format_results(
    experiments_csv: str = "results/experiments.csv",
    inference_csv: Optional[str] = "results/inference_results_all_seeds.csv",
    skab_output: str = "results/skab_fold_results.csv",
    batadal_output: str = "results/batadal_test_results.csv",
) -> dict:
    """
    SKAB ve BATADAL sonuçlarını formatlayıp CSV olarak kaydeder.

    Önce ``inference_csv`` dosyasına, yoksa ``experiments_csv`` dosyasına bakar.

    Parameters
    ----------
    experiments_csv : str
        ExperimentLogger CSV yolu.
    inference_csv : str or None
        Inference pipeline CSV yolu (all_seeds).
    skab_output : str
        SKAB çıktı CSV yolu.
    batadal_output : str
        BATADAL çıktı CSV yolu.

    Returns
    -------
    dict
        "skab_rows" ve "batadal_rows" anahtarlı sonuç sözlüğü.
    """
    # Önce inference CSV'yi dene, sonra experiments CSV
    records = []
    for source in [inference_csv, experiments_csv]:
        if source:
            records = _read_csv(source)
            if records:
                logging.info("Kayıtlar yüklendi: %s (%d satır)", source, len(records))
                break

    if not records:
        logging.error("Hiçbir kayıt bulunamadı. Önce deneyleri çalıştırın.")
        return {"skab_rows": [], "batadal_rows": []}

    skab_rows    = format_skab_fold_results(records, output_path=skab_output)
    batadal_rows = format_batadal_test_results(records, output_path=batadal_output)

    print_skab_table(skab_rows)
    print_batadal_table(batadal_rows)

    return {"skab_rows": skab_rows, "batadal_rows": batadal_rows}


if __name__ == "__main__":
    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)

    experiments_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "experiments.csv")
    inference_csv   = sys.argv[2] if len(sys.argv) > 2 else os.path.join("results", "inference_results_all_seeds.csv")

    run_format_results(
        experiments_csv=experiments_csv,
        inference_csv=inference_csv,
        skab_output=os.path.join("results", "skab_fold_results.csv"),
        batadal_output=os.path.join("results", "batadal_test_results.csv"),
    )
