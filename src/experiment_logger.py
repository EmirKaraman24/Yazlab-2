"""
Deney Kayıt Sistemi (Experiment Logger).

Bu modül, tüm modellerin (ProbabilisticAutomata, LSTM, 1D-CNN) farklı
deney koşulları altındaki (seed, window_size, alphabet_size, senaryo vb.)
performans metriklerini tek bir CSV dosyasında birikimli olarak saklar.

Her kayıt satırı şu bilgileri içerir:

- Zaman damgası (timestamp)
- Deney kimliği (experiment_id) — tekil, oto-üretilmiş
- Model tipi, veri seti, fold, seed, senaryo
- Parametre değerleri (window_size, alphabet_size, threshold)
- Metrikler (accuracy, precision, recall, f1, tp, tn, fp, fn)
- Ek notlar (notes)

CSV dosyasına her çağrıda **ekleme (append)** modunda yazılır;
böylece önceki deneylerin kayıtları korunur.

Kullanım Örneği
---------------
>>> from experiment_logger import ExperimentLogger
>>> logger = ExperimentLogger(output_path="results/experiments.csv")
>>> logger.log(
...     model_type="LSTM",
...     dataset="SKAB",
...     metrics={"accuracy": 0.92, "precision": 0.88,
...               "recall": 0.85, "f1": 0.86,
...               "tp": 42, "tn": 50, "fp": 6, "fn": 7},
...     fold_id="fold_1",
...     seed=42,
...     scenario="original",
... )
>>> summary = logger.summary()
>>> best = logger.best_models(metric="f1", top_n=3)
"""

import csv
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Sabit sütun sırası (CSV başlığı)
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES: list[str] = [
    "experiment_id",
    "timestamp",
    "model_type",
    "dataset",
    "fold_id",
    "seed",
    "scenario",
    "window_size",
    "alphabet_size",
    "threshold",
    "n_samples",
    "n_anomalies_true",
    "n_anomalies_pred",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "notes",
]


# ---------------------------------------------------------------------------
# ExperimentLogger sınıfı
# ---------------------------------------------------------------------------


class ExperimentLogger:
    """
    Model karşılaştırmalarını CSV formatında birikimli olarak loglayan sistem.

    Parameters
    ----------
    output_path : str
        Kayıtların yazılacağı CSV dosyasının yolu.
        Klasör yoksa otomatik oluşturulur.
        Varsayılan: ``"results/experiments.csv"``.

    Attributes
    ----------
    output_path : str
        Kullanılan CSV dosya yolu.
    """

    def __init__(self, output_path: str = "results/experiments.csv") -> None:
        self.output_path = output_path
        self._ensure_file_exists()

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------

    def _ensure_file_exists(self) -> None:
        """
        CSV dosyası ve üst klasörü yoksa oluşturur; başlık satırını yazar.
        Dosya zaten mevcutsa dokunmaz (mevcut kayıtları korur).
        """
        parent_dir = os.path.dirname(self.output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if not os.path.exists(self.output_path):
            with open(self.output_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=_CSV_FIELDNAMES)
                writer.writeheader()
            logging.info(
                "Deney kayıt dosyası oluşturuldu: %s", self.output_path
            )

    @staticmethod
    def _utc_now_iso() -> str:
        """Şu anki UTC zamanını ISO 8601 formatında döndürür."""
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _generate_experiment_id() -> str:
        """UUID4 tabanlı kısa tekil kimlik üretir (ilk 8 karakter)."""
        return uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------
    # Ana API
    # ------------------------------------------------------------------

    def log(
        self,
        model_type: str,
        dataset: str,
        metrics: dict,
        fold_id: Optional[str] = None,
        seed: Optional[int] = None,
        scenario: str = "original",
        window_size: Optional[int] = None,
        alphabet_size: Optional[int] = None,
        threshold: Optional[float] = None,
        n_samples: Optional[int] = None,
        n_anomalies_true: Optional[int] = None,
        n_anomalies_pred: Optional[int] = None,
        notes: str = "",
        experiment_id: Optional[str] = None,
    ) -> str:
        """
        Bir deney sonucunu CSV dosyasına ekler.

        Parameters
        ----------
        model_type : str
            Model adı. Örn: ``"LSTM"``, ``"1D-CNN"``, ``"ProbabilisticAutomata"``.
        dataset : str
            Veri seti adı. Örn: ``"SKAB"``, ``"BATADAL"``.
        metrics : dict
            ``compute_binary_metrics`` çıktısı veya aynı anahtarları içeren
            herhangi bir sözlük. Beklenen anahtarlar:
            ``accuracy``, ``precision``, ``recall``, ``f1``,
            ``tp``, ``tn``, ``fp``, ``fn``.
        fold_id : str or None, optional
            SKAB fold numarası (ör. ``"fold_1"``). Varsayılan: ``None``.
        seed : int or None, optional
            Deney rastgele seed'i. Varsayılan: ``None``.
        scenario : str, optional
            Deney senaryosu. ``"original"``, ``"noisy"`` veya ``"unseen"``.
            Varsayılan: ``"original"``.
        window_size : int or None, optional
            SAX/PAA pencere boyutu. Varsayılan: ``None``.
        alphabet_size : int or None, optional
            SAX alfabe boyutu. Varsayılan: ``None``.
        threshold : float or None, optional
            Anomali karar eşiği. Varsayılan: ``None``.
        n_samples : int or None, optional
            Test örnek sayısı. Varsayılan: ``None``.
        n_anomalies_true : int or None, optional
            Gerçek anomali sayısı. Varsayılan: ``None``.
        n_anomalies_pred : int or None, optional
            Tahmin edilen anomali sayısı. Varsayılan: ``None``.
        notes : str, optional
            Serbest metin notu. Varsayılan: ``""``.
        experiment_id : str or None, optional
            Dışarıdan verilmek istenen kimlik. ``None`` ise oto-üretilir.

        Returns
        -------
        str
            Kaydın deney kimliği (experiment_id).

        Raises
        ------
        ValueError
            ``model_type`` veya ``dataset`` boş string ise.
        """
        if not model_type:
            raise ValueError("model_type boş olamaz.")
        if not dataset:
            raise ValueError("dataset boş olamaz.")

        exp_id = experiment_id or self._generate_experiment_id()

        row = {
            "experiment_id":    exp_id,
            "timestamp":        self._utc_now_iso(),
            "model_type":       model_type,
            "dataset":          dataset,
            "fold_id":          fold_id if fold_id is not None else "",
            "seed":             seed if seed is not None else "",
            "scenario":         scenario,
            "window_size":      window_size if window_size is not None else "",
            "alphabet_size":    alphabet_size if alphabet_size is not None else "",
            "threshold":        threshold if threshold is not None else "",
            "n_samples":        n_samples if n_samples is not None else "",
            "n_anomalies_true": n_anomalies_true if n_anomalies_true is not None else "",
            "n_anomalies_pred": n_anomalies_pred if n_anomalies_pred is not None else "",
            "accuracy":         metrics.get("accuracy", ""),
            "precision":        metrics.get("precision", ""),
            "recall":           metrics.get("recall", ""),
            "f1":               metrics.get("f1", ""),
            "tp":               metrics.get("tp", ""),
            "tn":               metrics.get("tn", ""),
            "fp":               metrics.get("fp", ""),
            "fn":               metrics.get("fn", ""),
            "notes":            notes,
        }

        with open(self.output_path, "a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=_CSV_FIELDNAMES)
            writer.writerow(row)

        logging.info(
            "Deney kaydedildi [%s] — Model: %s | Dataset: %s | "
            "fold: %s | seed: %s | F1: %s",
            exp_id, model_type, dataset, fold_id, seed,
            metrics.get("f1", "N/A"),
        )
        return exp_id

    def log_many(self, rows: list[dict]) -> list[str]:
        """
        Birden fazla deney sonucunu ardı ardına kaydeder.

        Parameters
        ----------
        rows : list of dict
            Her biri ``log()`` parametrelerini içeren sözlüklerin listesi.
            ``model_type``, ``dataset``, ``metrics`` anahtarları zorunludur.

        Returns
        -------
        list of str
            Kaydedilen her satıra ait experiment_id listesi.
        """
        ids: list[str] = []
        for row_kwargs in rows:
            exp_id = self.log(**row_kwargs)
            ids.append(exp_id)
        return ids

    # ------------------------------------------------------------------
    # Okuma / Analiz
    # ------------------------------------------------------------------

    def read_all(self) -> list[dict]:
        """
        CSV dosyasındaki tüm kayıtları liste olarak döndürür.

        Returns
        -------
        list of dict
            Her satır bir sözlük; anahtarlar CSV başlık isimleridir.
        """
        if not os.path.exists(self.output_path):
            return []

        with open(self.output_path, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            return list(reader)

    def summary(self) -> dict:
        """
        Kayıtlı deneylerin model ve veri seti bazında özet istatistiklerini
        hesaplar.

        Her (model_type, dataset) çifti için şu değerleri döndürür:

        - ``count``     : Toplam deney sayısı
        - ``mean_f1``   : Ortalama F1 skoru
        - ``mean_acc``  : Ortalama doğruluk (accuracy)
        - ``mean_prec`` : Ortalama kesinlik (precision)
        - ``mean_rec``  : Ortalama duyarlılık (recall)
        - ``best_f1``   : En yüksek F1 skoru
        - ``worst_f1``  : En düşük F1 skoru

        Returns
        -------
        dict
            ``{(model_type, dataset): {istatistikler}}`` formatında sözlük.
            Kayıt yoksa boş sözlük döner.
        """
        records = self.read_all()
        if not records:
            logging.warning("Özet için kayıt bulunamadı.")
            return {}

        # Grupla: (model_type, dataset) → f1 değerleri listesi
        groups: dict[tuple, list] = {}
        for rec in records:
            key = (rec.get("model_type", ""), rec.get("dataset", ""))
            try:
                f1_val = float(rec["f1"]) if rec.get("f1") != "" else None
                acc_val = float(rec["accuracy"]) if rec.get("accuracy") != "" else None
                prec_val = float(rec["precision"]) if rec.get("precision") != "" else None
                rec_val = float(rec["recall"]) if rec.get("recall") != "" else None
            except (ValueError, KeyError):
                continue

            if key not in groups:
                groups[key] = {"f1": [], "accuracy": [], "precision": [], "recall": []}

            if f1_val is not None:
                groups[key]["f1"].append(f1_val)
            if acc_val is not None:
                groups[key]["accuracy"].append(acc_val)
            if prec_val is not None:
                groups[key]["precision"].append(prec_val)
            if rec_val is not None:
                groups[key]["recall"].append(rec_val)

        summary_dict: dict = {}
        for (model_type, dataset), values in groups.items():
            f1_list = values["f1"]
            acc_list = values["accuracy"]
            prec_list = values["precision"]
            rec_list = values["recall"]

            summary_dict[(model_type, dataset)] = {
                "model_type":  model_type,
                "dataset":     dataset,
                "count":       len(f1_list),
                "mean_f1":     round(sum(f1_list) / len(f1_list), 4) if f1_list else None,
                "best_f1":     round(max(f1_list), 4) if f1_list else None,
                "worst_f1":    round(min(f1_list), 4) if f1_list else None,
                "mean_acc":    round(sum(acc_list) / len(acc_list), 4) if acc_list else None,
                "mean_prec":   round(sum(prec_list) / len(prec_list), 4) if prec_list else None,
                "mean_rec":    round(sum(rec_list) / len(rec_list), 4) if rec_list else None,
            }

        return summary_dict

    def best_models(
        self,
        metric: str = "f1",
        top_n: int = 5,
        dataset: Optional[str] = None,
    ) -> list[dict]:
        """
        Belirtilen metriğe göre en iyi deney kayıtlarını döndürür.

        Parameters
        ----------
        metric : str, optional
            Sıralama ölçütü. ``"f1"``, ``"accuracy"``, ``"precision"``
            veya ``"recall"`` olabilir. Varsayılan: ``"f1"``.
        top_n : int, optional
            Döndürülecek en iyi kayıt sayısı. Varsayılan: ``5``.
        dataset : str or None, optional
            Belirli bir veri setiyle sınırla. ``None`` ise tüm veri setleri
            dahil edilir.

        Returns
        -------
        list of dict
            Metriğe göre azalan sırada sıralanmış en iyi ``top_n`` kayıt.
        """
        valid_metrics = {"f1", "accuracy", "precision", "recall"}
        if metric not in valid_metrics:
            raise ValueError(
                f"Geçersiz metrik: '{metric}'. "
                f"Desteklenen değerler: {sorted(valid_metrics)}"
            )

        records = self.read_all()

        # Dataset filtresi
        if dataset is not None:
            records = [r for r in records if r.get("dataset") == dataset]

        # Metriği sayıya çevirilebilen kayıtları filtrele
        ranked: list[dict] = []
        for rec in records:
            raw = rec.get(metric, "")
            try:
                val = float(raw)
            except (ValueError, TypeError):
                continue
            ranked.append({**rec, f"_sort_{metric}": val})

        ranked.sort(key=lambda r: r[f"_sort_{metric}"], reverse=True)

        # Sıralama yardımcı anahtarını temizle
        for item in ranked:
            item.pop(f"_sort_{metric}", None)

        return ranked[:top_n]

    def print_summary_table(self) -> None:
        """
        Özet istatistikleri konsola hizalanmış bir tablo olarak yazdırır.
        """
        summary = self.summary()
        if not summary:
            print("Gösterilecek kayıt yok.")
            return

        header = (
            f"{'Model':<25} {'Dataset':<12} {'#':<5} "
            f"{'MeanF1':>7} {'BestF1':>7} {'MeanAcc':>8} {'MeanPrec':>9} {'MeanRec':>8}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)
        for stats in summary.values():
            print(
                f"{stats['model_type']:<25} {stats['dataset']:<12} "
                f"{stats['count']:<5} "
                f"{(stats['mean_f1'] or 0.0):>7.4f} "
                f"{(stats['best_f1'] or 0.0):>7.4f} "
                f"{(stats['mean_acc'] or 0.0):>8.4f} "
                f"{(stats['mean_prec'] or 0.0):>9.4f} "
                f"{(stats['mean_rec'] or 0.0):>8.4f}"
            )
        print(sep + "\n")
