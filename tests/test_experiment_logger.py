"""
ExperimentLogger için Birim Testleri.

Test Edilenler
--------------
- log()       : Tek kayıt yazma, zorunlu alan doğrulama, dönen experiment_id
- log_many()  : Toplu yazma
- read_all()  : CSV'den okuma
- summary()   : Grup istatistikleri (mean_f1, best_f1 vb.)
- best_models(): Metriğe göre sıralama ve dataset filtresi
- Dosya yokken otomatik başlık oluşturma
- Append davranışı (mevcut kayıtlar silinmez)
"""

import csv
import os
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from experiment_logger import ExperimentLogger, _CSV_FIELDNAMES


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

_SAMPLE_METRICS = {
    "accuracy": 0.92,
    "precision": 0.88,
    "recall": 0.85,
    "f1": 0.86,
    "tp": 42,
    "tn": 50,
    "fp": 6,
    "fn": 7,
}

_LOW_METRICS = {
    "accuracy": 0.60,
    "precision": 0.55,
    "recall": 0.50,
    "f1": 0.52,
    "tp": 10,
    "tn": 20,
    "fp": 8,
    "fn": 10,
}


@pytest.fixture
def logger(tmp_path):
    """Geçici dizinde ExperimentLogger örneği."""
    csv_path = str(tmp_path / "experiments.csv")
    return ExperimentLogger(output_path=csv_path)


# ---------------------------------------------------------------------------
# Dosya oluşturma
# ---------------------------------------------------------------------------


class TestFileCreation:

    def test_creates_file_on_init(self, tmp_path):
        """__init__ çağrısında CSV dosyası oluşturulmalıdır."""
        csv_path = str(tmp_path / "new.csv")
        assert not os.path.exists(csv_path)
        ExperimentLogger(output_path=csv_path)
        assert os.path.exists(csv_path)

    def test_creates_nested_directories(self, tmp_path):
        """Derinlemesine iç içe klasörler otomatik oluşturulmalıdır."""
        csv_path = str(tmp_path / "deep" / "nested" / "exp.csv")
        ExperimentLogger(output_path=csv_path)
        assert os.path.exists(csv_path)

    def test_header_written_on_creation(self, tmp_path):
        """Yeni dosyaya CSV başlığı yazılmalıdır."""
        csv_path = str(tmp_path / "exp.csv")
        ExperimentLogger(output_path=csv_path)
        with open(csv_path, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            assert reader.fieldnames == _CSV_FIELDNAMES

    def test_existing_file_not_overwritten(self, logger):
        """Mevcut dosyaya yeniden init yapıldığında içerik silinmemeli."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        # Aynı yola yeni bir logger oluştur
        logger2 = ExperimentLogger(output_path=logger.output_path)
        records = logger2.read_all()
        assert len(records) == 1  # kayıt hâlâ orada


# ---------------------------------------------------------------------------
# log()
# ---------------------------------------------------------------------------


class TestLog:

    def test_returns_experiment_id_string(self, logger):
        """log() bir string experiment_id döndürmelidir."""
        exp_id = logger.log(
            model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS
        )
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0

    def test_record_written_to_csv(self, logger):
        """log() sonrası kayıt CSV'de okunabilir olmalıdır."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        records = logger.read_all()
        assert len(records) == 1

    def test_all_metric_fields_present(self, logger):
        """Metrik değerleri CSV satırında doğru yazılmalıdır."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        rec = logger.read_all()[0]
        assert float(rec["accuracy"]) == pytest.approx(0.92)
        assert float(rec["f1"]) == pytest.approx(0.86)
        assert int(rec["tp"]) == 42

    def test_optional_fields_written(self, logger):
        """Opsiyonel alanlar verildiğinde CSV'ye yazılmalıdır."""
        logger.log(
            model_type="1D-CNN",
            dataset="BATADAL",
            metrics=_SAMPLE_METRICS,
            fold_id="fold_2",
            seed=42,
            scenario="noisy",
            window_size=4,
            alphabet_size=3,
            threshold=0.5,
            n_samples=100,
            n_anomalies_true=20,
            n_anomalies_pred=18,
            notes="test notu",
        )
        rec = logger.read_all()[0]
        assert rec["fold_id"] == "fold_2"
        assert rec["seed"] == "42"
        assert rec["scenario"] == "noisy"
        assert rec["window_size"] == "4"
        assert rec["alphabet_size"] == "3"
        assert rec["notes"] == "test notu"

    def test_custom_experiment_id_used(self, logger):
        """Dışarıdan verilen experiment_id kullanılmalıdır."""
        custom_id = "mytest01"
        exp_id = logger.log(
            model_type="LSTM",
            dataset="SKAB",
            metrics=_SAMPLE_METRICS,
            experiment_id=custom_id,
        )
        assert exp_id == custom_id
        rec = logger.read_all()[0]
        assert rec["experiment_id"] == custom_id

    def test_append_does_not_overwrite(self, logger):
        """İki ardışık log() çağrısı iki ayrı satır oluşturmalıdır."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        logger.log(model_type="1D-CNN", dataset="SKAB", metrics=_LOW_METRICS)
        records = logger.read_all()
        assert len(records) == 2

    def test_empty_model_type_raises(self, logger):
        """Boş model_type ValueError fırlatmalıdır."""
        with pytest.raises(ValueError):
            logger.log(model_type="", dataset="SKAB", metrics=_SAMPLE_METRICS)

    def test_empty_dataset_raises(self, logger):
        """Boş dataset ValueError fırlatmalıdır."""
        with pytest.raises(ValueError):
            logger.log(model_type="LSTM", dataset="", metrics=_SAMPLE_METRICS)

    def test_missing_metrics_keys_graceful(self, logger):
        """Eksik metrik anahtarları hata fırlatmadan boş string olarak yazılmalı."""
        logger.log(
            model_type="LSTM",
            dataset="SKAB",
            metrics={"accuracy": 0.9},  # f1, precision, recall yok
        )
        rec = logger.read_all()[0]
        assert float(rec["accuracy"]) == pytest.approx(0.9)
        assert rec["f1"] == ""  # eksik → boş string


# ---------------------------------------------------------------------------
# log_many()
# ---------------------------------------------------------------------------


class TestLogMany:

    def test_log_many_writes_all_rows(self, logger):
        """log_many() her girdi için bir satır yazmalıdır."""
        rows = [
            {"model_type": "LSTM", "dataset": "SKAB", "metrics": _SAMPLE_METRICS},
            {"model_type": "1D-CNN", "dataset": "SKAB", "metrics": _LOW_METRICS},
            {"model_type": "ProbabilisticAutomata", "dataset": "BATADAL",
             "metrics": _SAMPLE_METRICS},
        ]
        ids = logger.log_many(rows)
        assert len(ids) == 3
        assert len(logger.read_all()) == 3

    def test_log_many_returns_ids(self, logger):
        """log_many() her satır için ayrı bir experiment_id döndürmelidir."""
        rows = [
            {"model_type": "LSTM", "dataset": "SKAB", "metrics": _SAMPLE_METRICS},
            {"model_type": "LSTM", "dataset": "SKAB", "metrics": _SAMPLE_METRICS},
        ]
        ids = logger.log_many(rows)
        assert len(ids) == 2
        assert ids[0] != ids[1]  # ID'ler farklı olmalı


# ---------------------------------------------------------------------------
# read_all()
# ---------------------------------------------------------------------------


class TestReadAll:

    def test_read_empty_file(self, logger):
        """Hiç kayıt yoksa boş liste döndürülmelidir."""
        records = logger.read_all()
        assert records == []

    def test_read_returns_list_of_dicts(self, logger):
        """Kayıtlar dict listesi olarak döndürülmelidir."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        records = logger.read_all()
        assert isinstance(records, list)
        assert isinstance(records[0], dict)

    def test_read_nonexistent_file(self, tmp_path):
        """Dosya yoksa read_all() boş liste döndürmelidir."""
        csv_path = str(tmp_path / "nonexistent.csv")
        logger = ExperimentLogger.__new__(ExperimentLogger)
        logger.output_path = csv_path
        assert logger.read_all() == []


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:

    def test_summary_empty_returns_empty_dict(self, logger):
        """Kayıt yoksa boş sözlük döndürülmelidir."""
        assert logger.summary() == {}

    def test_summary_groups_by_model_and_dataset(self, logger):
        """Farklı model-dataset çiftleri ayrı gruplar oluşturmalıdır."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        logger.log(model_type="1D-CNN", dataset="SKAB", metrics=_LOW_METRICS)
        s = logger.summary()
        assert ("LSTM", "SKAB") in s
        assert ("1D-CNN", "SKAB") in s

    def test_summary_mean_f1_correct(self, logger):
        """Aynı model için iki kayıt varsa ortalama F1 doğru hesaplanmalıdır."""
        logger.log(model_type="LSTM", dataset="SKAB",
                   metrics={**_SAMPLE_METRICS, "f1": 0.80})
        logger.log(model_type="LSTM", dataset="SKAB",
                   metrics={**_SAMPLE_METRICS, "f1": 0.60})
        s = logger.summary()
        assert s[("LSTM", "SKAB")]["mean_f1"] == pytest.approx(0.70, abs=1e-4)

    def test_summary_best_f1_correct(self, logger):
        """best_f1 en yüksek F1 değerini içermelidir."""
        logger.log(model_type="LSTM", dataset="SKAB",
                   metrics={**_SAMPLE_METRICS, "f1": 0.90})
        logger.log(model_type="LSTM", dataset="SKAB",
                   metrics={**_SAMPLE_METRICS, "f1": 0.70})
        s = logger.summary()
        assert s[("LSTM", "SKAB")]["best_f1"] == pytest.approx(0.90, abs=1e-4)

    def test_summary_count_correct(self, logger):
        """count alanı, gruba ait kayıt sayısını göstermelidir."""
        for _ in range(3):
            logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        s = logger.summary()
        assert s[("LSTM", "SKAB")]["count"] == 3


# ---------------------------------------------------------------------------
# best_models()
# ---------------------------------------------------------------------------


class TestBestModels:

    def test_best_models_returns_list(self, logger):
        """best_models() bir liste döndürmelidir."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        result = logger.best_models()
        assert isinstance(result, list)

    def test_best_models_sorted_descending(self, logger):
        """Sonuçlar F1'e göre azalan sırada olmalıdır."""
        logger.log(model_type="LSTM", dataset="SKAB",
                   metrics={**_SAMPLE_METRICS, "f1": 0.70})
        logger.log(model_type="1D-CNN", dataset="SKAB",
                   metrics={**_SAMPLE_METRICS, "f1": 0.90})
        logger.log(model_type="ProbabilisticAutomata", dataset="SKAB",
                   metrics={**_SAMPLE_METRICS, "f1": 0.80})
        result = logger.best_models(metric="f1")
        f1_values = [float(r["f1"]) for r in result]
        assert f1_values == sorted(f1_values, reverse=True)

    def test_best_models_top_n_respected(self, logger):
        """top_n parametresi, döndürülen kayıt sayısını kısıtlamalıdır."""
        for i in range(5):
            logger.log(
                model_type="LSTM",
                dataset="SKAB",
                metrics={**_SAMPLE_METRICS, "f1": 0.5 + i * 0.05},
            )
        result = logger.best_models(top_n=3)
        assert len(result) <= 3

    def test_best_models_dataset_filter(self, logger):
        """dataset filtresi uygulandığında yalnızca o veri seti döndürülmeli."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        logger.log(model_type="LSTM", dataset="BATADAL", metrics=_LOW_METRICS)
        result = logger.best_models(dataset="BATADAL")
        assert all(r["dataset"] == "BATADAL" for r in result)

    def test_best_models_invalid_metric_raises(self, logger):
        """Geçersiz metrik adı ValueError fırlatmalıdır."""
        logger.log(model_type="LSTM", dataset="SKAB", metrics=_SAMPLE_METRICS)
        with pytest.raises(ValueError):
            logger.best_models(metric="invalid_metric")

    def test_best_models_empty_logger(self, logger):
        """Kayıt yokken best_models() boş liste döndürmelidir."""
        result = logger.best_models()
        assert result == []
