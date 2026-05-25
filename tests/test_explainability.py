"""
Explainability Modülü için Birim Testleri.

Test Edilenler
--------------
- explain_automata_decision: Otomata karar süreci JSON çıktısı
- explain_dl_decision: DL model karar süreci JSON çıktısı
- explain_combined_decision: Birleşik karar JSON çıktısı
- save_explanation / load_explanation: Dosyaya yaz / oku döngüsü
- Unseen durum raporlaması (Levenshtein mesafesi JSON'da doğru mu?)
- Anomali karar eşiği mantığı
"""

import json
import os
import sys
import tempfile

import pytest

# Proje kökünü ve src klasörünü Python yoluna ekle
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from automata import ProbabilisticAutomata
from explainability import (
    explain_automata_decision,
    explain_combined_decision,
    explain_dl_decision,
    load_explanation,
    save_explanation,
)


# ---------------------------------------------------------------------------
# Yardımcı fikstürler
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_automata():
    """Eğitilmiş küçük bir ProbabilisticAutomata örneği döndürür."""
    automata = ProbabilisticAutomata()
    train_sax = ["abc", "bcd", "cde", "abc", "bcd", "abc", "cde", "bcd"]
    automata.fit(train_sax)
    return automata


@pytest.fixture
def sample_metrics():
    """compute_binary_metrics formatında örnek metrik sözlüğü."""
    return {
        "accuracy": 0.9,
        "precision": 0.85,
        "recall": 0.80,
        "f1": 0.82,
        "tp": 40,
        "tn": 50,
        "fp": 7,
        "fn": 10,
    }


# ---------------------------------------------------------------------------
# explain_automata_decision testleri
# ---------------------------------------------------------------------------


class TestExplainAutomataDecision:
    """explain_automata_decision fonksiyonunun birim testleri."""

    def test_returns_dict(self, fitted_automata):
        """Fonksiyon bir dict döndürmelidir."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd", "cde"],
        )
        assert isinstance(report, dict)

    def test_required_top_level_keys(self, fitted_automata):
        """Zorunlu üst düzey anahtarlar mevcut olmalıdır."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd"],
        )
        for key in ("metadata", "automata", "deep_learning", "decision"):
            assert key in report, f"'{key}' anahtarı eksik"

    def test_metadata_fields(self, fitted_automata):
        """metadata bölümündeki zorunlu alanlar dolu olmalıdır."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd"],
            dataset_name="SKAB",
            fold_id="fold_1",
            seed=42,
        )
        meta = report["metadata"]
        assert meta["model_type"] == "ProbabilisticAutomata"
        assert meta["dataset"] == "SKAB"
        assert meta["fold_id"] == "fold_1"
        assert meta["seed"] == 42
        assert meta["sequence_length"] == 2
        assert "timestamp" in meta

    def test_automata_section_structure(self, fitted_automata):
        """automata bölümü gerekli tüm alt anahtarları içermelidir."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd", "cde"],
        )
        aut = report["automata"]
        for key in (
            "known_states",
            "input_sequence",
            "resolved_sequence",
            "transitions",
            "unseen_count",
            "unseen_summary",
            "path_probability",
            "confidence_score",
        ):
            assert key in aut, f"automata['{key}'] eksik"

    def test_deep_learning_is_none_for_automata_only(self, fitted_automata):
        """Yalnızca otomata raporu oluşturulduğunda deep_learning None olmalı."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd"],
        )
        assert report["deep_learning"] is None

    def test_no_unseen_states(self, fitted_automata):
        """Tüm durumlar biliniyorsa unseen_count 0 olmalı."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd", "cde"],
        )
        assert report["automata"]["unseen_count"] == 0
        assert report["automata"]["unseen_summary"] == []

    def test_unseen_state_detected_and_reported(self, fitted_automata):
        """Unseen durum, raporda doğru şekilde eşlenmiş ve raporlanmış olmalı."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "xyz", "bcd"],
        )
        aut = report["automata"]
        assert aut["unseen_count"] == 1

        unseen_entry = aut["unseen_summary"][0]
        assert unseen_entry["raw_state"] == "xyz"
        # Levenshtein mesafesi pozitif olmalı (xyz hiçbir bilinen durumla aynı değil)
        assert unseen_entry["levenshtein_distance"] > 0
        # Çözülen durum, bilinen durumlardan biri olmalı
        assert unseen_entry["resolved_to"] in fitted_automata.states

    def test_unseen_resolved_state_in_known_states(self, fitted_automata):
        """resolved_sequence'daki tüm resolved_state değerleri bilinen durumlarda olmalı."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "zzz", "bcd"],
        )
        for entry in report["automata"]["resolved_sequence"]:
            assert entry["resolved_state"] in fitted_automata.states

    def test_transition_count_matches_sequence_length(self, fitted_automata):
        """Geçiş sayısı, dizi uzunluğu - 1 olmalıdır."""
        seq = ["abc", "bcd", "cde"]
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=seq,
        )
        assert len(report["automata"]["transitions"]) == len(seq) - 1

    def test_transition_probabilities_are_floats(self, fitted_automata):
        """Her geçişin transition_probability değeri float olmalı."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd", "cde"],
        )
        for t in report["automata"]["transitions"]:
            assert isinstance(t["transition_probability"], float)

    def test_anomaly_decision_below_threshold(self, fitted_automata):
        """Güven skoru eşiğin altındaysa is_anomaly True olmalı."""
        # Eşiği çok yüksek ayarlarsak anomali kararı verilmeli
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd"],
            anomaly_threshold=1.0,  # her zaman anomali
        )
        assert report["decision"]["is_anomaly"] is True

    def test_no_anomaly_decision_above_threshold(self, fitted_automata):
        """Güven skoru eşiğin üzerindeyse is_anomaly False olmalı."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd"],
            anomaly_threshold=0.0,  # hiç anomali değil
        )
        assert report["decision"]["is_anomaly"] is False

    def test_empty_sequence_raises_value_error(self, fitted_automata):
        """Boş dizi ValueError fırlatmalıdır."""
        with pytest.raises(ValueError):
            explain_automata_decision(
                automata=fitted_automata,
                sax_sequence=[],
            )

    def test_unfitted_automata_raises_runtime_error(self):
        """Eğitilmemiş otomata RuntimeError fırlatmalıdır."""
        automata = ProbabilisticAutomata()
        with pytest.raises(RuntimeError):
            explain_automata_decision(
                automata=automata,
                sax_sequence=["abc"],
            )

    def test_json_serializable(self, fitted_automata):
        """Rapor JSON'a serileştirilebilir olmalıdır."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "xyz", "bcd"],
        )
        json_str = json.dumps(report)
        assert len(json_str) > 0

    def test_single_element_sequence(self, fitted_automata):
        """Tek elemanlı dizi hata üretmemeli; geçiş listesi boş olmalı."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc"],
        )
        assert report["automata"]["transitions"] == []
        assert report["automata"]["path_probability"] == 1.0


# ---------------------------------------------------------------------------
# explain_dl_decision testleri
# ---------------------------------------------------------------------------


class TestExplainDlDecision:
    """explain_dl_decision fonksiyonunun birim testleri."""

    def test_returns_dict(self, sample_metrics):
        """Fonksiyon bir dict döndürmelidir."""
        report = explain_dl_decision(
            model_name="LSTM",
            raw_probability=0.8,
            y_pred=1,
            metrics=sample_metrics,
        )
        assert isinstance(report, dict)

    def test_required_top_level_keys(self, sample_metrics):
        """Zorunlu üst düzey anahtarlar mevcut olmalıdır."""
        report = explain_dl_decision(
            model_name="LSTM",
            raw_probability=0.8,
            y_pred=1,
            metrics=sample_metrics,
        )
        for key in ("metadata", "automata", "deep_learning", "decision"):
            assert key in report

    def test_automata_is_none_for_dl_only(self, sample_metrics):
        """Yalnızca DL raporu oluşturulduğunda automata None olmalı."""
        report = explain_dl_decision(
            model_name="1D-CNN",
            raw_probability=0.3,
            y_pred=0,
            metrics=sample_metrics,
        )
        assert report["automata"] is None

    def test_anomaly_when_y_pred_is_1(self, sample_metrics):
        """y_pred=1 olduğunda is_anomaly True olmalı."""
        report = explain_dl_decision(
            model_name="LSTM",
            raw_probability=0.9,
            y_pred=1,
            metrics=sample_metrics,
        )
        assert report["decision"]["is_anomaly"] is True

    def test_no_anomaly_when_y_pred_is_0(self, sample_metrics):
        """y_pred=0 olduğunda is_anomaly False olmalı."""
        report = explain_dl_decision(
            model_name="LSTM",
            raw_probability=0.2,
            y_pred=0,
            metrics=sample_metrics,
        )
        assert report["decision"]["is_anomaly"] is False

    def test_performance_metrics_included(self, sample_metrics):
        """DL bölümündeki performance_metrics alanı dolu olmalı."""
        report = explain_dl_decision(
            model_name="LSTM",
            raw_probability=0.7,
            y_pred=1,
            metrics=sample_metrics,
        )
        perf = report["deep_learning"]["performance_metrics"]
        assert perf["accuracy"] == 0.9
        assert perf["precision"] == 0.85
        assert perf["recall"] == 0.8
        assert perf["f1_score"] == 0.82

    def test_json_serializable(self, sample_metrics):
        """DL raporu JSON'a serileştirilebilir olmalıdır."""
        report = explain_dl_decision(
            model_name="1D-CNN",
            raw_probability=0.6,
            y_pred=1,
            metrics=sample_metrics,
        )
        json_str = json.dumps(report)
        assert len(json_str) > 0

    def test_metadata_model_type(self, sample_metrics):
        """metadata.model_type, model adıyla eşleşmelidir."""
        report = explain_dl_decision(
            model_name="LSTM",
            raw_probability=0.5,
            y_pred=1,
            metrics=sample_metrics,
            dataset_name="BATADAL",
            seed=42,
        )
        assert report["metadata"]["model_type"] == "LSTM"
        assert report["metadata"]["dataset"] == "BATADAL"
        assert report["metadata"]["seed"] == 42


# ---------------------------------------------------------------------------
# explain_combined_decision testleri
# ---------------------------------------------------------------------------


class TestExplainCombinedDecision:
    """explain_combined_decision fonksiyonunun birim testleri."""

    def _make_automata_report(self, is_anomaly: bool) -> dict:
        """Sahte otomata raporu oluşturur."""
        return {
            "decision": {
                "model_type": "ProbabilisticAutomata",
                "is_anomaly": is_anomaly,
                "confidence_score": 0.2 if is_anomaly else 0.8,
                "reason": "test",
            },
            "automata": {},
        }

    def _make_dl_report(self, model_name: str, is_anomaly: bool) -> dict:
        """Sahte DL raporu oluşturur."""
        return {
            "decision": {
                "model_type": model_name,
                "is_anomaly": is_anomaly,
                "confidence_score": 0.9 if is_anomaly else 0.1,
                "reason": "test",
            },
            "deep_learning": {},
        }

    def test_majority_vote_anomaly(self):
        """2/3 model anomali oylarsa nihai karar anomali olmalı."""
        aut_report = self._make_automata_report(is_anomaly=True)
        dl_reports = [
            self._make_dl_report("LSTM", is_anomaly=True),
            self._make_dl_report("1D-CNN", is_anomaly=False),
        ]
        report = explain_combined_decision(
            automata_report=aut_report,
            dl_reports=dl_reports,
            combination_strategy="majority_vote",
        )
        assert report["decision"]["is_anomaly"] is True

    def test_majority_vote_no_anomaly(self):
        """1/3 model anomali oylarsa nihai karar normal olmalı."""
        aut_report = self._make_automata_report(is_anomaly=False)
        dl_reports = [
            self._make_dl_report("LSTM", is_anomaly=False),
            self._make_dl_report("1D-CNN", is_anomaly=True),
        ]
        report = explain_combined_decision(
            automata_report=aut_report,
            dl_reports=dl_reports,
            combination_strategy="majority_vote",
        )
        assert report["decision"]["is_anomaly"] is False

    def test_none_inputs_returns_no_anomaly(self):
        """Hiçbir giriş yoksa is_anomaly False olmalı."""
        report = explain_combined_decision(
            automata_report=None,
            dl_reports=None,
        )
        assert report["decision"]["is_anomaly"] is False

    def test_model_summaries_count(self):
        """model_summaries listesi toplam model sayısı kadar eleman içermeli."""
        aut_report = self._make_automata_report(is_anomaly=True)
        dl_reports = [
            self._make_dl_report("LSTM", is_anomaly=False),
        ]
        report = explain_combined_decision(
            automata_report=aut_report,
            dl_reports=dl_reports,
        )
        assert len(report["decision"]["model_summaries"]) == 2

    def test_required_keys_present(self):
        """Birleşik raporda zorunlu anahtarlar bulunmalıdır."""
        report = explain_combined_decision(
            automata_report=None,
            dl_reports=None,
        )
        for key in ("metadata", "automata", "deep_learning", "decision"):
            assert key in report

    def test_json_serializable(self):
        """Birleşik rapor JSON'a serileştirilebilir olmalıdır."""
        aut_report = self._make_automata_report(is_anomaly=True)
        dl_reports = [self._make_dl_report("LSTM", is_anomaly=False)]
        report = explain_combined_decision(
            automata_report=aut_report,
            dl_reports=dl_reports,
        )
        json_str = json.dumps(report)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# save_explanation / load_explanation testleri
# ---------------------------------------------------------------------------


class TestSaveLoadExplanation:
    """Dosyaya yaz / oku işlemlerinin birim testleri."""

    def test_save_and_load_roundtrip(self, fitted_automata):
        """Kaydedilen rapor eksiksiz geri yüklenmelidir."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd", "cde"],
            dataset_name="SKAB",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "report.json")
            save_explanation(report, output_path)
            loaded = load_explanation(output_path)

        assert loaded["metadata"]["dataset"] == "SKAB"
        assert loaded["decision"]["model_type"] == "ProbabilisticAutomata"
        assert "automata" in loaded

    def test_save_creates_parent_directories(self, fitted_automata):
        """Hedef klasör yoksa otomatik oluşturulmalıdır."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_path = os.path.join(
                tmp_dir, "deep", "nested", "report.json"
            )
            save_explanation(report, nested_path)
            assert os.path.exists(nested_path)

    def test_load_nonexistent_file_raises_error(self):
        """Var olmayan dosyayı yüklemeye çalışmak FileNotFoundError fırlatmalı."""
        with pytest.raises(FileNotFoundError):
            load_explanation("/nonexistent/path/report.json")

    def test_saved_file_is_valid_json(self, fitted_automata):
        """Kaydedilen dosya geçerli JSON içermelidir."""
        report = explain_automata_decision(
            automata=fitted_automata,
            sax_sequence=["abc", "bcd"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "report.json")
            save_explanation(report, output_path)

            with open(output_path, "r", encoding="utf-8") as fp:
                loaded_json = json.load(fp)

        assert isinstance(loaded_json, dict)
