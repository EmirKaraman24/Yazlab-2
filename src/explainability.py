"""
Açıklanabilirlik (Explainability) Modülü.

Bu modül, anomali tespit sürecindeki karar adımlarını zorunlu JSON formatına
dönüştürür. Hem Olasılıksal Otomata hem de Derin Öğrenme modellerinin karar
süreçleri, insan tarafından okunabilir (human-readable) ve makine tarafından
işlenebilir (machine-processable) bir biçimde raporlanır.

Zorunlu JSON şeması şu bölümlerden oluşur:

- ``metadata``     : Rapor üst bilgileri (zaman damgası, model tipi, veri seti vb.)
- ``automata``     : Otomata karar süreci (varsa)
- ``deep_learning``: DL modeli karar süreci (varsa)
- ``decision``     : Nihai anomali kararı ve skor özeti

Kullanım Örneği
---------------
>>> from explainability import explain_automata_decision, save_explanation
>>> report = explain_automata_decision(
...     automata=fitted_automata,
...     sax_sequence=["abc", "xyz", "bcd"],
...     dataset_name="SKAB",
...     threshold=0.3,
... )
>>> save_explanation(report, output_path="results/explanation.json")
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Yardımcı
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Şu anki UTC zamanını ISO 8601 formatında döndürür."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: float, precision: int = 6) -> float:
    """
    Sayıyı JSON'a yazılabilir bir float değere dönüştürür.

    NaN veya sonsuzluk (inf) değerleri None olarak döndürülür; bu sayede
    JSON serileştirme hatası önlenir.
    """
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), precision)


# ---------------------------------------------------------------------------
# Otomata Açıklanabilirliği
# ---------------------------------------------------------------------------


def explain_automata_decision(
    automata,
    sax_sequence: list,
    dataset_name: str = "unknown",
    fold_id: Optional[str] = None,
    seed: Optional[int] = None,
    anomaly_threshold: float = 0.3,
) -> dict:
    """
    Olasılıksal Otomata'nın verilen SAX dizisi üzerindeki karar sürecini
    zorunlu JSON şemasına uygun bir Python sözlüğü olarak döndürür.

    Rapor aşağıdaki bilgileri içerir:

    - Hangi durumlardan hangilerine geçildi (geçiş zinciri).
    - Unseen (görülmemiş) örüntülerin hangi bilinen duruma eşlendiği ve
      bu eşleştirmede kullanılan Levenshtein mesafesi.
    - Her geçişin olasılığı.
    - Toplam yol olasılığı (path probability) ve güven skoru.
    - Güven skoruna dayalı anomali kararı.

    Parameters
    ----------
    automata : ProbabilisticAutomata
        ``fit()`` çağrılmış, hazır otomata nesnesi.
    sax_sequence : list of str
        İncelenecek SAX örüntü dizisi. En az 1 eleman içermelidir.
    dataset_name : str, optional
        Veri seti adı (ör. ``"SKAB"`` veya ``"BATADAL"``). Varsayılan: ``"unknown"``.
    fold_id : str or None, optional
        SKAB kullanılıyorsa fold numarası. Varsayılan: ``None``.
    seed : int or None, optional
        Deney tekrarı için kullanılan rastgele seed. Varsayılan: ``None``.
    anomaly_threshold : float, optional
        Güven skoru bu değerin altındaysa anomali kararı verilir.
        Varsayılan: ``0.3``.

    Returns
    -------
    dict
        Zorunlu JSON şemasını karşılayan Python sözlüğü.

    Raises
    ------
    RuntimeError
        Otomata henüz eğitilmemişse.
    ValueError
        ``sax_sequence`` boş veya None ise.
    """
    if not automata.is_fitted:
        raise RuntimeError(
            "Otomata modeli eğitilmedi. Lütfen önce fit() çağırın."
        )

    if not sax_sequence:
        raise ValueError("sax_sequence boş veya None olamaz.")

    # -----------------------------------------------------------------------
    # 1. Dizi boyunca durum çözümleme
    # -----------------------------------------------------------------------
    from automata import levenshtein_distance  # döngüsel import'u önlemek için

    resolved_sequence = []  # her adımın detaylı bilgisi
    for raw_state in sax_sequence:
        is_unseen = raw_state not in automata.state_to_id

        if is_unseen:
            resolved = automata.handle_unseen_state(raw_state)
            lev_dist = levenshtein_distance(raw_state, resolved)
        else:
            resolved = raw_state
            lev_dist = 0

        resolved_sequence.append(
            {
                "raw_state": raw_state,
                "resolved_state": resolved,
                "is_unseen": is_unseen,
                "levenshtein_distance": lev_dist,
            }
        )

    # -----------------------------------------------------------------------
    # 2. Adım adım geçiş (transition) detayları
    # -----------------------------------------------------------------------
    transitions = []
    for i in range(len(resolved_sequence) - 1):
        from_entry = resolved_sequence[i]
        to_entry = resolved_sequence[i + 1]

        from_id = automata.state_to_id[from_entry["resolved_state"]]
        to_id = automata.state_to_id[to_entry["resolved_state"]]
        prob = automata.transition_probabilities[from_id][to_id]

        transitions.append(
            {
                "step": i,
                "from_raw": from_entry["raw_state"],
                "from_resolved": from_entry["resolved_state"],
                "to_raw": to_entry["raw_state"],
                "to_resolved": to_entry["resolved_state"],
                "transition_probability": _safe_float(prob),
                "from_unseen": from_entry["is_unseen"],
                "to_unseen": to_entry["is_unseen"],
            }
        )

    # -----------------------------------------------------------------------
    # 3. Toplam yol olasılığı ve güven skoru
    # -----------------------------------------------------------------------
    path_probability = (
        automata.compute_path_probability(sax_sequence)
        if len(sax_sequence) >= 2
        else 1.0
    )

    confidence_score = (
        automata.compute_confidence_score(sax_sequence)
        if len(sax_sequence) >= 2
        else 1.0
    )

    # -----------------------------------------------------------------------
    # 4. Unseen özeti
    # -----------------------------------------------------------------------
    unseen_entries = [e for e in resolved_sequence if e["is_unseen"]]
    unseen_summary = [
        {
            "raw_state": e["raw_state"],
            "resolved_to": e["resolved_state"],
            "levenshtein_distance": e["levenshtein_distance"],
        }
        for e in unseen_entries
    ]

    # -----------------------------------------------------------------------
    # 5. Anomali kararı
    # -----------------------------------------------------------------------
    is_anomaly = confidence_score < anomaly_threshold
    decision_reason = (
        f"Güven skoru ({confidence_score:.4f}) < eşik ({anomaly_threshold})"
        if is_anomaly
        else f"Güven skoru ({confidence_score:.4f}) >= eşik ({anomaly_threshold})"
    )

    # -----------------------------------------------------------------------
    # 6. Tam raporu oluştur
    # -----------------------------------------------------------------------
    report = {
        "metadata": {
            "timestamp": _utc_now_iso(),
            "model_type": "ProbabilisticAutomata",
            "dataset": dataset_name,
            "fold_id": fold_id,
            "seed": seed,
            "sequence_length": len(sax_sequence),
            "num_known_states": automata.num_states,
            "anomaly_threshold": anomaly_threshold,
        },
        "automata": {
            "known_states": automata.states,
            "input_sequence": sax_sequence,
            "resolved_sequence": resolved_sequence,
            "transitions": transitions,
            "unseen_count": len(unseen_entries),
            "unseen_summary": unseen_summary,
            "path_probability": _safe_float(path_probability),
            "confidence_score": _safe_float(confidence_score),
        },
        "deep_learning": None,
        "decision": {
            "model_type": "ProbabilisticAutomata",
            "is_anomaly": is_anomaly,
            "confidence_score": _safe_float(confidence_score),
            "anomaly_threshold": anomaly_threshold,
            "reason": decision_reason,
        },
    }

    logging.info(
        "Otomata açıklama raporu oluşturuldu. "
        "Dizi uzunluğu: %d, Unseen: %d, Güven: %.4f, Anomali: %s",
        len(sax_sequence),
        len(unseen_entries),
        confidence_score,
        is_anomaly,
    )
    return report


# ---------------------------------------------------------------------------
# Derin Öğrenme Açıklanabilirliği
# ---------------------------------------------------------------------------


def explain_dl_decision(
    model_name: str,
    raw_probability: float,
    y_pred: int,
    metrics: dict,
    dataset_name: str = "unknown",
    fold_id: Optional[str] = None,
    seed: Optional[int] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Derin öğrenme modelinin (LSTM veya 1D-CNN) tek bir örnek üzerindeki
    karar sürecini zorunlu JSON şemasına uygun bir Python sözlüğü olarak
    döndürür.

    Parameters
    ----------
    model_name : str
        Model adı (ör. ``"LSTM"`` veya ``"1D-CNN"``).
    raw_probability : float
        Modelin sigmoid/softmax çıktısı (0.0 – 1.0 arası anomali skoru).
    y_pred : int
        Eşikle ikili sınıfa dönüştürülmüş tahmin (0 veya 1).
    metrics : dict
        ``compute_binary_metrics`` fonksiyonunun döndürdüğü metrik sözlüğü.
        ``'accuracy'``, ``'precision'``, ``'recall'``, ``'f1'`` vb. içerir.
    dataset_name : str, optional
        Veri seti adı. Varsayılan: ``"unknown"``.
    fold_id : str or None, optional
        Fold numarası (SKAB için). Varsayılan: ``None``.
    seed : int or None, optional
        Deney seed'i. Varsayılan: ``None``.
    threshold : float, optional
        İkili karar eşiği. Varsayılan: ``0.5``.

    Returns
    -------
    dict
        Zorunlu JSON şemasını karşılayan Python sözlüğü.
    """
    is_anomaly = bool(y_pred == 1)
    decision_reason = (
        f"Ham olasılık ({raw_probability:.4f}) >= eşik ({threshold})"
        if is_anomaly
        else f"Ham olasılık ({raw_probability:.4f}) < eşik ({threshold})"
    )

    report = {
        "metadata": {
            "timestamp": _utc_now_iso(),
            "model_type": model_name,
            "dataset": dataset_name,
            "fold_id": fold_id,
            "seed": seed,
            "decision_threshold": threshold,
        },
        "automata": None,
        "deep_learning": {
            "model_name": model_name,
            "raw_anomaly_probability": _safe_float(raw_probability),
            "binary_prediction": y_pred,
            "threshold": threshold,
            "performance_metrics": {
                "accuracy": _safe_float(metrics.get("accuracy")),
                "precision": _safe_float(metrics.get("precision")),
                "recall": _safe_float(metrics.get("recall")),
                "f1_score": _safe_float(metrics.get("f1")),
                "true_positives": metrics.get("tp"),
                "true_negatives": metrics.get("tn"),
                "false_positives": metrics.get("fp"),
                "false_negatives": metrics.get("fn"),
            },
        },
        "decision": {
            "model_type": model_name,
            "is_anomaly": is_anomaly,
            "confidence_score": _safe_float(raw_probability),
            "anomaly_threshold": threshold,
            "reason": decision_reason,
        },
    }

    logging.info(
        "DL açıklama raporu oluşturuldu. Model: %s, Olasılık: %.4f, Anomali: %s",
        model_name,
        raw_probability,
        is_anomaly,
    )
    return report


# ---------------------------------------------------------------------------
# Birleşik Açıklama (Ensemble)
# ---------------------------------------------------------------------------


def explain_combined_decision(
    automata_report: Optional[dict],
    dl_reports: Optional[list],
    dataset_name: str = "unknown",
    fold_id: Optional[str] = None,
    seed: Optional[int] = None,
    combination_strategy: str = "majority_vote",
) -> dict:
    """
    Otomata ve bir veya birden fazla DL modelinin kararlarını birleştirerek
    tek bir nihai açıklama raporu üretir.

    Parameters
    ----------
    automata_report : dict or None
        ``explain_automata_decision`` çıktısı. Yoksa ``None``.
    dl_reports : list of dict or None
        ``explain_dl_decision`` çıktılarının listesi. Yoksa ``None`` ya da boş liste.
    dataset_name : str, optional
        Veri seti adı. Varsayılan: ``"unknown"``.
    fold_id : str or None, optional
        Fold numarası. Varsayılan: ``None``.
    seed : int or None, optional
        Deney seed'i. Varsayılan: ``None``.
    combination_strategy : str, optional
        Karar birleştirme stratejisi. Desteklenen değer: ``"majority_vote"``.
        Varsayılan: ``"majority_vote"``.

    Returns
    -------
    dict
        Birleşik nihai JSON raporu.
    """
    all_decisions: list[bool] = []
    all_scores: list[float] = []
    model_summaries: list[dict] = []

    # Otomata kararı
    if automata_report is not None:
        aut_dec = automata_report["decision"]
        all_decisions.append(aut_dec["is_anomaly"])
        score = aut_dec.get("confidence_score")
        if score is not None:
            all_scores.append(score)

        model_summaries.append(
            {
                "model_type": "ProbabilisticAutomata",
                "is_anomaly": aut_dec["is_anomaly"],
                "confidence_score": aut_dec.get("confidence_score"),
                "reason": aut_dec.get("reason"),
            }
        )

    # DL model kararları
    if dl_reports:
        for dl_report in dl_reports:
            dl_dec = dl_report["decision"]
            all_decisions.append(dl_dec["is_anomaly"])
            score = dl_dec.get("confidence_score")
            if score is not None:
                all_scores.append(score)

            model_summaries.append(
                {
                    "model_type": dl_dec["model_type"],
                    "is_anomaly": dl_dec["is_anomaly"],
                    "confidence_score": dl_dec.get("confidence_score"),
                    "reason": dl_dec.get("reason"),
                }
            )

    # Birleştirme stratejisi
    if not all_decisions:
        final_is_anomaly = False
        strategy_explanation = "Hiçbir model çıktısı bulunamadı."
    elif combination_strategy == "majority_vote":
        anomaly_votes = sum(1 for d in all_decisions if d)
        final_is_anomaly = anomaly_votes > len(all_decisions) / 2
        strategy_explanation = (
            f"Çoğunluk oylaması: {anomaly_votes}/{len(all_decisions)} model anomali oyladı."
        )
    else:
        # Bilinmeyen strateji → çoğunluk oylaması ile düşeriz
        anomaly_votes = sum(1 for d in all_decisions if d)
        final_is_anomaly = anomaly_votes > len(all_decisions) / 2
        strategy_explanation = (
            f"Bilinmeyen strateji ('{combination_strategy}'); "
            f"çoğunluk oylamasına dönüldü. "
            f"{anomaly_votes}/{len(all_decisions)} anomali oyu."
        )

    mean_score = (
        _safe_float(sum(all_scores) / len(all_scores)) if all_scores else None
    )

    report = {
        "metadata": {
            "timestamp": _utc_now_iso(),
            "model_type": "ensemble",
            "dataset": dataset_name,
            "fold_id": fold_id,
            "seed": seed,
            "combination_strategy": combination_strategy,
            "num_models": len(all_decisions),
        },
        "automata": automata_report.get("automata") if automata_report else None,
        "deep_learning": (
            [r.get("deep_learning") for r in dl_reports] if dl_reports else None
        ),
        "decision": {
            "model_type": "ensemble",
            "is_anomaly": final_is_anomaly,
            "confidence_score": mean_score,
            "strategy": combination_strategy,
            "strategy_explanation": strategy_explanation,
            "model_summaries": model_summaries,
        },
    }

    logging.info(
        "Birleşik açıklama raporu oluşturuldu. Strateji: %s, Anomali: %s",
        combination_strategy,
        final_is_anomaly,
    )
    return report


# ---------------------------------------------------------------------------
# Kaydetme / Yükleme
# ---------------------------------------------------------------------------


def save_explanation(report: dict, output_path: str) -> None:
    """
    Açıklama raporunu JSON dosyasına kaydeder.

    Hedef klasör yoksa otomatik olarak oluşturulur.

    Parameters
    ----------
    report : dict
        ``explain_automata_decision``, ``explain_dl_decision`` veya
        ``explain_combined_decision`` tarafından üretilen sözlük.
    output_path : str
        Yazılacak ``.json`` dosyasının tam yolu.

    Raises
    ------
    TypeError
        Rapor JSON'a serileştirilemezse.
    """
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    logging.info("Açıklama raporu kaydedildi: %s", output_path)


def load_explanation(input_path: str) -> dict:
    """
    Daha önce kaydedilmiş bir JSON açıklama raporunu yükler.

    Parameters
    ----------
    input_path : str
        Okunacak ``.json`` dosyasının tam yolu.

    Returns
    -------
    dict
        Yüklenen açıklama raporu.

    Raises
    ------
    FileNotFoundError
        Dosya bulunamazsa.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Açıklama dosyası bulunamadı: {input_path}"
        )

    with open(input_path, "r", encoding="utf-8") as fp:
        report = json.load(fp)

    logging.info("Açıklama raporu yüklendi: %s", input_path)
    return report
