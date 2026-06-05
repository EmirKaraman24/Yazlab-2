"""
İstatistiksel Anlamlılık Analizi Modülü (Statistical Tests).

Wilcoxon işaretli sıra testi ve McNemar testi ile modeller arası
istatistiksel karşılaştırma yapılır.

Kullanım
--------
>>> from src.statistical_tests import wilcoxon_test, mcnemar_test, run_all_comparisons
>>> result = wilcoxon_test([0.82, 0.85, 0.80], [0.78, 0.74, 0.79])
>>> result = mcnemar_test(y_true, y_pred_a, y_pred_b)
>>> comparisons = run_all_comparisons("results/experiments.csv", output_csv="results/stat_tests.csv")
"""

import csv
import logging
import os
from typing import Optional

import numpy as np
from scipy import stats
from scipy.stats import chi2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def wilcoxon_test(
    scores_a: list,
    scores_b: list,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict:
    """
    İki modelin fold/seed bazlı metrik listelerini Wilcoxon testi ile karşılaştırır.

    Parameters
    ----------
    scores_a, scores_b : list of float
        Her iki modelin eşleştirilmiş (fold veya seed başına) metrik skorları.
    alpha : float
        Anlamlılık düzeyi (varsayılan 0.05).
    alternative : str
        "two-sided", "greater" veya "less".

    Returns
    -------
    dict
        statistic, p_value, significant, mean_diff, conclusion.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Eşit uzunluk gerekli: {len(a)} != {len(b)}")
    if len(a) < 5:
        raise ValueError(f"En az 5 örnek gerekli, alınan: {len(a)}")

    diffs = a - b
    if np.all(diffs == 0):
        return {
            "statistic": 0.0, "p_value": 1.0, "significant": False,
            "alpha": alpha, "n_samples": len(a), "mean_diff": 0.0,
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "conclusion": "Modeller arasında fark yok (tüm farklar sıfır).",
        }

    stat, p_value = stats.wilcoxon(a, b, alternative=alternative)
    significant = bool(p_value < alpha)
    mean_diff = float(np.mean(diffs))

    if significant:
        winner = "Model A" if mean_diff > 0 else "Model B"
        conclusion = f"Anlamlı fark var (p={p_value:.4f} < α={alpha}). {winner} istatistiksel olarak daha iyi."
    else:
        conclusion = f"Anlamlı fark yok (p={p_value:.4f} >= α={alpha}). Modeller eşdeğer kabul edilir."

    logging.info("Wilcoxon | stat=%.4f | p=%.4f | significant=%s | mean_diff=%.4f",
                 stat, p_value, significant, mean_diff)

    return {
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "significant": significant,
        "alpha": alpha,
        "n_samples": len(a),
        "mean_a": round(float(np.mean(a)), 4),
        "mean_b": round(float(np.mean(b)), 4),
        "mean_diff": round(mean_diff, 6),
        "conclusion": conclusion,
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    alpha: float = 0.05,
    continuity_correction: bool = True,
) -> dict:
    """
    İki modelin ikili tahminlerini McNemar testi ile karşılaştırır.

    Parameters
    ----------
    y_true : np.ndarray
        Gerçek etiketler (0/1).
    y_pred_a, y_pred_b : np.ndarray
        Model A ve B'nin tahminleri (0/1).
    alpha : float
        Anlamlılık düzeyi (varsayılan 0.05).
    continuity_correction : bool
        Edwards süreklililik düzeltmesi (varsayılan True).

    Returns
    -------
    dict
        b, c (uyumsuz çiftler), statistic, p_value, significant, conclusion.
    """
    y_true   = np.asarray(y_true).ravel()
    y_pred_a = np.asarray(y_pred_a).ravel()
    y_pred_b = np.asarray(y_pred_b).ravel()

    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("y_true, y_pred_a ve y_pred_b aynı uzunlukta olmalıdır.")

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int((~correct_a & correct_b).sum())   # A yanlış, B doğru
    c = int((correct_a & ~correct_b).sum())   # A doğru, B yanlış
    n_discordant = b + c

    if n_discordant == 0:
        return {
            "b": b, "c": c, "statistic": 0.0, "p_value": 1.0,
            "significant": False, "alpha": alpha,
            "n_samples": len(y_true), "n_discordant": 0,
            "conclusion": "Uyumsuz çift yok; modeller aynı hataları yapıyor.",
        }

    if continuity_correction:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        stat = (b - c) ** 2 / (b + c)

    p_value = float(1 - chi2.cdf(stat, df=1))
    significant = bool(p_value < alpha)

    if significant:
        better = "Model A" if c < b else "Model B"
        conclusion = f"Anlamlı fark var (p={p_value:.4f} < α={alpha}). {better} istatistiksel olarak daha iyi."
    else:
        conclusion = f"Anlamlı fark yok (p={p_value:.4f} >= α={alpha}). Modeller eşdeğer kabul edilir."

    logging.info("McNemar | b=%d | c=%d | stat=%.4f | p=%.4f | significant=%s",
                 b, c, stat, p_value, significant)

    return {
        "b": b, "c": c,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "significant": significant,
        "alpha": alpha,
        "n_samples": len(y_true),
        "n_discordant": n_discordant,
        "conclusion": conclusion,
    }


def run_all_comparisons(
    experiments_csv: str,
    metric: str = "f1",
    alpha: float = 0.05,
    output_csv: Optional[str] = None,
) -> list:
    """
    ExperimentLogger CSV'sindeki tüm model çiftleri için Wilcoxon testini çalıştırır.

    Her (model_A, model_B, dataset) üçlüsü için fold/seed başına metrik değerleri
    kullanılarak Wilcoxon testi yapılır.

    Parameters
    ----------
    experiments_csv : str
        ``results/experiments.csv`` yolu.
    metric : str
        Karşılaştırılacak metrik (varsayılan "f1").
    alpha : float
        Anlamlılık düzeyi (varsayılan 0.05).
    output_csv : str or None
        Sonuçların yazılacağı CSV. None ise kaydedilmez.

    Returns
    -------
    list of dict
        Her model çifti için test sonuçları.
    """
    if not os.path.exists(experiments_csv):
        logging.warning("Deney CSV dosyası bulunamadı: %s", experiments_csv)
        return []

    with open(experiments_csv, "r", encoding="utf-8") as fp:
        records = list(csv.DictReader(fp))

    if not records:
        logging.warning("CSV dosyası boş.")
        return []

    # (dataset) → {model → [scores]}
    dataset_model_scores: dict[str, dict[str, list]] = {}
    for rec in records:
        dataset = rec.get("dataset", "")
        model   = rec.get("model_type", "")
        try:
            val = float(rec.get(metric, ""))
        except (ValueError, TypeError):
            continue
        dataset_model_scores.setdefault(dataset, {}).setdefault(model, []).append(val)

    results = []
    for dataset, model_scores in dataset_model_scores.items():
        model_names = sorted(model_scores.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]
                sa = model_scores[model_a]
                sb = model_scores[model_b]
                min_len = min(len(sa), len(sb))

                if min_len < 5:
                    logging.warning(
                        "Dataset=%s | %s vs %s: Yetersiz örnek (%d). Atlanıyor.",
                        dataset, model_a, model_b, min_len,
                    )
                    continue

                try:
                    res = wilcoxon_test(sa[:min_len], sb[:min_len], alpha=alpha)
                except Exception as exc:
                    logging.warning("Wilcoxon hatası (%s vs %s @ %s): %s",
                                    model_a, model_b, dataset, exc)
                    continue

                results.append({
                    "dataset":     dataset,
                    "model_a":     model_a,
                    "model_b":     model_b,
                    "metric":      metric,
                    "n_samples":   res["n_samples"],
                    "mean_a":      res["mean_a"],
                    "mean_b":      res["mean_b"],
                    "mean_diff":   res["mean_diff"],
                    "statistic":   res["statistic"],
                    "p_value":     res["p_value"],
                    "significant": res["significant"],
                    "alpha":       alpha,
                    "conclusion":  res["conclusion"],
                })
                logging.info("[%s] %s vs %s | p=%.4f | significant=%s",
                             dataset, model_a, model_b, res["p_value"], res["significant"])

    if output_csv and results:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logging.info("Test sonuçları kaydedildi: %s", output_csv)

    return results


def print_comparison_table(results: list) -> None:
    """Karşılaştırma sonuçlarını konsola hizalanmış tablo olarak yazdırır."""
    if not results:
        print("Gösterilecek sonuç yok.")
        return

    header = (f"{'Dataset':<15} {'Model A':<20} {'Model B':<20} "
              f"{'Metric':<8} {'MeanA':>6} {'MeanB':>6} {'p-value':>8} {'Sig?':>5}")
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for row in results:
        sig = "EVET" if row["significant"] else "HAYIR"
        print(
            f"{row['dataset']:<15} {row['model_a']:<20} {row['model_b']:<20} "
            f"{row['metric']:<8} {row['mean_a']:>6.4f} {row['mean_b']:>6.4f} "
            f"{row['p_value']:>8.4f} {sig:>5}"
        )
    print(sep + "\n")


if __name__ == "__main__":
    import sys
    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)

    metric = sys.argv[1] if len(sys.argv) > 1 else "f1"
    print(f"\n=== İstatistiksel Anlamlılık Analizi (metrik: {metric}) ===")

    results = run_all_comparisons(
        experiments_csv=os.path.join("results", "experiments.csv"),
        metric=metric,
        alpha=0.05,
        output_csv=os.path.join("results", "statistical_test_results.csv"),
    )
    print_comparison_table(results)

    if results:
        sig_count = sum(1 for r in results if r["significant"])
        print(f"Toplam {len(results)} karşılaştırmadan {sig_count} tanesi anlamlı (alpha=0.05).\n")
