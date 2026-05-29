"""
Görselleştirme (Visualization) Modülü.

Bu modül, Olasılıksal Otomata ve sınıflandırma modellerinin analiz
sonuçlarını görsel olarak sunmak için üç temel fonksiyon sağlar:

1. ``plot_automata_state_diagram``
   ProbabilisticAutomata'nın durum (state) diyagramını yönlü çizge (directed
   graph) olarak çizer. Kenar kalınlıkları geçiş olasılıklarıyla orantılıdır.

2. ``plot_transition_heatmap``
   Geçiş olasılık matrisini (Transition Probability Matrix) ısı haritası
   (heatmap) olarak görselleştirir. Renk yoğunluğu olasılık büyüklüğünü
   gösterir.

3. ``plot_confusion_matrix_report``
   İkili sınıflandırma sonuçlarından confusion matrix ve Accuracy / Precision
   / Recall / F1-score metriklerini yan yana tablo+grafik olarak sunar.

Tüm fonksiyonlar dosyaya kaydetmeyi destekler (``output_path`` parametresi).
Matplotlib ve NetworkX kütüphanelerine ihtiyaç duyar.
"""

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # GUI olmayan ortamlar için güvenli backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Dahili yardımcılar
# ---------------------------------------------------------------------------


def _ensure_output_dir(output_path: str) -> None:
    """Hedef dizin yoksa oluşturur."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Automata State Diyagramı
# ---------------------------------------------------------------------------


def plot_automata_state_diagram(
    automata,
    title: str = "Probabilistic Automata — State Diagram",
    output_path: Optional[str] = None,
    min_prob: float = 0.01,
    figsize: tuple = (12, 9),
) -> None:
    """
    ProbabilisticAutomata'nın durum (state) diyagramını yönlü çizge (directed
    graph) olarak çizer.

    Düğümler otomata durumlarını, yönlü kenarlar ise olası geçişleri temsil
    eder. Kenar kalınlığı ve opaklığı geçiş olasılığıyla orantılıdır; ``min_prob``
    altındaki geçişler gösterilmez (grafik okunabilirliğini artırmak için).

    NetworkX yerine saf Matplotlib kullanılarak çizilir; böylece
    ``networkx`` bağımlılığı olmaksızın çalışır.

    Parameters
    ----------
    automata : ProbabilisticAutomata
        ``fit()`` çağrılmış, hazır otomata nesnesi.
    title : str, optional
        Grafik başlığı. Varsayılan: ``"Probabilistic Automata — State Diagram"``.
    output_path : str or None, optional
        Grafik dosya yolu (PNG/PDF). ``None`` ise ekranda gösterir.
    min_prob : float, optional
        Bu değerin altındaki geçiş olasılıkları gösterilmez. Varsayılan: ``0.01``.
    figsize : tuple, optional
        Şekil boyutu (genişlik, yükseklik). Varsayılan: ``(12, 9)``.

    Raises
    ------
    RuntimeError
        Otomata henüz eğitilmemişse.
    """
    if not automata.is_fitted:
        raise RuntimeError("Otomata modeli eğitilmedi. Lütfen önce fit() çağırın.")

    states = automata.states
    n = len(states)
    prob_matrix = automata.transition_probabilities

    if n == 0:
        logging.warning("Gösterilecek durum bulunamadı. State diagram atlandı.")
        return

    # --- Düğüm konumları: çember üzerinde eşit aralıklı ---
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 3.0
    pos_x = radius * np.cos(angles)
    pos_y = radius * np.sin(angles)
    positions = {state: (pos_x[i], pos_y[i]) for i, state in enumerate(states)}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#2c3e50")
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    node_radius = 0.38

    # --- Kenarları çiz ---
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            prob = prob_matrix[i][j]
            if prob < min_prob:
                continue

            x0, y0 = positions[from_state]
            x1, y1 = positions[to_state]

            # Öz-döngü (self-loop)
            if i == j:
                loop_center_x = x0 + node_radius * 1.8 * np.cos(angles[i])
                loop_center_y = y0 + node_radius * 1.8 * np.sin(angles[i])
                loop = plt.Circle(
                    (loop_center_x, loop_center_y),
                    radius=node_radius * 0.9,
                    color="#3498db",
                    fill=False,
                    linewidth=max(0.5, prob * 5),
                    alpha=min(1.0, 0.3 + prob),
                    linestyle="--",
                )
                ax.add_patch(loop)
                # Olasılık etiketi
                ax.text(
                    loop_center_x,
                    loop_center_y + node_radius * 0.9 + 0.1,
                    f"{prob:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2980b9",
                    fontweight="bold",
                )
            else:
                # Kenar vektörü ve yönü
                dx = x1 - x0
                dy = y1 - y0
                dist = np.hypot(dx, dy)
                ux = dx / dist
                uy = dy / dist

                # Düğüm sınırından başlat/bitir
                sx = x0 + ux * node_radius
                sy = y0 + uy * node_radius
                ex = x1 - ux * node_radius
                ey = y1 - uy * node_radius

                # Eğim (curvature) — üst üste binen çift yönlü okları ayırmak için
                mid_x = (sx + ex) / 2
                mid_y = (sy + ey) / 2
                perp_x = -uy * 0.25
                perp_y = ux * 0.25

                lw = max(0.5, prob * 6)
                alpha = min(1.0, 0.25 + prob * 1.5)

                ax.annotate(
                    "",
                    xy=(ex, ey),
                    xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#e74c3c",
                        lw=lw,
                        alpha=alpha,
                        connectionstyle="arc3,rad=0.15",
                    ),
                )

                # Olasılık etiketi — kenar orta noktası üzerinde
                label_x = mid_x + perp_x
                label_y = mid_y + perp_y
                ax.text(
                    label_x,
                    label_y,
                    f"{prob:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#c0392b",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                    ),
                )

    # --- Düğümleri çiz ---
    state_colors = plt.cm.Set3(np.linspace(0, 1, n))
    for i, state in enumerate(states):
        x, y = positions[state]
        circle = plt.Circle(
            (x, y),
            node_radius,
            color=state_colors[i],
            ec="#2c3e50",
            linewidth=2,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            state,
            ha="center",
            va="center",
            fontsize=max(6, 10 - n // 4),
            fontweight="bold",
            color="#2c3e50",
            zorder=4,
        )

    # --- Eksen sınırları ---
    margin = radius + node_radius + 1.5
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    # --- Açıklama kutusu ---
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="#e74c3c", label="Geçiş (transition)"),
        mpatches.Patch(facecolor="white", edgecolor="#3498db", linestyle="--",
                       label="Öz-döngü (self-loop)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=9,
        framealpha=0.85,
        edgecolor="#bdc3c7",
    )

    plt.tight_layout()

    if output_path:
        _ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logging.info("State diagram kaydedildi: %s", output_path)
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Transition Probability Heatmap
# ---------------------------------------------------------------------------


def plot_transition_heatmap(
    automata,
    title: str = "Transition Probability Heatmap",
    output_path: Optional[str] = None,
    cmap: str = "YlOrRd",
    figsize: Optional[tuple] = None,
    annot_fontsize: int = 8,
) -> None:
    """
    Geçiş olasılık matrisini (Transition Probability Matrix) ısı haritası
    (heatmap) olarak görselleştirir.

    Her hücre (i, j), i. durumdan j. duruma geçiş olasılığını gösterir.
    Her satır toplamı 1.0'e eşittir.

    Parameters
    ----------
    automata : ProbabilisticAutomata
        ``fit()`` çağrılmış, hazır otomata nesnesi.
    title : str, optional
        Grafik başlığı. Varsayılan: ``"Transition Probability Heatmap"``.
    output_path : str or None, optional
        Grafik dosya yolu (PNG/PDF). ``None`` ise ekranda gösterir.
    cmap : str, optional
        Matplotlib renk haritası. Varsayılan: ``"YlOrRd"``.
    figsize : tuple or None, optional
        Şekil boyutu. ``None`` ise durum sayısına göre otomatik hesaplanır.
    annot_fontsize : int, optional
        Hücre içi yazı boyutu. Varsayılan: ``8``.

    Raises
    ------
    RuntimeError
        Otomata henüz eğitilmemişse.
    """
    if not automata.is_fitted:
        raise RuntimeError("Otomata modeli eğitilmedi. Lütfen önce fit() çağırın.")

    states = automata.states
    n = len(states)
    prob_matrix = np.array(automata.transition_probabilities)

    if n == 0:
        logging.warning("Gösterilecek durum bulunamadı. Heatmap atlandı.")
        return

    # Otomatik figsize: en az 6x5, durum sayısıyla büyür
    if figsize is None:
        cell_size = max(0.7, min(1.4, 12 / n))
        figsize = (max(6, n * cell_size + 2), max(5, n * cell_size + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    im = ax.imshow(prob_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Eksen etiketleri
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(states, rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    ax.set_yticklabels(states, fontsize=max(6, 10 - n // 5))
    ax.set_xlabel("Hedef Durum (To State)", fontsize=11, labelpad=8)
    ax.set_ylabel("Kaynak Durum (From State)", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color="#2c3e50")

    # Hücre içi olasılık değerleri
    for i in range(n):
        for j in range(n):
            val = prob_matrix[i, j]
            text_color = "white" if val > 0.6 else "#2c3e50"
            if val >= 0.005:   # çok küçük değerleri yazmayı atla
                ax.text(
                    j, i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    color=text_color,
                    fontweight="bold",
                )

    # Renk çubuğu
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Geçiş Olasılığı", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Izgara çizgileri — hücre sınırlarını belirginleştir
    for x in np.arange(-0.5, n, 1):
        ax.axhline(x, color="white", linewidth=0.5)
        ax.axvline(x, color="white", linewidth=0.5)

    plt.tight_layout()

    if output_path:
        _ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logging.info("Transition heatmap kaydedildi: %s", output_path)
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Confusion Matrix ve Performans Metrikleri Raporu
# ---------------------------------------------------------------------------


def plot_confusion_matrix_report(
    y_true,
    y_pred,
    model_name: str = "Model",
    dataset_name: str = "Dataset",
    class_names: Optional[list] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> dict:
    """
    İkili sınıflandırma sonuçlarından confusion matrix ve performans
    metriklerini (Accuracy, Precision, Recall, F1-score) yan yana grafik ve
    tablo olarak görselleştirir.

    Sol panel: renklendirilmiş confusion matrix.
    Sağ panel: çubuk grafik olarak metrikler + sayısal tablo.

    Parameters
    ----------
    y_true : array-like
        Gerçek etiketler (0 veya 1).
    y_pred : array-like
        Tahmin edilen etiketler (0 veya 1).
    model_name : str, optional
        Model adı (grafik başlığında gösterilir). Varsayılan: ``"Model"``.
    dataset_name : str, optional
        Veri seti adı. Varsayılan: ``"Dataset"``.
    class_names : list of str or None, optional
        Sınıf etiketleri. ``None`` ise ``["Normal (0)", "Anomali (1)"]`` kullanılır.
    output_path : str or None, optional
        Grafik dosya yolu (PNG/PDF). ``None`` ise ekranda gösterir.
    figsize : tuple, optional
        Şekil boyutu. Varsayılan: ``(12, 5)``.

    Returns
    -------
    dict
        Hesaplanan metrikler: ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``tp``, ``tn``, ``fp``, ``fn``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # --- Metrik hesaplama ---
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

    if class_names is None:
        class_names = ["Normal (0)", "Anomali (1)"]

    # Confusion matrix dizisi
    cm = np.array([[tn, fp], [fn, tp]])

    # --- Şekil düzeni ---
    fig = plt.figure(figsize=figsize, facecolor="#f8f9fa")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], figure=fig, wspace=0.35)

    # ---- Sol: Confusion Matrix ----
    ax_cm = fig.add_subplot(gs[0])
    ax_cm.set_facecolor("#f8f9fa")

    # Özel renk haritası: doğru tahminler yeşil, yanlışlar kırmızı tona
    cm_colors = np.array([
        ["#27ae60", "#e74c3c"],   # TN yeşil, FP kırmızı
        ["#e74c3c", "#2ecc71"],   # FN kırmızı, TP koyu yeşil
    ])
    cm_alpha = np.array([
        [0.55, 0.65],
        [0.65, 0.85],
    ])

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    cell_values = [[tn, fp], [fn, tp]]

    for i in range(2):
        for j in range(2):
            rect = mpatches.FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=cm_colors[i][j],
                alpha=cm_alpha[i][j],
                edgecolor="white",
                linewidth=2,
            )
            ax_cm.add_patch(rect)

            # Sayısal değer
            ax_cm.text(
                j, i,
                str(cell_values[i][j]),
                ha="center", va="center",
                fontsize=22, fontweight="bold", color="white",
            )
            # Etiket (TP/TN/FP/FN)
            ax_cm.text(
                j, i + 0.32,
                cell_labels[i][j],
                ha="center", va="center",
                fontsize=10, color="white", alpha=0.9,
            )

    ax_cm.set_xlim(-0.6, 1.6)
    ax_cm.set_ylim(-0.6, 1.6)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels([f"Tahmin:\n{class_names[0]}", f"Tahmin:\n{class_names[1]}"],
                          fontsize=9)
    ax_cm.set_yticklabels([f"Gerçek:\n{class_names[0]}", f"Gerçek:\n{class_names[1]}"],
                          fontsize=9)
    ax_cm.set_title("Confusion Matrix", fontsize=13, fontweight="bold",
                    color="#2c3e50", pad=10)
    ax_cm.tick_params(length=0)
    for spine in ax_cm.spines.values():
        spine.set_visible(False)

    # ---- Sağ: Metrik çubuk grafik + sayısal tablo ----
    ax_right = fig.add_subplot(gs[1])
    ax_right.set_facecolor("#f8f9fa")

    metric_names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_values = [accuracy, precision, recall, f1]
    bar_colors    = ["#3498db", "#9b59b6", "#e67e22", "#27ae60"]

    bars = ax_right.barh(
        metric_names, metric_values,
        color=bar_colors, alpha=0.82,
        edgecolor="white", linewidth=1.5,
        height=0.55,
    )

    # Değer etiketleri çubukların sağ ucuna
    for bar, val in zip(bars, metric_values):
        ax_right.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left",
            fontsize=11, fontweight="bold", color="#2c3e50",
        )

    ax_right.set_xlim(0, 1.18)
    ax_right.set_xlabel("Değer", fontsize=10)
    ax_right.set_title(
        f"Performans Metrikleri\n{model_name} — {dataset_name}",
        fontsize=12, fontweight="bold", color="#2c3e50", pad=10,
    )
    ax_right.axvline(1.0, color="#bdc3c7", linestyle="--", linewidth=1)
    ax_right.tick_params(axis="y", labelsize=10)
    ax_right.tick_params(axis="x", labelsize=9)
    for spine in ["top", "right"]:
        ax_right.spines[spine].set_visible(False)

    # Sayısal özet tablo — grafiğin altına
    table_data = [
        ["TP", str(tp), "TN", str(tn)],
        ["FP", str(fp), "FN", str(fn)],
        ["Toplam", str(total), "", ""],
    ]
    table = ax_right.table(
        cellText=table_data,
        colLabels=["", "#", "", "#"],
        cellLoc="center",
        loc="lower right",
        bbox=[0.55, -0.38, 0.44, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#bdc3c7")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if row % 2 == 0 else "white")

    fig.suptitle(
        f"{model_name} — {dataset_name} | Sınıflandırma Raporu",
        fontsize=14, fontweight="bold", color="#2c3e50", y=1.02,
    )

    plt.tight_layout()

    if output_path:
        _ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logging.info("Confusion matrix raporu kaydedildi: %s", output_path)
    else:
        plt.show()

    plt.close(fig)

    logging.info(
        "Confusion matrix raporu oluşturuldu — %s | %s | "
        "Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f",
        model_name, dataset_name, accuracy, precision, recall, f1,
    )

    return metrics
