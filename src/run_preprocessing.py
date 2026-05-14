"""
Veri ön işleme pipeline'ının baştan sona (end-to-end) entegrasyon scripti.

Bu script; veri yükleme, bölme, normalizasyon, PCA ve sliding window adımlarını
sırayla çalıştırarak işlenmiş veriyi diske kaydeder. Tüm parametreler config.yaml'dan okunur.

Çalıştırmak için proje kökünde:
    python src/run_preprocessing.py
"""

import os
import sys
import logging
import yaml
import numpy as np

# Proje kökünü Python yoluna ekle (src klasöründen çalıştırılabilmesi için)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data_loader import (
    load_config,
    load_skab_data,
    load_batadal_data,
    split_skab_group_kfold,
    split_batadal_chronological,
)
from preprocessing import (
    preprocess_pipeline,
    add_gaussian_noise,
    create_sliding_windows,
    save_processed_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Yardımcı sabitler
# ---------------------------------------------------------------------------

# SKAB'da label ve metadata sütunları (feature değil)
SKAB_NON_FEATURE_COLS = {"anomaly", "changepoint", "source_file", "source_group"}

# BATADAL'da label sütunu
BATADAL_LABEL_COL = "ATT_FLAG"


def _get_feature_cols(df, non_feature_cols: set) -> list:
    """
    DataFrame'deki sayısal sütunlardan metadata/label sütunlarını çıkararak
    feature listesini döndürür.

    Parameters
    ----------
    df : pd.DataFrame
    non_feature_cols : set
        Feature olmayan sütun isimleri (label, metadata vb.)

    Returns
    -------
    list
        Kullanılabilir özellik sütunlarının listesi.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in non_feature_cols]
    return feature_cols


# ---------------------------------------------------------------------------
# SKAB Pipeline
# ---------------------------------------------------------------------------

def run_skab_pipeline(config: dict, save_base_dir: str) -> None:
    """
    SKAB veri seti için uçtan uca ön işleme pipeline'ını çalıştırır.

    Adımlar:
      1. Veriyi yükle.
      2. GroupKFold ile fold'lara böl (her fold'da train/test var, val yok).
      3. Her fold için: normalize et → PCA uygula → sliding window oluştur → kaydet.
      4. Gürültülü (noisy) senaryo için Gaussian noise eklenmiş versiyon da kaydedilir.

    Parameters
    ----------
    config : dict
        config.yaml içeriği.
    save_base_dir : str
        İşlenmiş verilerin kaydedileceği kök klasör yolu.
    """
    logging.info("=" * 60)
    logging.info("SKAB pipeline başlatıldı.")
    logging.info("=" * 60)

    skab_df = load_skab_data(config)
    if skab_df.empty:
        logging.error("SKAB verisi yüklenemedi. Pipeline durduruluyor.")
        return

    folds = split_skab_group_kfold(skab_df, config)
    feature_cols = _get_feature_cols(skab_df, SKAB_NON_FEATURE_COLS)
    logging.info(f"SKAB feature sütunları ({len(feature_cols)} adet): {feature_cols}")

    window_size = config["automata_params"]["window_size_fixed"]
    noise_std   = config["preprocessing"]["gaussian_noise_std"]
    label_col   = "anomaly" if "anomaly" in skab_df.columns else None

    for fold_info in folds:
        fold_num  = fold_info["fold"]
        train_raw = fold_info["train"]
        test_raw  = fold_info["test"]

        logging.info(f"--- SKAB Fold {fold_num} işleniyor ---")

        # 1. Normalize + PCA (val yok, SKAB için sadece train/test)
        final_train, _, final_test, scaler, pca = preprocess_pipeline(
            train_raw, val_df=None, test_df=test_raw,
            config=config, feature_cols=feature_cols,
        )

        # 2. Ham (işlenmiş) veri setlerini kaydet
        fold_save_dir = os.path.join(save_base_dir, "skab", f"fold_{fold_num}")
        save_processed_datasets(
            fold_save_dir,
            train_df=final_train,
            test_df=final_test,
            prefix="",
        )

        # 3. Sliding window (numpy dizileri olarak kaydet)
        pc_col = "PC1"  # PCA sonrası tek bileşen
        for split_name, split_df in [("train", final_train), ("test", final_test)]:
            if split_df is None or split_df.empty:
                continue
            X, y = create_sliding_windows(
                split_df,
                window_size=window_size,
                feature_cols=[pc_col],
                label_col=label_col,
            )
            np.save(os.path.join(fold_save_dir, f"{split_name}_X.npy"), X)
            np.save(os.path.join(fold_save_dir, f"{split_name}_y.npy"), y)
            logging.info(
                f"  Fold {fold_num} {split_name}: X={X.shape}, y={y.shape} → kayıt edildi."
            )

        # 4. Gürültülü senaryo: sadece train'e noise ekle, test aynı kalır
        noisy_train_raw = add_gaussian_noise(
            train_raw, noise_std=noise_std, feature_cols=feature_cols, random_seed=42
        )
        noisy_final_train, _, _, _, _ = preprocess_pipeline(
            noisy_train_raw, val_df=None, test_df=test_raw,
            config=config, feature_cols=feature_cols,
        )
        noisy_save_dir = os.path.join(save_base_dir, "skab_noisy", f"fold_{fold_num}")
        save_processed_datasets(
            noisy_save_dir,
            train_df=noisy_final_train,
            test_df=final_test,
            prefix="noisy_",
        )
        logging.info(f"  Fold {fold_num} gürültülü train kaydedildi: {noisy_save_dir}")

    logging.info("SKAB pipeline tamamlandı.")


# ---------------------------------------------------------------------------
# BATADAL Pipeline
# ---------------------------------------------------------------------------

def run_batadal_pipeline(config: dict, save_base_dir: str) -> None:
    """
    BATADAL veri seti için uçtan uca ön işleme pipeline'ını çalıştırır.

    Adımlar:
      1. Veriyi yükle.
      2. Kronolojik olarak %60/%20/%20 oranında train/val/test'e böl.
      3. Normalize et → PCA uygula → sliding window oluştur → kaydet.
      4. Gürültülü senaryo için Gaussian noise eklenmiş versiyon da kaydedilir.

    Parameters
    ----------
    config : dict
        config.yaml içeriği.
    save_base_dir : str
        İşlenmiş verilerin kaydedileceği kök klasör yolu.
    """
    logging.info("=" * 60)
    logging.info("BATADAL pipeline başlatıldı.")
    logging.info("=" * 60)

    batadal_df = load_batadal_data(config)
    if batadal_df.empty:
        logging.error("BATADAL verisi yüklenemedi. Pipeline durduruluyor.")
        return

    splits     = split_batadal_chronological(batadal_df, config)
    train_raw  = splits["train"]
    val_raw    = splits["val"]
    test_raw   = splits["test"]

    non_feature = {BATADAL_LABEL_COL}
    feature_cols = _get_feature_cols(batadal_df, non_feature)
    logging.info(f"BATADAL feature sütunları ({len(feature_cols)} adet): {feature_cols}")

    window_size = config["automata_params"]["window_size_fixed"]
    noise_std   = config["preprocessing"]["gaussian_noise_std"]
    label_col   = BATADAL_LABEL_COL if BATADAL_LABEL_COL in batadal_df.columns else None

    # 1. Normalize + PCA
    final_train, final_val, final_test, scaler, pca = preprocess_pipeline(
        train_raw, val_df=val_raw, test_df=test_raw,
        config=config, feature_cols=feature_cols,
    )

    # 2. Ham (işlenmiş) veri setlerini kaydet
    batadal_save_dir = os.path.join(save_base_dir, "batadal")
    save_processed_datasets(
        batadal_save_dir,
        train_df=final_train,
        val_df=final_val,
        test_df=final_test,
        prefix="",
    )

    # 3. Sliding window
    pc_col = "PC1"
    split_map = {
        "train": final_train,
        "val":   final_val,
        "test":  final_test,
    }
    for split_name, split_df in split_map.items():
        if split_df is None or split_df.empty:
            continue
        X, y = create_sliding_windows(
            split_df,
            window_size=window_size,
            feature_cols=[pc_col],
            label_col=label_col,
        )
        np.save(os.path.join(batadal_save_dir, f"{split_name}_X.npy"), X)
        np.save(os.path.join(batadal_save_dir, f"{split_name}_y.npy"), y)
        logging.info(f"  BATADAL {split_name}: X={X.shape}, y={y.shape} → kayıt edildi.")

    # 4. Gürültülü senaryo
    noisy_train_raw = add_gaussian_noise(
        train_raw, noise_std=noise_std, feature_cols=feature_cols, random_seed=42
    )
    noisy_final_train, _, _, _, _ = preprocess_pipeline(
        noisy_train_raw, val_df=val_raw, test_df=test_raw,
        config=config, feature_cols=feature_cols,
    )
    noisy_save_dir = os.path.join(save_base_dir, "batadal_noisy")
    save_processed_datasets(
        noisy_save_dir,
        train_df=noisy_final_train,
        val_df=final_val,
        test_df=final_test,
        prefix="noisy_",
    )
    logging.info(f"BATADAL gürültülü train kaydedildi: {noisy_save_dir}")
    logging.info("BATADAL pipeline tamamlandı.")


# ---------------------------------------------------------------------------
# Ana Giriş Noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Config dosyasının bulunması için çalışma dizinini proje kökü olarak ayarla
    project_root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(project_root)

    config = load_config("config.yaml")

    processed_data_dir = os.path.join("data", "processed")
    logging.info(f"İşlenmiş veriler şuraya kaydedilecek: {os.path.abspath(processed_data_dir)}")

    run_skab_pipeline(config, save_base_dir=processed_data_dir)
    run_batadal_pipeline(config, save_base_dir=processed_data_dir)

    logging.info("=" * 60)
    logging.info("Tüm ön işleme pipeline'ları başarıyla tamamlandı.")
    logging.info("=" * 60)
