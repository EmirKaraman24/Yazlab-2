"""
Commit 8 (ARKADAS): Veri setlerinin ilk analizlerini (shape, missing values) basan keşif kodu.
"""

import logging
import pandas as pd

from data_loader import load_config, load_skab_data, load_batadal_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Verilen DataFrame'in temel istatistiklerini (boyut, eksik değer) loglar ve ekrana basar.

    Parameters
    ----------
    df : pd.DataFrame
        Analiz edilecek veri seti.
    dataset_name : str
        Raporda görünecek veri seti adı.
    """
    separator = "=" * 60

    print(f"\n{separator}")
    print(f"  VERİ SETİ ANALİZİ: {dataset_name}")
    print(separator)

    # --- Boyut (Shape) ---
    print(f"\n[BOYUT]")
    print(f"  Satır sayısı  : {df.shape[0]:,}")
    print(f"  Sütun sayısı  : {df.shape[1]:,}")

    # --- Sütun Tipleri ---
    print(f"\n[SÜTUN TİPLERİ]")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):<15}: {count} sütun")

    # --- Eksik Değerler (Missing Values) ---
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    print(f"\n[EKSİK DEĞERLER]")
    if missing_cols.empty:
        print("  Eksik deger bulunamadi. [OK]")
    else:
        total_missing = missing_cols.sum()
        total_cells = df.shape[0] * df.shape[1]
        print(f"  Eksik değer içeren sütun sayısı : {len(missing_cols)}")
        print(f"  Toplam eksik hücre sayısı       : {total_missing:,} / {total_cells:,} "
              f"({100 * total_missing / total_cells:.2f}%)")
        print(f"\n  {'Sütun':<35} {'Eksik Sayı':>10} {'Oran (%)':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*10}")
        for col, count in missing_cols.sort_values(ascending=False).items():
            oran = 100 * count / df.shape[0]
            print(f"  {col:<35} {count:>10,} {oran:>9.2f}%")

    # --- Index Bilgisi ---
    print(f"\n[INDEX]")
    print(f"  Tip   : {type(df.index).__name__}")
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"  Başlangıç : {df.index.min()}")
        print(f"  Bitiş     : {df.index.max()}")
        print(f"  Süre      : {df.index.max() - df.index.min()}")

    print(f"\n{separator}\n")

    logging.info(f"{dataset_name} analizi tamamlandı. "
                 f"Shape={df.shape}, Eksik sütun sayısı={len(missing_cols)}")


def run_exploration(config_path: str = "config.yaml") -> None:
    """
    Tüm veri setlerini yükler ve keşif analizini çalıştırır.

    Parameters
    ----------
    config_path : str
        config.yaml dosya yolu.
    """
    logging.info("Keşif analizi başlatılıyor...")

    config = load_config(config_path)

    # --- SKAB ---
    skab_df = load_skab_data(config)
    if not skab_df.empty:
        analyze_dataset(skab_df, "SKAB (Birleşik)")

        # SKAB: dosya bazında özet
        print("  [SKAB – Dosya Bazında Satır Sayıları]")
        if 'source_file' in skab_df.columns:
            file_counts = skab_df['source_file'].value_counts().sort_index()
            for fname, cnt in file_counts.items():
                print(f"    {fname:<45} {cnt:>6,} satır")
        print()
    else:
        logging.warning("SKAB verisi boş, analiz atlandı.")

    # --- BATADAL ---
    batadal_df = load_batadal_data(config)
    if not batadal_df.empty:
        analyze_dataset(batadal_df, "BATADAL Training Dataset 2")

        # BATADAL: saldırı etiket dağılımı
        if 'ATT_FLAG' in batadal_df.columns:
            print("  [BATADAL – Etiket Dağılımı (ATT_FLAG)]")
            label_counts = batadal_df['ATT_FLAG'].value_counts().sort_index()
            for label, cnt in label_counts.items():
                oran = 100 * cnt / len(batadal_df)
                aciklama = "Normal" if label == 1 else "Saldırı"
                print(f"    {aciklama} (ATT_FLAG={label:>2}): {cnt:>6,} satır ({oran:.2f}%)")
            print()
    else:
        logging.warning("BATADAL verisi boş, analiz atlandı.")

    logging.info("Keşif analizi tamamlandı.")


if __name__ == "__main__":
    run_exploration()
