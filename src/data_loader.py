"""
data_loader.py: SKAB ve BATADAL veri setlerini yükleyip GroupKFold veya kronolojik olarak böler.
"""

import os
import pandas as pd
import yaml
import logging
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Config dosyası bulunamadı: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Config dosyası hatalı formatta: {e}")
        raise

def load_skab_data(config):
    skab_path = config['data']['skab_path']
    valves = ['valve1', 'valve2']
    data_list = []

    logging.info(f"SKAB verileri {skab_path} klasöründen yükleniyor...")

    for valve in valves:
        valve_dir = os.path.join(skab_path, valve)
        if not os.path.exists(valve_dir):
            logging.warning(f"Klasör bulunamadı, atlanıyor: {valve_dir}")
            continue

        for filename in sorted(os.listdir(valve_dir)):
            if filename.endswith(".csv"):
                filepath = os.path.join(valve_dir, filename)
                try:
                    df = pd.read_csv(filepath, sep=';', index_col='datetime', parse_dates=True)
                    df['source_group'] = valve
                    df['source_file'] = f"{valve}_{filename}"
                    data_list.append(df)
                    logging.info(f"{filepath} başarıyla okundu. Boyut: {df.shape}")
                except Exception as e:
                    logging.error(f"{filepath} okunurken hata oluştu: {e}")
                
    if not data_list:
        logging.error("Hiçbir SKAB verisi yüklenemedi!")
        return pd.DataFrame()
        
    combined_df = pd.concat(data_list)
    logging.info(f"Tüm SKAB verileri birleştirildi. Toplam boyut: {combined_df.shape}")
    return combined_df


def load_batadal_data(config):
    batadal_path = config['data']['batadal_path']
    logging.info(f"BATADAL verisi {batadal_path} dosyasından yükleniyor...")

    if not os.path.exists(batadal_path):
        logging.error(f"BATADAL veri dosyası bulunamadı: {batadal_path}. Önce 'download_data.py' çalıştırın.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(batadal_path, sep=',', skipinitialspace=True)
        df.columns = df.columns.str.strip()

        if 'DATETIME' in df.columns:
            df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%y %H')
            df = df.set_index('DATETIME')
        else:
            logging.warning(f"'DATETIME' sütunu bulunamadı. Mevcut sütunlar: {df.columns.tolist()}")

        if 'ATT_FLAG' in df.columns:
            df['ATT_FLAG'] = df['ATT_FLAG'].replace(-999, 0)
            logging.info("  [BATADAL] ATT_FLAG etiketlerindeki -999 değerleri 0 olarak eşlendi.")

        logging.info(f"BATADAL Training Dataset 2 başarıyla yüklendi. Boyut: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"BATADAL verisi okunurken hata oluştu: {e}")
        return pd.DataFrame()


def split_skab_group_kfold(df, config):
    if df.empty:
        raise ValueError("Boş DataFrame GroupKFold ile bölünemez.")
    if 'source_file' not in df.columns:
        raise ValueError("DataFrame'de 'source_file' sütunu bulunamadı.")

    n_splits = config.get('preprocessing', {}).get('n_splits', 5)
    logging.info(
        f"SKAB verisi GroupKFold ile bölünüyor. "
        f"n_splits={n_splits}, toplam satır={len(df)}, "
        f"benzersiz dosya sayısı={df['source_file'].nunique()}"
    )

    groups = df['source_file']
    gkf = GroupKFold(n_splits=n_splits)
    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]
        folds.append({'fold': fold_idx, 'train': train_df, 'test': test_df})
        logging.info(
            f"  Fold {fold_idx}: train={len(train_df)} satır "
            f"({train_df['source_file'].nunique()} dosya), "
            f"test={len(test_df)} satır "
            f"({test_df['source_file'].nunique()} dosya)"
        )

    return folds


def split_batadal_chronological(df, config):
    if df.empty:
        raise ValueError("Boş DataFrame bölünemez.")

    ratios = (
        config.get('preprocessing', {})
              .get('batadal_split_ratios', [0.6, 0.2, 0.2])
    )
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Oranların toplamı 1.0 olmalıdır; mevcut: {sum(ratios)}")

    n = len(df)
    train_end = int(n * ratios[0])
    val_end   = train_end + int(n * ratios[1])

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    logging.info(
        f"BATADAL kronolojik bölme: toplam={n} satır, "
        f"train={len(train_df)} (%{ratios[0]*100:.0f}), "
        f"val={len(val_df)} (%{ratios[1]*100:.0f}), "
        f"test={len(test_df)} (%{ratios[2]*100:.0f})"
    )

    return {'train': train_df, 'val': val_df, 'test': test_df}


if __name__ == "__main__":
    logging.info("Veri yükleme testleri başlatılıyor...")
    try:
        config = load_config()
        skab_df = load_skab_data(config)
        batadal_df = load_batadal_data(config)
        if not batadal_df.empty and 'ATT_FLAG' in batadal_df.columns:
            saldiri_sayisi = (batadal_df['ATT_FLAG'] == -1).sum()
            logging.info(f"BATADAL saldırı etiketli satır sayısı: {saldiri_sayisi}")
    except Exception as e:
        logging.error(f"Test sırasında beklenmeyen bir hata oluştu: {e}")
