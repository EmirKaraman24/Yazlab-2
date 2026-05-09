import os
import pandas as pd
import yaml
import logging

# Commit 7: Loglama (test) mekanizması eklendi
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
    """SKAB valve1 ve valve2 klasörlerindeki verileri okur, etiketler ve birleştirir."""
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
                    
                    # Commit 5: source_group ve source_file etiketlerini ekleme
                    df['source_group'] = valve
                    df['source_file'] = f"{valve}_{filename}"
                    
                    data_list.append(df)
                    logging.info(f"{filepath} başarıyla okundu. Boyut: {df.shape}")
                except Exception as e:
                    logging.error(f"{filepath} okunurken hata oluştu: {e}")
                
    if not data_list:
        logging.error("Hiçbir SKAB verisi yüklenemedi!")
        return pd.DataFrame()
        
    # Commit 5: Verileri tek bir DataFrame'de birleştirme (concat)
    combined_df = pd.concat(data_list)
    logging.info(f"Tüm SKAB verileri birleştirildi. Toplam boyut: {combined_df.shape}")
    return combined_df


def load_batadal_data(config):
    """
    BATADAL Training Dataset 2'yi (BATADAL_dataset04.csv) okur ve yükler.

    Veri kümesi; ~6 aylık saatlik SCADA kayıtlarını içerir.
    Bazı satırlar saldırı etiketiyle ("-1") işaretlenmiştir.

    Returns
    -------
    pd.DataFrame
        'DATETIME' kolonunu index olarak kullanan, temizlenmiş BATADAL verisi.
    """
    batadal_path = config['data']['batadal_path']

    logging.info(f"BATADAL verisi {batadal_path} dosyasından yükleniyor...")

    if not os.path.exists(batadal_path):
        logging.error(f"BATADAL veri dosyası bulunamadı: {batadal_path}. Önce 'download_data.py' çalıştırın.")
        return pd.DataFrame()

    try:
        # CSV okuma: bazı sütun adlarında baş/son boşluk olabilir
        df = pd.read_csv(batadal_path, sep=',', skipinitialspace=True)

        # Sütun adlarındaki olası boşlukları temizle
        df.columns = df.columns.str.strip()

        # Tarih sütünunu datetime'a çevir ve index yap
        if 'DATETIME' in df.columns:
            df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%y %H')
            df = df.set_index('DATETIME')
        else:
            logging.warning(f"'DATETIME' sütunu bulunamadı. Mevcut sütunlar: {df.columns.tolist()}")

        logging.info(f"BATADAL Training Dataset 2 başarıyla yüklendi. Boyut: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"BATADAL verisi okunurken hata oluştu: {e}")
        return pd.DataFrame()


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
