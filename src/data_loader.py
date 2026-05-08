import os
import pandas as pd
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_skab_data(config):
    """SKAB valve1 ve valve2 klasörlerindeki verileri okur, etiketler ve birleştirir."""
    skab_path = config['data']['skab_path']
    valves = ['valve1', 'valve2']
    data_list = []

    for valve in valves:
        valve_dir = os.path.join(skab_path, valve)
        if not os.path.exists(valve_dir):
            continue

        for filename in sorted(os.listdir(valve_dir)):
            if filename.endswith(".csv"):
                filepath = os.path.join(valve_dir, filename)
                df = pd.read_csv(filepath, sep=';', index_col='datetime', parse_dates=True)
                
                # Commit 5: source_group ve source_file etiketlerini ekleme
                df['source_group'] = valve
                df['source_file'] = f"{valve}_{filename}"
                
                data_list.append(df)
                print(f"{filepath} başarıyla okundu ve etiketlendi.")
                
    if not data_list:
        return pd.DataFrame()
        
    # Commit 5: Verileri tek bir DataFrame'de birleştirme (concat)
    combined_df = pd.concat(data_list)
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

    if not os.path.exists(batadal_path):
        print(f"BATADAL veri dosyası bulunamadı: {batadal_path}")
        print("Lütfen önce 'python download_data.py' komutunu çalıştırın.")
        return pd.DataFrame()

    # CSV okuma: bazı sütun adlarında baş/son boşluk olabilir
    df = pd.read_csv(batadal_path, sep=',', skipinitialspace=True)

    # Sütun adlarındaki olası boşlukları temizle
    df.columns = df.columns.str.strip()

    # Tarih sütünunu datetime'a çevir ve index yap
    if 'DATETIME' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%y %H')
        df = df.set_index('DATETIME')
    else:
        print("Uyarı: 'DATETIME' sütunu bulunamadı. Mevcut sütunlar:", df.columns.tolist())

    print(f"BATADAL Training Dataset 2 başarıyla yüklendi. Boyut: {df.shape}")
    return df


if __name__ == "__main__":
    config = load_config()

    skab_df = load_skab_data(config)
    if not skab_df.empty:
        print(f"SKAB verileri birleştirildi. Toplam boyut: {skab_df.shape}")

    batadal_df = load_batadal_data(config)
    if not batadal_df.empty:
        print(f"BATADAL satır sayısı     : {batadal_df.shape[0]}")
        print(f"BATADAL sütun sayısı     : {batadal_df.shape[1]}")
        print(f"BATADAL tarih aralığı    : {batadal_df.index.min()} -> {batadal_df.index.max()}")
        if 'ATT_FLAG' in batadal_df.columns:
            saldiri_sayisi = (batadal_df['ATT_FLAG'] == -1).sum()
            print(f"Saldırı etiketli satır   : {saldiri_sayisi}")
