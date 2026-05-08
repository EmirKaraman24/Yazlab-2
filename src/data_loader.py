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

if __name__ == "__main__":
    config = load_config()
    skab_df = load_skab_data(config)
    print(f"SKAB verileri birleştirildi. Toplam boyut: {skab_df.shape}")
