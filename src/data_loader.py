import os
import pandas as pd
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_skab_data_raw(config):
    """Sadece SKAB valve1 ve valve2 klasörlerindeki verileri okur (birleştirmez)."""
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
                data_list.append(df)
                print(f"{filepath} başarıyla okundu.")
                
    return data_list

if __name__ == "__main__":
    config = load_config()
    raw_data = load_skab_data_raw(config)
    print(f"Toplam {len(raw_data)} adet dosya okundu.")
