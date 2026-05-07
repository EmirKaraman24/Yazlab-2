import os
import urllib.request
import zipfile
import shutil

DATA_DIR = "data"

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download successful.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def setup_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Download SKAB
    skab_zip = os.path.join(DATA_DIR, "SKAB.zip")
    skab_url = "https://github.com/waico/SKAB/archive/refs/heads/master.zip"
    if not os.path.exists(os.path.join(DATA_DIR, "SKAB-master")):
        download_file(skab_url, skab_zip)
        print("Extracting SKAB...")
        with zipfile.ZipFile(skab_zip, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(skab_zip)
    else:
        print("SKAB already exists.")

    # Move valve1 and valve2
    for folder in ["valve1", "valve2"]:
        src = os.path.join(DATA_DIR, "SKAB-master", "data", folder)
        dest = os.path.join(DATA_DIR, folder)
        if os.path.exists(src) and not os.path.exists(dest):
            shutil.move(src, dest)
            print(f"Moved {folder} to {DATA_DIR}/{folder}")

    # 2. Download BATADAL (Training Dataset 2)
    batadal_dir = os.path.join(DATA_DIR, "BATADAL")
    if not os.path.exists(batadal_dir):
        os.makedirs(batadal_dir)
    
    batadal_url = "https://raw.githubusercontent.com/sateesh-kumar-b/BATADAL/master/BATADAL_dataset04.csv"
    batadal_csv = os.path.join(batadal_dir, "BATADAL_dataset04.csv")
    if not os.path.exists(batadal_csv):
        download_file(batadal_url, batadal_csv)
    else:
        print("BATADAL already exists.")

if __name__ == "__main__":
    setup_data()
