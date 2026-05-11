"""
Veri setlerinin ön işleme (preprocessing) adımlarını barındıran modül.
Özellikle veri sızıntısını (data leakage) engellemek adına scaler sadece eğitim
verisine fit edilir ve diğer setlere uygulanır.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_data(train_df, val_df=None, test_df=None, cols_to_scale=None):
    """
    Train verisi üzerinden StandardScaler fit eder ve verilen tüm veri setlerine uygular.
    Bu işlem, test setine dair hiçbir bilginin eğitim aşamasına sızmasını engeller.

    Parameters
    ----------
    train_df : pd.DataFrame
        Eğitim verisi (scaler bu veri üzerinden fit edilecektir).
    val_df : pd.DataFrame, optional
        Validasyon verisi.
    test_df : pd.DataFrame, optional
        Test verisi.
    cols_to_scale : list of str, optional
        Ölçeklendirilecek sütunların isimleri. Belirtilmezse tüm sayısal sütunlar ölçeklenir.
        Etiket veya benzeri dışta tutulması gereken sütunlar bu listeye konmamalıdır.

    Returns
    -------
    tuple
        (scaled_train_df, scaled_val_df, scaled_test_df, scaler)
        Eğer val_df veya test_df None olarak verildiyse, dönüşte ilgili yerlerde None bulunur.
    """
    if train_df is None or train_df.empty:
        raise ValueError("Eğitim verisi (train_df) boş veya None olamaz.")

    # Sayısal sütunları belirle
    if cols_to_scale is None:
        cols_to_scale = train_df.select_dtypes(include=['number']).columns.tolist()
    
    if not cols_to_scale:
        logging.warning("Ölçeklendirilecek hiçbir sayısal sütun bulunamadı.")
        return train_df, val_df, test_df, None

    scaler = StandardScaler()
    
    scaled_train = train_df.copy()
    scaled_train[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    logging.info(f"Scaler eğitim verisine fit edildi ve uygulandı. Sütun sayısı: {len(cols_to_scale)}")

    scaled_val = None
    if val_df is not None and not val_df.empty:
        scaled_val = val_df.copy()
        # Sadece transform uygulanır
        scaled_val[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
        logging.info("Scaler validasyon verisine uygulandı.")

    scaled_test = None
    if test_df is not None and not test_df.empty:
        scaled_test = test_df.copy()
        # Sadece transform uygulanır
        scaled_test[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
        logging.info("Scaler test verisine uygulandı.")

    return scaled_train, scaled_val, scaled_test, scaler
