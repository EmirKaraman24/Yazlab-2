"""
Veri setlerinin ön işleme (preprocessing) adımlarını barındıran modül.
Özellikle veri sızıntısını (data leakage) engellemek adına scaler ve PCA yalnızca
eğitim verisine fit edilir; validasyon ve test setlerine sadece transform uygulanır.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


def apply_pca(
    train_df,
    val_df=None,
    test_df=None,
    cols_to_reduce=None,
    n_components: int = 1,
    output_col_prefix: str = "PC",
):
    """
    Train verisi üzerinden PCA fit eder ve verilen tüm veri setlerine uygular.

    Kural: PCA **sadece** eğitim verisine fit edilir; validasyon ve test setlerine
    yalnızca ``transform`` uygulanır.  Bu sayede veri sızıntısı (data leakage)
    tamamen önlenir.

    Dönüştürülen bileşenler, orijinal özellik sütunlarının yerine eklenir.
    Bileşen sütunları ``{output_col_prefix}1``, ``{output_col_prefix}2`` … şeklinde isimlendirilir.

    Parameters
    ----------
    train_df : pd.DataFrame
        Eğitim verisi (PCA bu veri üzerinden fit edilecektir).
    val_df : pd.DataFrame, optional
        Validasyon verisi.
    test_df : pd.DataFrame, optional
        Test verisi.
    cols_to_reduce : list of str, optional
        PCA'ya sokulacak sütunların isimleri. Belirtilmezse tüm sayısal sütunlar
        kullanılır.  Etiket, kaynak dosya adı gibi metaveri sütunları hariç tutulmalıdır.
    n_components : int, optional
        Hedef boyut (bileşen sayısı). Varsayılan değer 1'dir (PC1).
        Hard-coded kullanım yasak olduğundan bu değer ``config.yaml`` üzerinden
        okunarak ilgili çağrıya parametre olarak geçirilmelidir.
    output_col_prefix : str, optional
        Oluşturulan bileşen sütunlarının ön eki. Varsayılan: ``"PC"``.

    Returns
    -------
    tuple
        ``(pca_train_df, pca_val_df, pca_test_df, pca)``

        * Her DataFrame; orijinal *cols_to_reduce* sütunları kaldırılmış,
          yerine PCA bileşen sütunları eklenmiş halini içerir.
        * Metaveri sütunları (etiket vb.) olduğu gibi korunur.
        * ``val_df`` veya ``test_df`` None olarak geçildiyse dönüşte de None döner.
        * ``pca``: Eğitim verisine fit edilmiş ``sklearn.decomposition.PCA`` nesnesi.

    Raises
    ------
    ValueError
        ``train_df`` boş veya None ise ya da ``n_components`` geçersizse.

    Examples
    --------
    >>> import yaml
    >>> with open("config.yaml") as f:
    ...     cfg = yaml.safe_load(f)
    >>> n_comp = cfg["preprocessing"]["pca_n_components"]
    >>> pca_train, pca_val, pca_test, fitted_pca = apply_pca(
    ...     train_df, val_df, test_df,
    ...     cols_to_reduce=feature_cols,
    ...     n_components=n_comp,
    ... )
    """
    if train_df is None or train_df.empty:
        raise ValueError("Eğitim verisi (train_df) boş veya None olamaz.")

    if n_components < 1:
        raise ValueError(f"n_components en az 1 olmalıdır, verilen: {n_components}")

    # PCA'ya girecek sütunları belirle
    if cols_to_reduce is None:
        cols_to_reduce = train_df.select_dtypes(include=['number']).columns.tolist()

    if not cols_to_reduce:
        logging.warning("PCA için hiçbir sayısal sütun bulunamadı; veri setleri değiştirilmeden döndürülüyor.")
        return train_df, val_df, test_df, None

    if n_components > len(cols_to_reduce):
        raise ValueError(
            f"n_components ({n_components}) sütun sayısından ({len(cols_to_reduce)}) büyük olamaz."
        )

    bileşen_isimleri = [f"{output_col_prefix}{i + 1}" for i in range(n_components)]

    pca = PCA(n_components=n_components)

    # --- Eğitim seti: fit + transform ---
    pca_egitim_degerleri = pca.fit_transform(train_df[cols_to_reduce])
    pca_train_df = _pca_sonuclarini_birlestir(train_df, pca_egitim_degerleri, cols_to_reduce, bileşen_isimleri)
    logging.info(
        f"PCA eğitim verisine fit edildi ve uygulandı. "
        f"Girdi boyutu: {len(cols_to_reduce)}, çıktı boyutu: {n_components}. "
        f"Açıklanan toplam varyans oranı: {pca.explained_variance_ratio_.sum():.4f}"
    )

    # --- Validasyon seti: sadece transform ---
    pca_val_df = None
    if val_df is not None and not val_df.empty:
        pca_val_degerleri = pca.transform(val_df[cols_to_reduce])
        pca_val_df = _pca_sonuclarini_birlestir(val_df, pca_val_degerleri, cols_to_reduce, bileşen_isimleri)
        logging.info("PCA validasyon verisine uygulandı.")

    # --- Test seti: sadece transform ---
    pca_test_df = None
    if test_df is not None and not test_df.empty:
        pca_test_degerleri = pca.transform(test_df[cols_to_reduce])
        pca_test_df = _pca_sonuclarini_birlestir(test_df, pca_test_degerleri, cols_to_reduce, bileşen_isimleri)
        logging.info("PCA test verisine uygulandı.")

    return pca_train_df, pca_val_df, pca_test_df, pca


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _pca_sonuclarini_birlestir(
    kaynak_df: pd.DataFrame,
    pca_degerleri: np.ndarray,
    orijinal_sutunlar: list,
    bileşen_isimleri: list,
) -> pd.DataFrame:
    """
    PCA dönüşüm sonuçlarını orijinal DataFrame'e entegre eder.

    Orijinal özellik sütunları kaldırılır; yerlerine PCA bileşen sütunları eklenir.
    Etiket, kaynak dosya adı gibi metaveri sütunları değiştirilmeden korunur.

    Parameters
    ----------
    kaynak_df : pd.DataFrame
        Orijinal veri çerçevesi.
    pca_degerleri : np.ndarray
        ``PCA.fit_transform`` veya ``PCA.transform`` çıktısı.
    orijinal_sutunlar : list of str
        PCA'ya sokulan sütun isimleri (kaldırılacak).
    bileşen_isimleri : list of str
        Yeni bileşen sütun isimleri (eklenecek).

    Returns
    -------
    pd.DataFrame
        Metaveri sütunları + PCA bileşen sütunlarından oluşan yeni DataFrame.
    """
    bileşen_df = pd.DataFrame(pca_degerleri, columns=bileşen_isimleri, index=kaynak_df.index)
    metaveri_df = kaynak_df.drop(columns=orijinal_sutunlar)
    return pd.concat([metaveri_df, bileşen_df], axis=1)
