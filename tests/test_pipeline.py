"""
Birim Testleri: Ön İşleme Pipeline'ı (preprocess_pipeline)

Bu test modülü, `preprocessing.py` içindeki `preprocess_pipeline` fonksiyonunun
veri sızıntısı (data leakage) olmadan çalıştığını doğrular.

Test edilen kurallar:
  1. Scaler (StandardScaler) yalnızca train verisine fit edilmelidir.
  2. PCA yalnızca train verisine fit edilmelidir.
  3. Val ve test setleri transform'a tabii tutulmalıdır (fit'e değil).
  4. Pipeline çıktılarının boyutları ve şekilleri beklenen değerlerde olmalıdır.
  5. Config'den PCA bileşen sayısı doğru okunmalıdır.
  6. Train ortalaması 0, standart sapması ≈ 1 olmalıdır (scaler fit edildi kanıtı).
  7. Val/test ortalaması train'e göre farklı olabilir (yani onlara ayrı fit yapılmamıştır).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
import os

# Proje kökünü Python yoluna ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import preprocess_pipeline, normalize_data, apply_pca


# ---------------------------------------------------------------------------
# Yardımcı sabitler ve fixture'lar
# ---------------------------------------------------------------------------

FEATURE_COLS = ['sensor_1', 'sensor_2', 'sensor_3']
LABEL_COL = 'anomaly'

MOCK_CONFIG = {
    'preprocessing': {
        'pca_n_components': 1
    }
}


def _make_df(n_rows: int, seed: int = 0, offset: float = 0.0) -> pd.DataFrame:
    """
    Belirtilen boyutta rastgele sayısal bir DataFrame oluşturur.

    Parameters
    ----------
    n_rows : int
        Satır sayısı.
    seed : int
        Rastgelelik tohumu.
    offset : float
        Sensör değerlerine eklenecek sabit kaydırma miktarı (farklı dağılımları simüle eder).

    Returns
    -------
    pd.DataFrame
        `sensor_1`, `sensor_2`, `sensor_3` ve `anomaly` sütunlarına sahip DataFrame.
    """
    rng = np.random.default_rng(seed)
    data = {
        'sensor_1': rng.normal(loc=10.0 + offset, scale=2.0, size=n_rows),
        'sensor_2': rng.normal(loc=5.0 + offset, scale=1.5, size=n_rows),
        'sensor_3': rng.normal(loc=20.0 + offset, scale=5.0, size=n_rows),
        LABEL_COL:  rng.integers(0, 2, size=n_rows),
    }
    return pd.DataFrame(data)


@pytest.fixture
def datasets():
    """Train (100 satır), val (30 satır) ve test (30 satır) veri setlerini döndürür."""
    train = _make_df(100, seed=42, offset=0.0)
    val   = _make_df(30,  seed=7,  offset=50.0)   # Kasıtlı farklı dağılım
    test  = _make_df(30,  seed=99, offset=100.0)  # Kasıtlı farklı dağılım
    return train, val, test


# ---------------------------------------------------------------------------
# Test 1: Pipeline temel çalışma kontrolü
# ---------------------------------------------------------------------------

class TestPipelineBasic:
    """Pipeline'ın temel işlevselliğini doğrulayan testler."""

    def test_pipeline_returns_five_tuple(self, datasets):
        """Pipeline 5 elemanlı bir tuple döndürmelidir: (train, val, test, scaler, pca)."""
        train, val, test = datasets
        result = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        assert isinstance(result, tuple), "Pipeline bir tuple döndürmelidir."
        assert len(result) == 5, "Pipeline tam olarak 5 eleman içermelidir."

    def test_pipeline_output_types(self, datasets):
        """Pipeline çıktıları beklenen türlerde olmalıdır."""
        train, val, test = datasets
        final_train, final_val, final_test, scaler, pca = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        assert isinstance(final_train, pd.DataFrame), "Train çıktısı DataFrame olmalıdır."
        assert isinstance(final_val, pd.DataFrame),   "Val çıktısı DataFrame olmalıdır."
        assert isinstance(final_test, pd.DataFrame),  "Test çıktısı DataFrame olmalıdır."
        assert isinstance(scaler, StandardScaler),    "Scaler StandardScaler olmalıdır."
        assert isinstance(pca, PCA),                  "PCA sklearn PCA nesnesi olmalıdır."

    def test_pipeline_raises_without_config(self, datasets):
        """Config verilmezse ValueError fırlatılmalıdır."""
        train, val, test = datasets
        with pytest.raises(ValueError, match="config parametresi"):
            preprocess_pipeline(train, val, test, config=None, feature_cols=FEATURE_COLS)


# ---------------------------------------------------------------------------
# Test 2: Veri Sızıntısı Yokluğu — Scaler
# ---------------------------------------------------------------------------

class TestNoDataLeakageScaler:
    """Scaler'ın yalnızca train verisine fit edildiğini doğrulayan testler."""

    def test_scaler_fit_only_on_train(self, datasets):
        """
        Scaler'ın mean_ değerleri, train setinin ham ortalamasına yakın olmalıdır.
        Val/test setine fit edilseydi mean_ farklı olurdu (offset=50/100 farkı var).
        """
        train, val, test = datasets
        _, _, _, scaler, _ = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        # Scaler'ın öğrendiği ortalamalar train verisinin ortalamasına yakın olmalı
        train_means = train[FEATURE_COLS].mean().values
        np.testing.assert_allclose(
            scaler.mean_, train_means, atol=1e-6,
            err_msg="Scaler mean_ değerleri train verisi ortalamasına eşit olmalıdır."
        )

    def test_scaler_std_matches_train(self, datasets):
        """
        Scaler'ın öğrendiği standart sapmalar train verisinden hesaplanmış olmalıdır.
        """
        train, val, test = datasets
        _, _, _, scaler, _ = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        train_stds = train[FEATURE_COLS].std(ddof=0).values  # sklearn ddof=0 kullanır
        np.testing.assert_allclose(
            scaler.scale_, train_stds, atol=1e-6,
            err_msg="Scaler scale_ değerleri train verisi std'sine (ddof=0) eşit olmalıdır."
        )

    def test_scaled_train_mean_near_zero(self, datasets):
        """
        Scaler train'e fit edildiğinden, normalize_data sonrası train
        sütun ortalamalarının ≈ 0 olması gerekir (StandardScaler garantisi).
        """
        train, val, _ = datasets
        scaled_train, scaled_val, _, scaler = normalize_data(
            train, val, None, cols_to_scale=FEATURE_COLS
        )
        train_col_means = scaled_train[FEATURE_COLS].mean().values
        np.testing.assert_allclose(
            train_col_means, np.zeros(len(FEATURE_COLS)), atol=1e-9,
            err_msg="Train normalleştirme sonrası sütun ortalamaları 0'a yakın olmalıdır."
        )

    def test_val_mean_not_zero_due_to_offset(self, datasets):
        """
        Val verisi kasıtlı olarak farklı bir dağılımdan (offset=50) gelmektedir.
        Scaler yalnızca train'e fit edildiğinden val'ın normalize sonrası ortalaması
        0 değil, belirgin şekilde farklı bir değer olmalıdır.
        Bu, scaler'ın val'a ayrı fit YAPILMADIĞININ kanıtıdır.
        """
        train, val, _ = datasets
        _, scaled_val, _, _ = normalize_data(
            train, val, None, cols_to_scale=FEATURE_COLS
        )
        val_col_means = scaled_val[FEATURE_COLS].mean().values
        # Val offset=50, train std~2 → val normalize ortalama ~25 civarında olmalı (0 olamaz)
        assert np.any(np.abs(val_col_means) > 1.0), (
            "Val normalize sonrası ortalama 0'dan uzak olmalıdır; "
            "scaler train'e fit edildi, val'a değil."
        )

    def test_scaler_not_fit_on_test_directly(self, datasets):
        """
        Pipeline'ın scaler.fit veya scaler.fit_transform'u test verisiyle
        çağırmadığını spy ile doğrular.
        """
        train, val, test = datasets
        original_fit = StandardScaler.fit

        fit_calls_data = []

        def tracking_fit(self, X, y=None, sample_weight=None):
            fit_calls_data.append(len(X))
            return original_fit(self, X, y, sample_weight)

        with patch.object(StandardScaler, 'fit', tracking_fit):
            preprocess_pipeline(
                train, val, test,
                config=MOCK_CONFIG,
                feature_cols=FEATURE_COLS
            )

        # Scaler yalnızca bir kez fit edilmiş olmalı (train için)
        assert len(fit_calls_data) == 1, (
            f"StandardScaler.fit yalnızca 1 kez çağrılmalıdır, "
            f"{len(fit_calls_data)} kez çağrıldı."
        )
        # Ve bu çağrı train boyutunda olmalıdır
        assert fit_calls_data[0] == len(train), (
            "Scaler fit çağrısı train verisi boyutunda olmalıdır."
        )


# ---------------------------------------------------------------------------
# Test 3: Veri Sızıntısı Yokluğu — PCA
# ---------------------------------------------------------------------------

class TestNoDataLeakagePCA:
    """PCA'nın yalnızca train verisine fit edildiğini doğrulayan testler."""

    def test_pca_fit_only_on_train(self, datasets):
        """
        PCA'nın fit yalnızca train verisine yapılmalıdır.
        apply_pca fonksiyonuna spy eklenerek pca.fit_transform çağrısı izlenir.
        """
        train, val, test = datasets
        original_fit_transform = PCA.fit_transform

        fit_transform_sizes = []

        def tracking_fit_transform(self, X, y=None):
            fit_transform_sizes.append(len(X))
            return original_fit_transform(self, X, y)

        with patch.object(PCA, 'fit_transform', tracking_fit_transform):
            preprocess_pipeline(
                train, val, test,
                config=MOCK_CONFIG,
                feature_cols=FEATURE_COLS
            )

        # PCA.fit_transform yalnızca 1 kez çağrılmış olmalı (train için)
        assert len(fit_transform_sizes) == 1, (
            f"PCA.fit_transform yalnızca 1 kez çağrılmalıdır, "
            f"{len(fit_transform_sizes)} kez çağrıldı."
        )
        assert fit_transform_sizes[0] == len(train), (
            "PCA.fit_transform train boyutunda çağrılmalıdır."
        )

    def test_pca_transform_called_for_val_and_test(self, datasets):
        """
        PCA.transform, val ve test setleri için çağrılmış olmalıdır (fit değil).
        """
        train, val, test = datasets
        original_transform = PCA.transform

        transform_sizes = []

        def tracking_transform(self, X):
            transform_sizes.append(len(X))
            return original_transform(self, X)

        with patch.object(PCA, 'transform', tracking_transform):
            preprocess_pipeline(
                train, val, test,
                config=MOCK_CONFIG,
                feature_cols=FEATURE_COLS
            )

        # transform, val (30) ve test (30) için 2 kez çağrılmış olmalı
        assert len(transform_sizes) == 2, (
            f"PCA.transform val ve test için 2 kez çağrılmalıdır, "
            f"{len(transform_sizes)} kez çağrıldı."
        )

    def test_pca_components_from_config(self, datasets):
        """
        PCA bileşen sayısı config['preprocessing']['pca_n_components'] değerinden
        okunmalı; sabit kodlanmış (hard-coded) olmamalıdır.
        """
        train, val, test = datasets
        config_with_2_components = {'preprocessing': {'pca_n_components': 2}}

        _, _, _, _, pca = preprocess_pipeline(
            train, val, test,
            config=config_with_2_components,
            feature_cols=FEATURE_COLS
        )
        assert pca.n_components_ == 2, (
            "PCA bileşen sayısı config'den okunmalı; burada 2 beklenmektedir."
        )

    def test_pca_output_has_correct_columns(self, datasets):
        """
        Pipeline çıktısı, orijinal feature sütunları yerine PCA bileşen sütunlarını
        içermelidir. n_components=1 için 'PC1' sütunu bulunmalıdır.
        """
        train, val, test = datasets
        final_train, final_val, final_test, _, _ = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        for df_name, df in [('train', final_train), ('val', final_val), ('test', final_test)]:
            assert 'PC1' in df.columns, f"{df_name} DataFrame'inde 'PC1' sütunu bulunmalıdır."
            for col in FEATURE_COLS:
                assert col not in df.columns, (
                    f"{df_name} DataFrame'inde orijinal özellik sütunu '{col}' "
                    "kaldırılmış olmalıdır."
                )

    def test_pca_label_column_preserved(self, datasets):
        """
        Etiket (label) sütunu PCA sonrasında veri setlerinde korunmalıdır.
        """
        train, val, test = datasets
        final_train, final_val, final_test, _, _ = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        for df_name, df in [('train', final_train), ('val', final_val), ('test', final_test)]:
            assert LABEL_COL in df.columns, (
                f"{df_name} DataFrame'inde '{LABEL_COL}' sütunu korunmalıdır."
            )


# ---------------------------------------------------------------------------
# Test 4: Kenar Durumlar (Edge Cases)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Kenar durumları ve hata senaryoları."""

    def test_pipeline_without_val_and_test(self):
        """Val ve test olmadan pipeline çalışabilmelidir; sonuçlar None döner."""
        train = _make_df(50, seed=42)
        final_train, final_val, final_test, scaler, pca = preprocess_pipeline(
            train, val_df=None, test_df=None,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        assert final_train is not None,  "Train çıktısı None olmamalıdır."
        assert final_val is None,        "Val çıktısı None olmalıdır."
        assert final_test is None,       "Test çıktısı None olmalıdır."

    def test_pipeline_without_val_only(self):
        """Sadece test verisi verildiğinde pipeline çalışabilmelidir."""
        train = _make_df(50, seed=42)
        test  = _make_df(20, seed=99)
        final_train, final_val, final_test, scaler, pca = preprocess_pipeline(
            train, val_df=None, test_df=test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        assert final_train is not None,  "Train çıktısı None olmamalıdır."
        assert final_val is None,        "Val çıktısı None olmalıdır."
        assert final_test is not None,   "Test çıktısı None olmamalıdır."

    def test_pipeline_train_row_count_preserved(self, datasets):
        """Pipeline sonrasında train satır sayısı değişmemelidir."""
        train, val, test = datasets
        final_train, _, _, _, _ = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        assert len(final_train) == len(train), (
            "Train satır sayısı pipeline öncesi ve sonrası aynı olmalıdır."
        )

    def test_pipeline_val_row_count_preserved(self, datasets):
        """Pipeline sonrasında val satır sayısı değişmemelidir."""
        train, val, test = datasets
        _, final_val, _, _, _ = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        assert len(final_val) == len(val), (
            "Val satır sayısı pipeline öncesi ve sonrası aynı olmalıdır."
        )

    def test_pipeline_test_row_count_preserved(self, datasets):
        """Pipeline sonrasında test satır sayısı değişmemelidir."""
        train, val, test = datasets
        _, _, final_test, _, _ = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        assert len(final_test) == len(test), (
            "Test satır sayısı pipeline öncesi ve sonrası aynı olmalıdır."
        )

    def test_no_nan_in_outputs(self, datasets):
        """Pipeline çıktılarında NaN değer bulunmamalıdır."""
        train, val, test = datasets
        final_train, final_val, final_test, _, _ = preprocess_pipeline(
            train, val, test,
            config=MOCK_CONFIG,
            feature_cols=FEATURE_COLS
        )
        for df_name, df in [('train', final_train), ('val', final_val), ('test', final_test)]:
            assert not df.isnull().any().any(), (
                f"{df_name} DataFrame'inde NaN değerler bulunmamalıdır."
            )
