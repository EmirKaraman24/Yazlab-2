"""
SAX (Symbolic Aggregate approXimation) algoritmasını barındıran modül.

SAX, PAA ile boyutu küçültülmüş zaman serilerini sembolik dizilere (string)
dönüştürür. Gaussian dağılımı eşit olasılıklı bölgelere (breakpoint) ayrılarak
her PAA katsayısına bir harf atanır.

Kural (Data Leakage Yasak):
    Breakpoint'ler (kesim noktaları) YALNIZCA eğitim verisi üzerinden hesaplanır;
    validasyon ve test setlerine yalnızca transform uygulanır.
"""

import numpy as np
import logging
from scipy.stats import norm

from paa import apply_paa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Yardımcı: Breakpoint hesaplama
# ---------------------------------------------------------------------------

def compute_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    Gaussian dağılımını eşit olasılıklı bölgelere bölen breakpoint'leri hesaplar.

    Standart normal dağılımı (mu=0, sigma=1) ``alphabet_size`` eşit alana bölen
    ``alphabet_size - 1`` adet kesim noktası döndürülür.

    Parameters
    ----------
    alphabet_size : int
        Alfabe büyüklüğü (sembol sayısı). Minimum 2 olmalıdır.

    Returns
    -------
    np.ndarray
        ``alphabet_size - 1`` elemanlı breakpoint dizisi (artan sıralı).

    Raises
    ------
    ValueError
        ``alphabet_size`` 2'den küçükse.

    Examples
    --------
    >>> compute_breakpoints(3)
    array([-0.4307, 0.4307])
    """
    if alphabet_size < 2:
        raise ValueError(
            f"alphabet_size en az 2 olmalıdır, verilen: {alphabet_size}"
        )
    # Eşit olasılıklı dilimlerin sınır değerleri (ekstraları hariç: -inf ve +inf)
    kesim_olasılıkları = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    return norm.ppf(kesim_olasılıkları)


# ---------------------------------------------------------------------------
# Yardımcı: Tek bir PAA dizisini sembol dizisine çevirme
# ---------------------------------------------------------------------------

def _paa_to_symbols(paa_row: np.ndarray, breakpoints: np.ndarray) -> str:
    """
    Tek bir PAA katsayı dizisini breakpoint'lere göre sembol dizisine çevirir.

    Harfler küçük İngilizce alfabesinden seçilir: a, b, c, …

    Parameters
    ----------
    paa_row : np.ndarray
        1-D PAA katsayı dizisi.
    breakpoints : np.ndarray
        ``compute_breakpoints`` ile elde edilen kesim noktaları.

    Returns
    -------
    str
        PAA katsayılarının sembolik gösterimi (örn. ``"abc"``).
    """
    # np.searchsorted ile her katsayının hangi aralığa düştüğünü bul (indeks = harf sırası)
    harf_indeksleri = np.searchsorted(breakpoints, paa_row, side='right')
    # 'a' = 97 ASCII kodu
    semboller = ''.join(chr(97 + idx) for idx in harf_indeksleri)
    return semboller


# ---------------------------------------------------------------------------
# Ana SAX sınıfı
# ---------------------------------------------------------------------------

class SAXTransformer:
    """
    SAX (Symbolic Aggregate approXimation) dönüşüm sınıfı.

    Kullanım akışı:
        1. ``fit(train_data)``  → Breakpoint'leri eğitim verisi üzerinden hesapla.
        2. ``transform(data)``  → Herhangi bir veri setine dönüşüm uygula.

    Kural: ``fit`` yalnızca eğitim verisinde çağrılmalıdır. Validasyon ve test
    verisi için doğrudan ``transform`` kullanılmalıdır.

    Parameters
    ----------
    num_segments : int
        PAA parça sayısı (``config.yaml`` → ``automata_params.window_size_fixed``).
    alphabet_size : int
        Alfabe büyüklüğü (``config.yaml`` → ``automata_params.alphabet_size_fixed``).

    Attributes
    ----------
    breakpoints_ : np.ndarray or None
        ``fit`` sonrasında hesaplanan breakpoint değerleri.
    is_fitted_ : bool
        Modelin eğitilip eğitilmediğini gösteren bayrak.

    Examples
    --------
    >>> import yaml
    >>> with open("config.yaml") as f:
    ...     cfg = yaml.safe_load(f)
    >>> w = cfg["automata_params"]["window_size_fixed"]
    >>> a = cfg["automata_params"]["alphabet_size_fixed"]
    >>> sax = SAXTransformer(num_segments=w, alphabet_size=a)
    >>> sax.fit(train_pca_array)
    >>> train_symbols = sax.transform(train_pca_array)
    >>> test_symbols  = sax.transform(test_pca_array)
    """

    def __init__(self, num_segments: int, alphabet_size: int):
        if num_segments < 1:
            raise ValueError(
                f"num_segments en az 1 olmalıdır, verilen: {num_segments}"
            )
        if alphabet_size < 2:
            raise ValueError(
                f"alphabet_size en az 2 olmalıdır, verilen: {alphabet_size}"
            )

        self.num_segments = num_segments
        self.alphabet_size = alphabet_size
        self.breakpoints_: np.ndarray | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_data: np.ndarray) -> "SAXTransformer":
        """
        Breakpoint'leri YALNIZCA eğitim verisi üzerinden hesaplar.

        SAX standardında breakpoint'ler Gaussian dağılımından teorik olarak
        türetildiğinden, eğitim verisinin dağılımı burada kullanılmaz; ancak
        yalnızca ``fit`` metodunun eğitim verisinde çağrıldığından emin olmak
        için bu adım zorunludur.

        Parameters
        ----------
        train_data : np.ndarray
            Eğitim verisi.  ``(örnek_sayısı,)`` veya
            ``(örnek_sayısı, özellik_sayısı)`` gibi herhangi bir boyutta
            olabilir; breakpoint hesabı boyuttan bağımsızdır.

        Returns
        -------
        SAXTransformer
            ``self`` (zincirleme çağrı desteği).

        Raises
        ------
        ValueError
            ``train_data`` boş veya None ise.
        """
        if train_data is None or (hasattr(train_data, '__len__') and len(train_data) == 0):
            raise ValueError("Eğitim verisi (train_data) boş veya None olamaz.")

        self.breakpoints_ = compute_breakpoints(self.alphabet_size)
        self.is_fitted_ = True

        logging.info(
            f"SAX fit tamamlandı. num_segments={self.num_segments}, "
            f"alphabet_size={self.alphabet_size}, "
            f"breakpoints={np.round(self.breakpoints_, 4).tolist()}"
        )
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, data: np.ndarray) -> list[str]:
        """
        Verilen veriyi önce PAA ile özetler, ardından SAX sembollerine dönüştürür.

        ``fit`` çağrılmadan kullanılırsa ``RuntimeError`` fırlatır.

        Parameters
        ----------
        data : np.ndarray
            Dönüştürülecek veri.
            - 1-D ``(zaman_adımı,)``      → tek örnek, tek özellik
            - 2-D ``(örnek, zaman_adımı)``→ çok sayıda tek-özellikli örnek
            - 3-D ``(örnek, zaman_adımı, özellik)`` → pencerelenmiş çok özellikli veri;
              bu durumda her özellik bağımsız olarak SAX'lanır ve semboller
              yan yana yazılır.

        Returns
        -------
        list of str
            Her örnek için bir SAX sembol dizisi.

        Raises
        ------
        RuntimeError
            ``fit`` çağrılmadan ``transform`` kullanılırsa.
        ValueError
            ``data`` boş veya None ise.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "SAXTransformer.transform() çağrılmadan önce fit() çağrılmalıdır."
            )

        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            raise ValueError("Dönüştürülecek veri boş veya None olamaz.")

        data = np.array(data)

        # --- Boyuta göre işleme ---
        if data.ndim == 1:
            # Tek bir zaman serisi örneği
            paa_result = apply_paa(data, self.num_segments)
            sax_dizileri = [_paa_to_symbols(paa_result, self.breakpoints_)]

        elif data.ndim == 2:
            # (örnek_sayısı, zaman_adımı) → her satır bir örnek
            paa_result = apply_paa(data, self.num_segments)
            sax_dizileri = [
                _paa_to_symbols(paa_result[i], self.breakpoints_)
                for i in range(paa_result.shape[0])
            ]

        elif data.ndim == 3:
            # (örnek_sayısı, zaman_adımı, özellik_sayısı)
            # Her özellik ayrı SAX'lanır; sonuçlar birleştirilir.
            num_samples, _, num_features = data.shape
            sax_dizileri = [""] * num_samples
            for ozellik_idx in range(num_features):
                ozellik_dilimi = data[:, :, ozellik_idx]           # (örnek, zaman)
                paa_ozellik = apply_paa(ozellik_dilimi, self.num_segments)
                for ornek_idx in range(num_samples):
                    sax_dizileri[ornek_idx] += _paa_to_symbols(
                        paa_ozellik[ornek_idx], self.breakpoints_
                    )
        else:
            raise ValueError(
                f"Desteklenmeyen veri boyutu: {data.ndim}. 1-D, 2-D veya 3-D veri bekleniyor."
            )

        logging.info(
            f"SAX transform tamamlandı. Örnek sayısı: {len(sax_dizileri)}, "
            f"Sembol uzunluğu: {len(sax_dizileri[0]) if sax_dizileri else 0}"
        )
        return sax_dizileri

    # ------------------------------------------------------------------
    # fit_transform (kısayol - SADECE eğitim verisi için)
    # ------------------------------------------------------------------

    def fit_transform(self, train_data: np.ndarray) -> list[str]:
        """
        ``fit`` ve ``transform`` işlemlerini eğitim verisi üzerinde ardışık uygular.

        .. warning::
            Bu metot YALNIZCA eğitim (train) verisi için kullanılmalıdır.
            Validasyon/test setleri için ``transform`` metodunu kullanın.

        Parameters
        ----------
        train_data : np.ndarray
            Eğitim verisi.

        Returns
        -------
        list of str
            Eğitim verisinin SAX sembol dizileri.
        """
        return self.fit(train_data).transform(train_data)
