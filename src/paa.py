import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_paa(data, num_segments):
    """
    Piecewise Aggregate Approximation (PAA) algoritmasını uygular.
    
    Zaman serisi verisini boyutunu küçültmek için parçalara böler ve ortalamasını alır.
    Orijinal zaman adımı sayısı (time_steps), hedef parça sayısına (num_segments)
    tam bölünemiyorsa ağırlıklı ortalama mantığını vektörize olarak uygular.

    Parameters
    ----------
    data : np.ndarray
        Girdi verisi. Genellikle (örnek_sayısı, zaman_adımı, özellik_sayısı) 
        veya (zaman_adımı, özellik_sayısı) boyutunda olabilir.
        Zaman adımı boyutu (sondan bir önceki boyut) `num_segments` boyutuna indirgenir.
    num_segments : int
        İndirgenecek hedef zaman adımı sayısı (parça sayısı).

    Returns
    -------
    np.ndarray
        PAA uygulanmış veri. Zaman adımı boyutu `num_segments` olarak değişir.
    """
    if num_segments <= 0:
        raise ValueError(f"num_segments 0'dan büyük olmalıdır, verilen: {num_segments}")

    shape = data.shape
    if len(shape) < 1:
        raise ValueError("Veri boş veya geçersiz boyutta.")

    if len(shape) == 1:
        time_steps = shape[0]
        axis = 0
    elif len(shape) == 2:
        time_steps = shape[1]
        axis = 1
    else:
        time_steps = shape[1]
        axis = 1

    if num_segments > time_steps:
        raise ValueError(f"num_segments ({num_segments}) orijinal zaman adımı sayısından ({time_steps}) büyük olamaz.")

    if num_segments == time_steps:
        logging.info("num_segments, zaman adımı sayısına eşit. Orijinal veri döndürülüyor.")
        return data.copy()

    # Eğer tam bölünebiliyorsa, basit reshape ve mean işlemi çok daha hızlıdır.
    if time_steps % num_segments == 0:
        factor = time_steps // num_segments
        if len(shape) == 1:
            paa_data = data.reshape(num_segments, factor).mean(axis=1)
        elif len(shape) == 2:
            paa_data = data.reshape(shape[0], num_segments, factor).mean(axis=2)
        else:
            paa_data = data.reshape(shape[0], num_segments, factor, shape[2]).mean(axis=2)
        logging.info(f"PAA başarıyla uygulandı (tam bölünebilir). Yeni boyut: {paa_data.shape}")
        return paa_data

    # Tam bölünemeyen durumlar için gerçek ağırlıklı ortalama (vektörize yöntem):
    # Verideki her zaman adımını num_segments kadar tekrarlıyoruz,
    # daha sonra reshape işlemiyle her segmente düşecek (time_steps) eleman ayırıp ortalamasını alıyoruz.
    if len(shape) == 1:
        expanded = np.repeat(data, num_segments, axis=0)
        paa_data = expanded.reshape(num_segments, time_steps).mean(axis=1)
    elif len(shape) == 2:
        expanded = np.repeat(data, num_segments, axis=1)
        paa_data = expanded.reshape(shape[0], num_segments, time_steps).mean(axis=2)
    else:
        expanded = np.repeat(data, num_segments, axis=1)
        paa_data = expanded.reshape(shape[0], num_segments, time_steps, shape[2]).mean(axis=2)

    logging.info(f"PAA başarıyla uygulandı (ağırlıklı bölme). Yeni boyut: {paa_data.shape}")
    return paa_data
