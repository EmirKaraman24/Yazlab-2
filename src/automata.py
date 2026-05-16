"""
Olasılıksal Otomata (Probabilistic Automata) modeli için gerekli bileşenleri
barındıran modül.

Bu modül, SAX ile sembolleştirilmiş zaman serisi verilerinden pattern sözlüğü
çıkarmak ve devamında durum (state) geçişlerini hesaplamak için kullanılacaktır.
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_sax_dictionary(sax_strings: list[str]) -> list[str]:
    """
    Sadece eğitim (train) verisi üzerinden üretilmiş SAX sembol dizilerinden
    benzersiz olanları bularak SAX sözlüğünü (dictionary/pattern listesi) oluşturur.

    Kural (Data Leakage Yasak):
        Sözlük (ve dolayısıyla otomatadaki olası durumlar) YALNIZCA eğitim verisinde
        görülen örüntüler (pattern) üzerinden belirlenmelidir.

    Parameters
    ----------
    sax_strings : list of str
        Train verisine uygulanan SAX dönüşümü sonucu elde edilen sembol dizileri.
        Örn: ["abc", "bcd", "abc", "cde"]

    Returns
    -------
    list of str
        Benzersiz SAX pattern'lerini (alfabetik sıralı) içeren sözlük.
        Örn: ["abc", "bcd", "cde"]
    """
    if not sax_strings:
        logging.warning("Boş bir SAX dizisi listesi verildi. Sözlük boş döndürülüyor.")
        return []

    # Benzersiz pattern'leri bul ve alfabetik olarak sırala
    unique_patterns = sorted(list(set(sax_strings)))
    
    logging.info(
        f"SAX sözlüğü çıkarıldı. Toplam benzersiz pattern sayısı: {len(unique_patterns)}"
    )
    return unique_patterns
