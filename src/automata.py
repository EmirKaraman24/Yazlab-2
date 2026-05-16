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


def extract_state_list(sax_strings: list[str]) -> dict:
    """
    Sadece eğitim (train) verisi üzerinden elde edilmiş SAX sembol dizilerinden
    otomata için sıralı durum (state) listesini oluşturur.

    Her benzersiz SAX pattern bir durum olarak kabul edilir. Durumlar alfabetik
    olarak sıralanır ve her birine 0'dan başlayan benzersiz bir tam sayı kimliği
    (state_id) atanır.

    Kural (Data Leakage Yasak):
        Durum listesi YALNIZCA eğitim (train) verisinde gözlemlenen pattern'lerden
        oluşturulmalıdır. Validasyon veya test setindeki görülmemiş (unseen)
        pattern'ler bu listeye dahil edilmez.

    Parameters
    ----------
    sax_strings : list of str
        Train verisine uygulanan SAX dönüşümü sonucu elde edilen sembol dizileri.
        Örn: ["abc", "bcd", "abc", "cde"]

    Returns
    -------
    dict
        Aşağıdaki anahtarları içeren sözlük:

        - ``states`` (list of str): Alfabetik sıralı benzersiz durum adları
          (SAX pattern'leri). Örn: ["abc", "bcd", "cde"]
        - ``state_to_id`` (dict): Durum adından tam sayı kimliğine eşleme.
          Örn: {"abc": 0, "bcd": 1, "cde": 2}
        - ``id_to_state`` (dict): Tam sayı kimliğinden durum adına ters eşleme.
          Örn: {0: "abc", 1: "bcd", 2: "cde"}
        - ``num_states`` (int): Toplam durum sayısı. Örn: 3

    Raises
    ------
    ValueError
        ``sax_strings`` None ise.

    Examples
    --------
    >>> train_sax = ["abc", "bcd", "abc", "cde", "bcd"]
    >>> state_info = extract_state_list(train_sax)
    >>> state_info["states"]
    ['abc', 'bcd', 'cde']
    >>> state_info["state_to_id"]
    {'abc': 0, 'bcd': 1, 'cde': 2}
    >>> state_info["num_states"]
    3
    """
    if sax_strings is None:
        raise ValueError("sax_strings None olamaz.")

    if not sax_strings:
        logging.warning("Boş bir SAX dizisi listesi verildi. Durum listesi boş döndürülüyor.")
        return {
            "states": [],
            "state_to_id": {},
            "id_to_state": {},
            "num_states": 0,
        }

    # Benzersiz durumları bul, alfabetik sıraya diz
    states = sorted(set(sax_strings))

    # İki yönlü eşleme tabloları
    state_to_id = {state: idx for idx, state in enumerate(states)}
    id_to_state = {idx: state for state, idx in state_to_id.items()}

    logging.info(
        f"Durum listesi oluşturuldu. Toplam durum sayısı: {len(states)}"
    )

    return {
        "states": states,
        "state_to_id": state_to_id,
        "id_to_state": id_to_state,
        "num_states": len(states),
    }
