"""
Olasılıksal Otomata (Probabilistic Automata) modeli için gerekli bileşenleri
barındıran modül.

Bu modül, SAX ile sembolleştirilmiş zaman serisi verilerinden pattern sözlüğü
çıkarmak, durum (state) listesini belirlemek, geçiş sayılarını saymak ve
bu sayıları frekans tabanlı geçiş olasılıklarına (Transition Probabilities)
dönüştürmek için kullanılır.
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


def count_state_transitions(sax_strings: list[str], state_to_id: dict) -> list[list[int]]:
    """
    SAX sembol dizileri üzerinde ardışık durum (state) geçişlerinin sayılarını hesaplar.

    Kural (Data Leakage Yasak):
        Geçiş matrisi SADECE eğitim (train) verisi dizilimi korunarak hesaplanmalıdır.

    Parameters
    ----------
    sax_strings : list of str
        Train verisine uygulanan SAX dönüşümü sonucu elde edilen ardışık sembol dizileri.
    state_to_id : dict
        Durum adından tam sayı kimliğine eşleme. `extract_state_list` çıktısındaki 
        'state_to_id' kullanılmalıdır.

    Returns
    -------
    list of list of int
        Boyutu (num_states x num_states) olan 2 boyutlu geçiş sayım matrisi.
        (i, j) elemanı, i durumundan j durumuna kaç kez geçildiğini gösterir.
    """
    if not sax_strings or not state_to_id:
        return []

    num_states = len(state_to_id)
    transition_counts = [[0 for _ in range(num_states)] for _ in range(num_states)]

    for i in range(len(sax_strings) - 1):
        current_state = sax_strings[i]
        next_state = sax_strings[i+1]

        if current_state in state_to_id and next_state in state_to_id:
            current_id = state_to_id[current_state]
            next_id = state_to_id[next_state]
            transition_counts[current_id][next_id] += 1

    logging.info("Durumlar arası geçiş sayıları (transition counts) hesaplandı.")
    return transition_counts


def compute_transition_probabilities(
    transition_counts: list[list[int]],
) -> list[list[float]]:
    """
    Geçiş sayım matrisini frekans tabanlı geçiş olasılıklarına (Transition
    Probability Matrix) dönüştürür.

    Her satır için matematiksel dönüşüm şu şekildedir::

        P(i -> j) = count(i -> j) / sum_j(count(i -> j))

    Eğer bir satırın toplam geçiş sayısı sıfır ise (yani o durum eğitim
    verisinde hiçbir zaman kaynak durum olarak görülmemişse), o satıra düzgün
    (uniform) dağılım atanır; bu sayede bölme-sıfır hatası önlenir ve
    sonraki aşamalar için geçerli bir olasılık matrisi garanti edilir.

    Kural (Data Leakage Yasak):
        Bu fonksiyon yalnızca ``count_state_transitions`` ile üretilmiş,
        SADECE eğitim verisine ait sayım matrisi üzerinde çağrılmalıdır.

    Parameters
    ----------
    transition_counts : list of list of int
        ``count_state_transitions`` fonksiyonunun döndürdüğü
        (num_states x num_states) boyutunda geçiş sayım matrisi.

    Returns
    -------
    list of list of float
        (num_states x num_states) boyutunda geçiş olasılık matrisi.
        Her satırın toplamı 1.0'a eşittir.
        Boş girdi verilirse boş liste döner.

    Raises
    ------
    ValueError
        Matris kare (square) değilse.

    Examples
    --------
    >>> counts = [[2, 1, 0], [0, 3, 1], [0, 0, 0]]
    >>> probs = compute_transition_probabilities(counts)
    >>> probs[0]          # 2/(2+1+0) = 0.667, 1/3 = 0.333, 0.0
    [0.6666666666666666, 0.3333333333333333, 0.0]
    >>> probs[2]          # Sıfır satırı → düzgün (uniform) dağılım
    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    """
    if not transition_counts:
        logging.warning(
            "Boş geçiş sayım matrisi verildi. Boş olasılık matrisi döndürülüyor."
        )
        return []

    num_states = len(transition_counts)

    # Kare matris doğrulama
    for row_idx, row in enumerate(transition_counts):
        if len(row) != num_states:
            raise ValueError(
                f"Geçiş sayım matrisi kare (square) olmalıdır. "
                f"{row_idx}. satırın uzunluğu {len(row)}, "
                f"beklenen {num_states}."
            )

    uniform_probability = 1.0 / num_states  # sıfır satırları için
    transition_probabilities: list[list[float]] = []

    for row in transition_counts:
        row_total = sum(row)

        if row_total == 0:
            # Bu durum eğitimde hiç kaynak olmamış → uniform dağılım ata
            normalized_row = [uniform_probability] * num_states
            logging.debug(
                "Geçiş sayısı sıfır olan bir satır tespit edildi; "
                "düzgün (uniform) dağılım atandı."
            )
        else:
            # Frekans tabanlı normalleştirme: count / toplam
            normalized_row = [count / row_total for count in row]

        transition_probabilities.append(normalized_row)

    logging.info(
        "Geçiş olasılık matrisi (Transition Probability Matrix) hesaplandı. "
        f"Boyut: {num_states}x{num_states}"
    )
    return transition_probabilities
