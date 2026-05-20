"""
Olasılıksal Otomata (Probabilistic Automata) modeli için gerekli bileşenleri
barındıran modül.

Bu modül, SAX ile sembolleştirilmiş zaman serisi verilerinden pattern sözlüğü
çıkarmak, durum (state) listesini belirlemek, geçiş sayılarını saymak,
bu sayıları frekans tabanlı geçiş olasılıklarına (Transition Probabilities)
dönüştürmek ve görülmemiş (unseen) örüntüleri Levenshtein mesafesi ile en yakın
bilinen duruma eşleyerek sistemin akışını oradan sürdürmek için kullanılır.
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


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    İki sembol dizisi (string) arasındaki Levenshtein (Edit) mesafesini hesaplar.
    Ekleme (insertion), silme (deletion) ve yer değiştirme (substitution)
    işlemlerinin minimum maliyetini bulur.

    Parameters
    ----------
    s1 : str
        Birinci sembol dizisi.
    s2 : str
        İkinci sembol dizisi.

    Returns
    -------
    int
        İki dizi arasındaki minimum düzenleme mesafesi.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class ProbabilisticAutomata:
    """
    Zaman serisi anomali tespiti için Olasılıksal Otomata (Probabilistic Automata) modeli.

    Eğitim verisi üzerinden SAX sembollerini alarak durumları, sözlüğü ve
    geçiş olasılıklarını (transition probabilities) öğrenir. Test aşamasında,
    görülmemiş (unseen) bir örüntü gelirse, Levenshtein mesafesi kullanarak
    en yakın duruma atama yapar ve sistem oradan devam eder.

    Temel Metotlar
    --------------
    fit                  : Modeli eğitim verisiyle eğitir.
    handle_unseen_state  : Levenshtein ile en yakın bilinen durumu döndürür.
    resolve_state        : Herhangi bir durumu (seen ya da unseen) bilinen bir
                           duruma çözer.
    resolve_and_advance  : Mevcut durumdan bir adım ilerler; unseen ise önce
                           çözer, ardından geçiş satırını döndürür.
    map_sequence_to_states: Bir SAX dizisini baştan sona otomata üzerinde
                           yürüterek çözülmüş durum zincirini ve geçiş
                           olasılıklarını döndürür.
    compute_path_probability: Verilen bir dizinin (path) toplam geçiş olasılığını hesaplar.
    get_state_id         : Durum adından tamsayı ID'si döndürür (unseen-safe).
    """

    def __init__(self):
        self.sax_dictionary: list[str] = []
        self.states: list[str] = []
        self.state_to_id: dict[str, int] = {}
        self.id_to_state: dict[int, str] = {}
        self.transition_probabilities: list[list[float]] = []
        self.num_states: int = 0
        self.is_fitted: bool = False

    def fit(self, train_sax_strings: list[str]):
        """
        SADECE eğitim (train) verisi kullanılarak modeli eğitir.
        Sözlük, durumlar ve olasılık matrisi hesaplanır.

        Parameters
        ----------
        train_sax_strings : list of str
            Eğitim verisinden elde edilmiş sıralı SAX dizileri.
        """
        if not train_sax_strings:
            raise ValueError("Eğitim verisi (train_sax_strings) boş olamaz.")

        self.sax_dictionary = extract_sax_dictionary(train_sax_strings)
        
        state_info = extract_state_list(train_sax_strings)
        self.states = state_info["states"]
        self.state_to_id = state_info["state_to_id"]
        self.id_to_state = state_info["id_to_state"]
        self.num_states = state_info["num_states"]

        transition_counts = count_state_transitions(train_sax_strings, self.state_to_id)
        self.transition_probabilities = compute_transition_probabilities(transition_counts)
        
        self.is_fitted = True
        logging.info("ProbabilisticAutomata modeli başarıyla eğitildi.")

    def handle_unseen_state(self, unseen_state: str) -> str:
        """
        Sistem daha önce karşılaşmadığı (unseen) bir SAX örüntüsüyle
        karşılaştığında çalışır. Levenshtein mesafesi kullanarak 
        sözlükteki en yakın duruma eşleştirir.

        Kural (Unseen Kuralı):
            Eğitimde olmayan örüntüler Levenshtein ile en yakın duruma
            atanır ve sistem oradan devam eder.

        Parameters
        ----------
        unseen_state : str
            Sözlükte bulunmayan yeni SAX örüntüsü.

        Returns
        -------
        str
            Sözlükte (dictionary) bulunan ve Levenshtein mesafesi
            en düşük olan durum.
        """
        if not self.is_fitted:
            raise RuntimeError("Model eğitilmedi. Lütfen önce fit() çağırın.")

        if not self.sax_dictionary:
            raise RuntimeError("SAX sözlüğü boş.")

        closest_state = min(
            self.sax_dictionary,
            key=lambda state: levenshtein_distance(unseen_state, state)
        )
        
        logging.warning(
            f"Unseen durum tespit edildi: '{unseen_state}'. "
            f"Levenshtein mesafesi ile en yakın duruma atandı: '{closest_state}'"
        )
        return closest_state

    def get_state_id(self, state: str) -> int:
        """
        Bir SAX durumunun tam sayı (integer) kimliğini döndürür.
        Eğer durum daha önce görülmemiş (unseen) ise, önce handle_unseen_state
        kullanılarak en yakın duruma eşleştirilir.

        Parameters
        ----------
        state : str
            Kimliği istenen SAX durumu.

        Returns
        -------
        int
            Durumun tam sayı kimliği (ID).
        """
        if not self.is_fitted:
            raise RuntimeError("Model eğitilmedi. Lütfen önce fit() çağırın.")

        if state not in self.state_to_id:
            matched_state = self.handle_unseen_state(state)
            return self.state_to_id[matched_state]
        
        return self.state_to_id[state]

    # ------------------------------------------------------------------
    # Unseen eşleştirme ve devam mantığı
    # ------------------------------------------------------------------

    def resolve_state(self, state: str) -> str:
        """
        Herhangi bir SAX örüntüsünü sözlükte bilinen bir duruma çözer.

        Durum zaten sözlükte mevcutsa doğrudan döndürür. Görülmemiş
        (unseen) bir örüntü ise ``handle_unseen_state`` ile Levenshtein
        mesafesi en düşük duruma eşleştirilir.

        Parameters
        ----------
        state : str
            Çözülecek SAX örüntüsü (seen veya unseen olabilir).

        Returns
        -------
        str
            Sözlükte kesinlikle var olan bir durum adı.
        """
        if not self.is_fitted:
            raise RuntimeError("Model eğitilmedi. Lütfen önce fit() çağırın.")

        if state in self.state_to_id:
            return state

        return self.handle_unseen_state(state)

    def resolve_and_advance(
        self,
        current_state: str,
        next_state: str,
    ) -> tuple[str, str, list[float]]:
        """
        Mevcut durumdan bir adım ilerler.

        Her iki durum da ``resolve_state`` ile çözülür (unseen ise
        Levenshtein ile en yakın bilinen duruma eşleştirilir). Ardından
        mevcut durumun geçiş olasılık satırı döndürülerek sistemin
        takip eden adımda bu bilgiyi kullanması sağlanır.

        Parameters
        ----------
        current_state : str
            Mevcut SAX durumu (seen veya unseen olabilir).
        next_state : str
            Bir sonraki SAX durumu (seen veya unseen olabilir).

        Returns
        -------
        tuple[str, str, list[float]]
            - resolved_current : Çözülmüş mevcut durum adı.
            - resolved_next    : Çözülmüş sonraki durum adı.
            - transition_row   : Çözülmüş mevcut durumun geçiş olasılık
                                 vektörü (uzunluk = num_states).

        Examples
        --------
        >>> automata = ProbabilisticAutomata()
        >>> automata.fit(["abc", "bcd", "abc", "cde", "bcd"])
        >>> resolved_cur, resolved_nxt, probs = automata.resolve_and_advance("xyz", "bcd")
        >>> resolved_cur in automata.states
        True
        >>> len(probs) == automata.num_states
        True
        """
        if not self.is_fitted:
            raise RuntimeError("Model eğitilmedi. Lütfen önce fit() çağırın.")

        resolved_current = self.resolve_state(current_state)
        resolved_next = self.resolve_state(next_state)

        current_id = self.state_to_id[resolved_current]
        transition_row = self.transition_probabilities[current_id]

        logging.debug(
            f"resolve_and_advance: '{current_state}' → '{resolved_current}' "
            f"| sonraki: '{next_state}' → '{resolved_next}'"
        )
        return resolved_current, resolved_next, transition_row

    def map_sequence_to_states(
        self,
        sax_sequence: list[str],
    ) -> dict:
        """
        Bir SAX sembol dizisini baştan sona otomata üzerinde yürütür.

        Her eleman ``resolve_state`` ile bilinen bir duruma çözülür;
        ardışık çiftler için ``resolve_and_advance`` çağrılarak geçiş
        olasılık vektörleri toplanır. Bu sayede görülmemiş (unseen)
        örüntüler içeren bir test dizisi bile kesintisiz işlenebilir.

        Parameters
        ----------
        sax_sequence : list of str
            İşlenecek SAX örüntü dizisi. En az 1 eleman içermelidir.

        Returns
        -------
        dict
            Aşağıdaki anahtarları içeren sözlük:

            - ``resolved_states`` (list of str): Her elemanın çözülmüş
              durum adını içeren liste (uzunluk = len(sax_sequence)).
            - ``transition_rows`` (list of list[float]): Ardışık her
              (i, i+1) çifti için mevcut durumun geçiş olasılık vektörü.
              Uzunluğu len(sax_sequence) - 1 olup boş dizi için [].
            - ``unseen_count`` (int): Orijinal dizide kaç adet unseen
              örüntü bulunduğu.

        Raises
        ------
        ValueError
            ``sax_sequence`` None veya boş ise.
        RuntimeError
            Model henüz eğitilmemişse.

        Examples
        --------
        >>> automata = ProbabilisticAutomata()
        >>> automata.fit(["abc", "bcd", "abc", "cde", "bcd"])
        >>> result = automata.map_sequence_to_states(["abc", "xyz", "bcd"])
        >>> len(result["resolved_states"])
        3
        >>> result["unseen_count"]
        1
        """
        if not self.is_fitted:
            raise RuntimeError("Model eğitilmedi. Lütfen önce fit() çağırın.")

        if not sax_sequence:
            raise ValueError("sax_sequence boş veya None olamaz.")

        resolved_states: list[str] = []
        transition_rows: list[list[float]] = []
        unseen_count: int = 0

        # İlk elemanı çöz
        for raw_state in sax_sequence:
            if raw_state not in self.state_to_id:
                unseen_count += 1
            resolved_states.append(self.resolve_state(raw_state))

        # Ardışık çiftler için geçiş vektörlerini topla
        for i in range(len(resolved_states) - 1):
            current_id = self.state_to_id[resolved_states[i]]
            transition_rows.append(self.transition_probabilities[current_id])

        if unseen_count > 0:
            logging.info(
                f"map_sequence_to_states: {unseen_count} unseen örüntü "
                "Levenshtein ile en yakın duruma eşleştirildi."
            )

        logging.info(
            f"map_sequence_to_states tamamlandı. "
            f"Dizi uzunluğu: {len(sax_sequence)}, "
            f"Unseen sayısı: {unseen_count}"
        )

        return {
            "resolved_states": resolved_states,
            "transition_rows": transition_rows,
            "unseen_count": unseen_count,
        }

    def compute_path_probability(self, sax_sequence: list[str]) -> float:
        """
        Verilen SAX dizisinin (path) otomata üzerindeki toplam olasılığını hesaplar.
        Ardışık durum geçiş olasılıklarının çarpımı (product of consecutive transition 
        probabilities) olarak tanımlanır.

        Parameters
        ----------
        sax_sequence : list of str
            Olasılığı hesaplanacak SAX örüntü dizisi. En az 2 eleman içermelidir.

        Returns
        -------
        float
            Yolun toplam geçiş olasılığı (Path Probability).
            Eğer dizi 2 elemandan kısaysa 1.0 döner.
            
        Raises
        ------
        RuntimeError
            Model henüz eğitilmemişse.
        """
        if not self.is_fitted:
            raise RuntimeError("Model eğitilmedi. Lütfen önce fit() çağırın.")

        if not sax_sequence or len(sax_sequence) < 2:
            return 1.0

        path_prob = 1.0
        
        mapped_data = self.map_sequence_to_states(sax_sequence)
        resolved_states = mapped_data["resolved_states"]

        for i in range(len(resolved_states) - 1):
            next_state = resolved_states[i+1]
            next_id = self.state_to_id[next_state]
            
            prob = mapped_data["transition_rows"][i][next_id]
            path_prob *= prob

        logging.info(
            f"Path probability hesaplandı. "
            f"Dizi uzunluğu: {len(sax_sequence)}, Olasılık: {path_prob:.4e}"
        )
        return path_prob
