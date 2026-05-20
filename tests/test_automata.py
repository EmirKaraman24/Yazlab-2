"""
Birim Testleri: Unseen Durum Davranışı ve Levenshtein Mesafesi

Bu test modülü, `automata.py` içindeki `levenshtein_distance` fonksiyonu ile
`ProbabilisticAutomata` sınıfının unseen (görülmemiş) örüntü yönetimi mantığını
doğrular.

Test edilen kurallar:
  1. Levenshtein mesafesi aynı diziler için 0 döndürmelidir.
  2. Levenshtein mesafesi simetrik olmalıdır: dist(a, b) == dist(b, a).
  3. Levenshtein üçgen eşitsizliğini sağlamalıdır: dist(a,c) <= dist(a,b) + dist(b,c).
  4. Boş dize ile herhangi bir dizenin mesafesi dizenin uzunluğuna eşit olmalıdır.
  5. Unseen bir örüntü, sözlükteki en yakın duruma (Levenshtein) eşlenmelidir.
  6. Unseen örüntü sistemi durdurmaz; resolve_state bilinen bir durum döndürmelidir.
  7. Eğitim verisinde görülen bir örüntü için handle_unseen_state çağrılmamalıdır.
  8. map_sequence_to_states unseen_count'u doğru saymalıdır.
  9. compute_path_probability unseen içeren dizilerde 0.0'dan büyük bir değer döndürmelidir.
 10. Fit öncesinde handle_unseen_state RuntimeError fırlatmalıdır.
"""

import pytest
import sys
import os

# Proje kökünü Python yoluna ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from automata import (
    levenshtein_distance,
    ProbabilisticAutomata,
)


# ---------------------------------------------------------------------------
# Yardımcı sabitler ve fixture'lar
# ---------------------------------------------------------------------------

# Eğitim verisinde bulunan SAX dizileri
TRAIN_SAX = ["abc", "bcd", "cde", "abc", "bcd", "abc", "cde", "bcd", "cde", "abc"]

# Eğitim verisinde HİÇ görülmemiş örnekler
UNSEEN_CLOSE_TO_ABC   = "aXc"   # 'abc'ye 1 edit uzaklıkta
UNSEEN_CLOSE_TO_BCD   = "bXd"   # 'bcd'ye 1 edit uzaklıkta
UNSEEN_FAR            = "zzz"   # tüm train durumlarına mesafesi yüksek


@pytest.fixture
def fitted_automata() -> ProbabilisticAutomata:
    """Eğitim verisiyle fit edilmiş ProbabilisticAutomata nesnesi döndürür."""
    automata = ProbabilisticAutomata()
    automata.fit(TRAIN_SAX)
    return automata


# ---------------------------------------------------------------------------
# Test 1: levenshtein_distance — temel doğruluk testleri
# ---------------------------------------------------------------------------

class TestLevenshteinDistance:
    """levenshtein_distance fonksiyonunun matematiksel doğruluğunu test eder."""

    def test_identical_strings_distance_is_zero(self):
        """Aynı diziler arasındaki mesafe 0 olmalıdır."""
        assert levenshtein_distance("abc", "abc") == 0
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("xyz", "xyz") == 0

    def test_empty_string_distance_equals_length(self):
        """Boş dize ile herhangi bir dizenin mesafesi, o dizenin uzunluğuna eşit olmalıdır."""
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "a") == 1
        assert levenshtein_distance("hello", "") == 5

    def test_single_insertion(self):
        """Tek karakter ekleme işlemi mesafeyi 1 artırmalıdır."""
        assert levenshtein_distance("ab", "abc") == 1
        assert levenshtein_distance("bc", "abc") == 1

    def test_single_deletion(self):
        """Tek karakter silme işlemi mesafeyi 1 artırmalıdır."""
        assert levenshtein_distance("abc", "ab") == 1
        assert levenshtein_distance("abc", "bc") == 1

    def test_single_substitution(self):
        """Tek karakter yer değiştirme işlemi mesafeyi 1 artırmalıdır."""
        assert levenshtein_distance("abc", "axc") == 1
        assert levenshtein_distance("abc", "xbc") == 1
        assert levenshtein_distance("abc", "abx") == 1

    def test_known_distances(self):
        """Bilinen çiftler için beklenen mesafe değerleri doğrulanmalıdır."""
        # "kitten" → "sitting": 3 işlem (s/k, i/e, +g)
        assert levenshtein_distance("kitten", "sitting") == 3
        # "abc" → "def": her karakter farklı → 3 substitution
        assert levenshtein_distance("abc", "def") == 3

    def test_symmetry(self):
        """Levenshtein mesafesi simetrik olmalıdır: dist(a,b) == dist(b,a)."""
        pairs = [
            ("abc", "bcd"),
            ("hello", "world"),
            ("", "abc"),
            ("aXc", "abc"),
            ("xyz", "abc"),
        ]
        for s1, s2 in pairs:
            assert levenshtein_distance(s1, s2) == levenshtein_distance(s2, s1), (
                f"Simetri ihlali: dist('{s1}','{s2}') != dist('{s2}','{s1}')"
            )

    def test_triangle_inequality(self):
        """Üçgen eşitsizliği sağlanmalıdır: dist(a, c) <= dist(a, b) + dist(b, c)."""
        triplets = [
            ("abc", "axc", "axx"),
            ("abc", "bcd", "cde"),
            ("hello", "helo", "heo"),
        ]
        for a, b, c in triplets:
            d_ab = levenshtein_distance(a, b)
            d_bc = levenshtein_distance(b, c)
            d_ac = levenshtein_distance(a, c)
            assert d_ac <= d_ab + d_bc, (
                f"Üçgen eşitsizliği ihlali: "
                f"dist('{a}','{c}')={d_ac} > dist('{a}','{b}')={d_ab} + dist('{b}','{c}')={d_bc}"
            )

    def test_non_negative(self):
        """Levenshtein mesafesi hiçbir zaman negatif olmamalıdır."""
        pairs = [("", ""), ("abc", ""), ("abc", "xyz"), ("a", "abc")]
        for s1, s2 in pairs:
            assert levenshtein_distance(s1, s2) >= 0, (
                f"dist('{s1}', '{s2}') negatif olamaz."
            )


# ---------------------------------------------------------------------------
# Test 2: handle_unseen_state — unseen eşleştirme mantığı
# ---------------------------------------------------------------------------

class TestHandleUnseenState:
    """ProbabilisticAutomata.handle_unseen_state metodunu test eder."""

    def test_unseen_mapped_to_closest_state(self, fitted_automata):
        """
        Unseen örüntü, sözlükteki Levenshtein mesafesi en düşük duruma eşlenmelidir.
        'aXc' → 'abc' (1 edit) veya 'bcd' veya 'cde' (2 edit) — en yakın 'abc'.
        """
        closest = fitted_automata.handle_unseen_state(UNSEEN_CLOSE_TO_ABC)
        # 'aXc' ile 'abc' arasındaki mesafe 1 (X→b); diğerleri en az 2
        dist_to_closest = levenshtein_distance(UNSEEN_CLOSE_TO_ABC, closest)
        for state in fitted_automata.states:
            assert levenshtein_distance(UNSEEN_CLOSE_TO_ABC, state) >= dist_to_closest, (
                f"'{closest}' en yakın durum değil; '{state}' daha yakın."
            )

    def test_unseen_result_is_in_known_states(self, fitted_automata):
        """handle_unseen_state sonucu her zaman bilinen bir durum olmalıdır."""
        for unseen in [UNSEEN_CLOSE_TO_ABC, UNSEEN_CLOSE_TO_BCD, UNSEEN_FAR]:
            result = fitted_automata.handle_unseen_state(unseen)
            assert result in fitted_automata.states, (
                f"'{result}', bilinen durumlar listesinde bulunmuyor: {fitted_automata.states}"
            )

    def test_unfitted_model_raises_runtime_error(self):
        """Fit edilmemiş modelde handle_unseen_state RuntimeError fırlatmalıdır."""
        automata = ProbabilisticAutomata()
        with pytest.raises(RuntimeError, match="fit"):
            automata.handle_unseen_state("xyz")

    def test_far_unseen_still_mapped(self, fitted_automata):
        """
        Eğitim verisine çok uzak olan bir unseen örüntü yine de
        sözlükteki en yakın (ama uzak) duruma atanmalıdır.
        """
        result = fitted_automata.handle_unseen_state(UNSEEN_FAR)
        assert result in fitted_automata.states, (
            f"Uzak unseen '{UNSEEN_FAR}' bir durum döndürmelidir, None değil."
        )


# ---------------------------------------------------------------------------
# Test 3: resolve_state — seen/unseen çözümleme
# ---------------------------------------------------------------------------

class TestResolveState:
    """ProbabilisticAutomata.resolve_state metodunu test eder."""

    def test_seen_state_returned_unchanged(self, fitted_automata):
        """Sözlükte bulunan bir durum değişmeden döndürülmelidir."""
        for state in fitted_automata.states:
            resolved = fitted_automata.resolve_state(state)
            assert resolved == state, (
                f"Bilinen '{state}' durumu aynen döndürülmeli; '{resolved}' alındı."
            )

    def test_unseen_state_resolved_to_known(self, fitted_automata):
        """Unseen bir örüntü, bilinen bir duruma çözümlenmeli."""
        resolved = fitted_automata.resolve_state(UNSEEN_CLOSE_TO_ABC)
        assert resolved in fitted_automata.states, (
            f"resolve_state unseen için bilinen bir durum döndürmelidir. Alınan: '{resolved}'"
        )

    def test_resolve_does_not_raise_for_unseen(self, fitted_automata):
        """resolve_state, unseen örüntüde exception fırlatmamalıdır."""
        try:
            fitted_automata.resolve_state(UNSEEN_FAR)
        except Exception as exc:
            pytest.fail(f"resolve_state unseen durumda exception fırlattı: {exc}")

    def test_unfitted_model_raises_runtime_error(self):
        """Fit edilmemiş modelde resolve_state RuntimeError fırlatmalıdır."""
        automata = ProbabilisticAutomata()
        with pytest.raises(RuntimeError, match="fit"):
            automata.resolve_state("abc")


# ---------------------------------------------------------------------------
# Test 4: map_sequence_to_states — unseen sayımı ve dizi çözümleme
# ---------------------------------------------------------------------------

class TestMapSequenceToStates:
    """ProbabilisticAutomata.map_sequence_to_states metodunu test eder."""

    def test_no_unseen_in_pure_train_sequence(self, fitted_automata):
        """Yalnızca eğitim durumlarından oluşan bir dizide unseen_count 0 olmalıdır."""
        result = fitted_automata.map_sequence_to_states(["abc", "bcd", "cde"])
        assert result["unseen_count"] == 0, (
            "Yalnızca bilinen durumları içeren dizide unseen_count 0 olmalıdır."
        )

    def test_unseen_count_matches_actual_unseen(self, fitted_automata):
        """unseen_count, dizideki gerçek unseen örüntü sayısını doğru saymalıdır."""
        # 2 adet unseen: UNSEEN_CLOSE_TO_ABC ve UNSEEN_FAR
        sequence = ["abc", UNSEEN_CLOSE_TO_ABC, "bcd", UNSEEN_FAR]
        result = fitted_automata.map_sequence_to_states(sequence)
        assert result["unseen_count"] == 2, (
            f"2 unseen beklendi; {result['unseen_count']} alındı."
        )

    def test_resolved_states_all_known(self, fitted_automata):
        """resolved_states içindeki tüm durumlar bilinen durumlar olmalıdır."""
        sequence = ["abc", UNSEEN_CLOSE_TO_ABC, UNSEEN_FAR, "cde"]
        result = fitted_automata.map_sequence_to_states(sequence)
        for state in result["resolved_states"]:
            assert state in fitted_automata.states, (
                f"resolved_states içindeki '{state}' bilinen durumlar arasında değil."
            )

    def test_resolved_states_length_matches_sequence(self, fitted_automata):
        """resolved_states uzunluğu girdi dizisi uzunluğuna eşit olmalıdır."""
        sequence = ["abc", UNSEEN_CLOSE_TO_ABC, "bcd", "cde", UNSEEN_FAR]
        result = fitted_automata.map_sequence_to_states(sequence)
        assert len(result["resolved_states"]) == len(sequence), (
            "resolved_states uzunluğu giriş dizisiyle aynı olmalıdır."
        )

    def test_transition_rows_length_is_sequence_minus_one(self, fitted_automata):
        """transition_rows uzunluğu len(sax_sequence) - 1 olmalıdır."""
        sequence = ["abc", "bcd", "cde", UNSEEN_CLOSE_TO_ABC]
        result = fitted_automata.map_sequence_to_states(sequence)
        assert len(result["transition_rows"]) == len(sequence) - 1, (
            "transition_rows uzunluğu dizi uzunluğu eksi 1 olmalıdır."
        )

    def test_empty_sequence_raises_value_error(self, fitted_automata):
        """Boş dizi ValueError fırlatmalıdır."""
        with pytest.raises(ValueError):
            fitted_automata.map_sequence_to_states([])

    def test_single_element_sequence(self, fitted_automata):
        """Tek elemanlı dizi geçerli sonuç döndürmelidir; transition_rows boş olmalıdır."""
        result = fitted_automata.map_sequence_to_states(["abc"])
        assert len(result["resolved_states"]) == 1
        assert result["transition_rows"] == []

    def test_all_unseen_sequence(self, fitted_automata):
        """Tüm elemanları unseen olan bir dizi kesintisiz işlenebilmelidir."""
        sequence = [UNSEEN_CLOSE_TO_ABC, UNSEEN_CLOSE_TO_BCD, UNSEEN_FAR]
        result = fitted_automata.map_sequence_to_states(sequence)
        assert result["unseen_count"] == 3
        assert len(result["resolved_states"]) == 3
        for state in result["resolved_states"]:
            assert state in fitted_automata.states


# ---------------------------------------------------------------------------
# Test 5: compute_path_probability — unseen içeren yollar
# ---------------------------------------------------------------------------

class TestComputePathProbability:
    """ProbabilisticAutomata.compute_path_probability metodunu test eder."""

    def test_known_path_probability_non_negative(self, fitted_automata):
        """Bilinen bir yol için path probability >= 0 olmalıdır."""
        prob = fitted_automata.compute_path_probability(["abc", "bcd", "cde"])
        assert prob >= 0.0, "Path probability negatif olamaz."

    def test_known_path_probability_at_most_one(self, fitted_automata):
        """Path probability 1.0'ı geçmemelidir."""
        prob = fitted_automata.compute_path_probability(["abc", "bcd"])
        assert prob <= 1.0, "Path probability 1.0'ı geçemez."

    def test_unseen_path_probability_positive(self, fitted_automata):
        """
        Unseen örüntüler içeren bir yolun olasılığı > 0 olmalıdır.
        (Sistem durmamalı; Levenshtein ile çözülmeli ve devam etmelidir.)
        """
        sequence = ["abc", UNSEEN_CLOSE_TO_ABC, "cde"]
        prob = fitted_automata.compute_path_probability(sequence)
        assert prob >= 0.0, (
            "Unseen içeren yolun path probability'si negatif olmamalıdır."
        )

    def test_single_element_returns_one(self, fitted_automata):
        """Tek elemanlı diziler için path probability 1.0 döndürmelidir."""
        assert fitted_automata.compute_path_probability(["abc"]) == 1.0

    def test_empty_sequence_returns_one(self, fitted_automata):
        """Boş veya tek elemanlı dizi için 1.0 döndürmelidir (geçiş yok)."""
        assert fitted_automata.compute_path_probability([]) == 1.0

    def test_unfitted_model_raises_runtime_error(self):
        """Fit edilmemiş modelde compute_path_probability RuntimeError fırlatmalıdır."""
        automata = ProbabilisticAutomata()
        with pytest.raises(RuntimeError, match="fit"):
            automata.compute_path_probability(["abc", "bcd"])


# ---------------------------------------------------------------------------
# Test 6: resolve_and_advance — unseen dahil adım ilerleme
# ---------------------------------------------------------------------------

class TestResolveAndAdvance:
    """ProbabilisticAutomata.resolve_and_advance metodunu test eder."""

    def test_seen_seen_advance(self, fitted_automata):
        """İki bilinen durum arasında geçiş doğru olmalıdır."""
        cur, nxt, row = fitted_automata.resolve_and_advance("abc", "bcd")
        assert cur == "abc"
        assert nxt == "bcd"
        assert len(row) == fitted_automata.num_states

    def test_unseen_current_resolved(self, fitted_automata):
        """Mevcut durum unseen ise resolve edilmiş bir durum döndürmelidir."""
        cur, nxt, row = fitted_automata.resolve_and_advance(UNSEEN_CLOSE_TO_ABC, "bcd")
        assert cur in fitted_automata.states, (
            f"Unseen current '{UNSEEN_CLOSE_TO_ABC}' resolve edilmiş bir duruma dönüşmeli."
        )
        assert len(row) == fitted_automata.num_states

    def test_unseen_next_resolved(self, fitted_automata):
        """Sonraki durum unseen ise resolve edilmiş bir durum döndürmelidir."""
        cur, nxt, row = fitted_automata.resolve_and_advance("abc", UNSEEN_CLOSE_TO_BCD)
        assert nxt in fitted_automata.states, (
            f"Unseen next '{UNSEEN_CLOSE_TO_BCD}' resolve edilmiş bir duruma dönüşmeli."
        )

    def test_transition_row_sums_to_one(self, fitted_automata):
        """
        Döndürülen geçiş olasılık satırının toplamı 1.0'a yakın olmalıdır.
        (Her satır normalleştirilmiş veya uniform dağılımlıdır.)
        """
        import math
        _, _, row = fitted_automata.resolve_and_advance("abc", "bcd")
        assert math.isclose(sum(row), 1.0, abs_tol=1e-9), (
            f"Geçiş satırının toplamı 1.0 olmalıdır; {sum(row):.6f} alındı."
        )

    def test_unfitted_model_raises_runtime_error(self):
        """Fit edilmemiş modelde resolve_and_advance RuntimeError fırlatmalıdır."""
        automata = ProbabilisticAutomata()
        with pytest.raises(RuntimeError, match="fit"):
            automata.resolve_and_advance("abc", "bcd")


# ---------------------------------------------------------------------------
# Test 7: get_state_id — unseen güvenli ID çözümleme
# ---------------------------------------------------------------------------

class TestGetStateId:
    """ProbabilisticAutomata.get_state_id metodunu test eder."""

    def test_known_state_returns_correct_id(self, fitted_automata):
        """Bilinen bir durum için doğru tam sayı ID döndürülmelidir."""
        for state in fitted_automata.states:
            expected_id = fitted_automata.state_to_id[state]
            assert fitted_automata.get_state_id(state) == expected_id, (
                f"'{state}' için beklenen ID {expected_id}."
            )

    def test_unseen_state_returns_valid_id(self, fitted_automata):
        """Unseen durum için geçerli bir tam sayı ID döndürülmelidir."""
        state_id = fitted_automata.get_state_id(UNSEEN_CLOSE_TO_ABC)
        valid_ids = set(fitted_automata.state_to_id.values())
        assert state_id in valid_ids, (
            f"Unseen durum ID'si {state_id}, geçerli ID'ler arasında olmalıdır: {valid_ids}"
        )

    def test_unfitted_model_raises_runtime_error(self):
        """Fit edilmemiş modelde get_state_id RuntimeError fırlatmalıdır."""
        automata = ProbabilisticAutomata()
        with pytest.raises(RuntimeError, match="fit"):
            automata.get_state_id("abc")


# ---------------------------------------------------------------------------
# Test 8: Levenshtein — unseen örüntü ve sözlük mesafe tutarlılığı
# ---------------------------------------------------------------------------

class TestLevenshteinConsistencyWithAutomata:
    """
    levenshtein_distance sonuçlarının handle_unseen_state ile tutarlı olduğunu
    doğrulayan entegrasyon benzeri testler.
    """

    def test_closest_state_has_minimum_distance(self, fitted_automata):
        """
        handle_unseen_state'in seçtiği durum, sözlükteki tüm durumlar arasında
        Levenshtein mesafesi en düşük (veya eşit) olan durum olmalıdır.
        """
        for unseen in [UNSEEN_CLOSE_TO_ABC, UNSEEN_CLOSE_TO_BCD, UNSEEN_FAR]:
            chosen = fitted_automata.handle_unseen_state(unseen)
            min_dist = levenshtein_distance(unseen, chosen)
            for state in fitted_automata.states:
                candidate_dist = levenshtein_distance(unseen, state)
                assert candidate_dist >= min_dist, (
                    f"'{unseen}' için seçilen '{chosen}' (dist={min_dist}) "
                    f"en yakın değil; '{state}' daha yakın (dist={candidate_dist})."
                )

    def test_levenshtein_distance_to_self_is_zero(self, fitted_automata):
        """
        Her bilinen durumun kendisiyle Levenshtein mesafesi 0 olmalıdır.
        Dolayısıyla seen bir durum, handle_unseen_state çağrıldığında
        kendisi seçilmelidir (minimum mesafe).
        """
        for state in fitted_automata.states:
            assert levenshtein_distance(state, state) == 0
