# Proje Kuralları ve Özn Hazırlık (Sürekli Kontrol Edilecek)

> [!IMPORTANT]
> **Zorunlu Kurallar:**
> - **Data Leakage Yasak:** Normalizasyon, PCA, SAX/PAA sözlüğü ve otomata geçiş olasılıkları SADECE eğitim (train) verisinde oluşturulmalıdır.
> - **Sabit Değer (Hard-coded) Yasak:** Tüm değişkenler (window_size=4, alphabet_size=3 vb.) `config.yaml` üzerinden okunmalıdır.
> - **Bölme (Split) Kuralı:** SKAB için rastgele bölmek yasak, `GroupKFold` kullanılacak (`source_file` bazında). BATADAL için rastgele bölmek yasak, zaman sırasıyla %60, %20, %20 ayrılacak.
> - **Otomata ve PCA Kuralı:** Otomata sadece tek boyutlu veri alır, bu yüzden veriler PCA ile 1 boyuta (PC1) indirilecek.
> - **Unseen Kuralı:** Eğitimde olmayan bir örüntü gelirse Levenshtein (Edit Distance) ile en yakın olana atanacak ve sistem oradan devam edecek. Birim testi zorunlu!
> - **Deneyler:** 5 farklı random seed (42, 123, 2026, 7, 999) ile tekrarlanacak.

> [!TIP]
> **Temiz Kod (Clean Code) Yazımı:**
> Kodun sadece çalışması yeterli değildir; başkaları (veya 6 ay sonraki haliniz) tarafından okunabilir ve geliştirilebilir olması gerekir.
> - **Anlamlı İsimlendirme:** Değişken ve fonksiyon isimleri, ne iş yaptıklarını açıkça belli etmelidir (Örn: `d` yerine `gunluk_kazanc`).
> - **DRY (Don't Repeat Yourself):** Aynı kod bloğunu birden fazla yerde kullanmayın; bunun yerine fonksiyonlar veya sınıflar oluşturun.
> - **Fonksiyon Boyutu:** Bir fonksiyon tek bir işi yapmalı ve mümkün olduğunca kısa olmalıdır.

# Geliştirme Takvimi ve Commit Sırası (Bitiş: 7 Haziran 2026)

Bu liste, 50-60 commit olacak şekilde sizin (EMIR) ve arkadaşınızın (ARKADAS) sırayla atacağı commitleri belirlemektedir. İşareti `[/]` olanlar üzerinde çalıştığımız, `[x]` olanlar tamamlanan adımlardır.

## Aşama 1: Proje Kurulumu ve Veri İndirme (Commit 1 - 8)
- `[x]` (EMIR) İlk commit: Proje klasör yapısı, `.gitignore` ve `README.md` eklendi.
- `[x]` (ARKADAS) config.yaml oluşturuldu, sabit model parametreleri tanımlandı.
- `[x]` (EMIR) requirements.txt eklendi (TensorFlow, pandas, scikit-learn vb.).
- `[x]` (ARKADAS) SKAB veri setini indiren ve otomatik `valve1`, `valve2` klasörlerini okuyan script yazıldı.
- `[x]` (EMIR) SKAB verilerini concat ile birleştiren ve source_file, source_group sütunlarını ekleyen kod yazıldı.
- `[x]` (ARKADAS) BATADAL Training Dataset 2'yi indiren ve yükleyen script yazıldı.
- `[x]` (EMIR) Veri yükleme fonksiyonlarına test (loglama) eklendi.
- `[x]` (ARKADAS) Veri setlerinin ilk analizlerini (shape, missing values) basan bir keşif kodu yazıldı.

## Aşama 2: Veri Ön İşleme (Commit 9 - 18)
- `[x]` (EMIR) SKAB verisini `source_file` kullanarak GroupKFold'a göre bölen fonksiyon eklendi.
- `[x]` (ARKADAS) BATADAL verisini zaman sırasına göre %60-%20-%20 oranında bölen fonksiyon eklendi.
- `[x]` (EMIR) Train verisi üzerinden scaler fit eden ve tüm setlere uygulayan normalizasyon modülü eklendi.
- `[x]` (ARKADAS) Train verisi üzerinden PCA (1 bileşen) uygulayan modül eklendi.
- `[x]` (EMIR) Veri sızıntısını engelleyecek şekilde preprocess adımlarını bağlayan pipeline oluşturuldu.
- `[x]` (ARKADAS) Pipeline için birim testi yazıldı (test veri setine fit edilmediğinden emin olundu).
- `[x]` (EMIR) Zaman bağımsız veri setleri (X_train, y_train vs.) diske kaydedilme mantığı eklendi.
- `[x]` (ARKADAS) Gürültülü veri senaryosu için Gaussian Noise ekleyen fonksiyon eklendi.
- `[x]` (EMIR) Sliding window işlemi ile verileri (window_size) pencerelerine bölen algoritma yazıldı.
- `[X]` (ARKADAS) Veri ön işleme tamamlandı, pipeline baştan sona entegre edildi.

## Aşama 3: Otomata Modeli ve SAX (Commit 19 - 30)
- `[x]` (EMIR) PAA (Piecewise Aggregate Approximation) algoritması eklendi.
- `[x]` (ARKADAS) SAX (Symbolic Aggregate approXimation) algoritması eklendi.
- `[x]` (EMIR) Sadece train verisi üzerinden SAX sözlüğü çıkaran ve pattern belirleyen kod yazıldı.
- `[x]` (ARKADAS) Train içerisindeki state (durum) listesini belirleyen yapı eklendi.
- `[x]` (EMIR) Durumlar arası (State transitions) geçiş sayılarını sayan algoritma oluşturuldu.
- `[x]` (ARKADAS) Sayımları frekans tabanlı geçiş olasılıklarına (Transition Probabilities) çeviren matematiksel mantık eklendi.
- `[x]` (EMIR) Olasılıksal Otomata ana sınıfı (Probabilistic Automata Class) yaratıldı.
- `[x]` (ARKADAS) Unseen durum tespiti mantığı sisteme dahil edildi.
- `[x]` (EMIR) Levenshtein (Edit distance) mesafe bulma algoritması uygulandı.
- `[x]` (ARKADAS) Unseen durumunda en yakın pattern'ı bulup sistemin oradan devam etmesini sağlayan eşleştirme yazıldı.
- `[x]` (EMIR) Path probability hesaplayan (ardışık geçiş olasılıklarının çarpımı) metot eklendi.
- `[x]` (ARKADAS) Unseen davranışı ve Levenshtein için zorunlu birim testleri (unit tests) yazıldı.

## Aşama 4: Derin Öğrenme Modelleri (Commit 31 - 40)
- `[x]` (EMIR) Keras/TensorFlow importları ve genel model class yapısı kuruldu.
- `[x]` (ARKADAS) 1D-CNN modeli `config.yaml` parametrelerine bağlı olarak tanımlandı.
- `[x]` (EMIR) LSTM modeli `config.yaml` parametrelerine bağlı olarak tanımlandı.
- `[x]` (ARKADAS) Model eğitim fonksiyonu (fit_model), batch_size=32 kuralı ile yazıldı.
- `[x]` (EMIR) Early stopping callback (patience=5, validation_loss takibi) eklendi.
- `[x]` (ARKADAS) Model tahminleri ve sonuçlarını döndüren fonksiyonlar (predict_model) yazıldı.
- `[ ]` (EMIR) Modellerin eğitilmesi için tek bir döngü (train_all_models) eklendi.
- `[ ]` (ARKADAS) Eğitim sürecindeki loss değerlerini loglayan callback sistemi yapıldı.
- `[ ]` (EMIR) Eğitilmiş modelleri diske kaydeden ağırlık (weights) yönetim sistemi eklendi.
- `[ ]` (ARKADAS) Modeller arası performans çıkarımı (inference) script'i hazırlandı.

## Aşama 5: Açıklanabilirlik ve Değerlendirme (Commit 41 - 55)
- `[ ]` (EMIR) Accuracy, Precision, Recall, F1-score hesaplayan metrikler modülü yazıldı.
- `[ ]` (ARKADAS) Otomata için "Güven Skoru (Confidence Score)" hesaplama mantığı eklendi.
- `[ ]` (EMIR) Karar sürecini zorunlu formata (JSON) çeviren "Explainability Modülü" oluşturuldu.
- `[ ]` (ARKADAS) Model karşılaştırmalarını loglayacak deney kayıt sistemi (CSV vb.) kuruldu.
- `[ ]` (EMIR) 5 farklı seed (42, 123, 2026, 7, 999) üzerinde dönecek ana script (`main.py`) revize edildi.
- `[ ]` (ARKADAS) Gürültülü ve unseen veriler üzerinde de test yapacak ayrı deneme senaryosu eklendi.
- `[ ]` (EMIR) Window size (3,4,5,6) ve Alphabet size (3,4,5,6) parametre varyasyonları için loop kuruldu.
- `[ ]` (ARKADAS) Automata state diyagramı (graph) çizen görsellik fonksiyonu yazıldı.
- `[ ]` (EMIR) Transition probability heatmap çizen görsellik fonksiyonu eklendi.
- `[ ]` (ARKADAS) Confusion matrix ve performans metriklerini tablo/görsel yapan raporlama eklendi.
- `[ ]` (EMIR) Parametre duyarlılık grafikleri oluşturuldu.
- `[ ]` (ARKADAS) Orijinal ve Gürültülü sonuçların farkını karşılaştıran script yazıldı.
- `[ ]` (EMIR) Wilcoxon / McNemar testleri ile modellerin istatistiksel anlamlılık analizi eklendi.
- `[ ]` (ARKADAS) SKAB sonuçlarını fold bazında, BATADAL sonuçlarını test veri setinde formatlayıp çıkartan kodlar yazıldı.
- `[ ]` (EMIR) Projenin baştan sona (end-to-end) birleştirilip test edilmesi.

## Aşama 6: Raporlama ve Final (Commit 56 - 60)
- `[ ]` (ARKADAS) `readme.md` dosyasının projenin raporlama isterlerine uygun şekilde (Markdown formatında) yazılması.
- `[ ]` (EMIR) Tüm kodun PEP-8 standartlarına uyacak şekilde formatlanması ve gereksiz logların temizlenmesi.
- `[ ]` (ARKADAS) Rapor için elde edilen grafiklerin ve tabloların `readme.md` içine dahil edilmesi.
- `[ ]` (EMIR) Kapsamlı kod review'i ve son hata denetimi (bug fixes).
- `[ ]` (ARKADAS) Github üzerinde Release alınıp, proje teslimine hazır hale getirilmesi.
