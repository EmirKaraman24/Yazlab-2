"""
Derin Öğrenme Modelleri (Deep Learning Models) modülü.
Bu modül TensorFlow/Keras importlarını ve tüm derin öğrenme modelleri
(LSTM, 1D-CNN vb.) için ortak olan temel sınıf (base class) yapısını içerir.
"""

import logging
import yaml
import numpy as np

# Keras ve TensorFlow importları
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str = "config.yaml") -> dict:
    """
    config.yaml dosyasını okur ve içeriğini bir sözlük olarak döndürür.

    Parameters
    ----------
    config_path : str
        YAML konfigürasyon dosyasının yolu. Varsayılan: "config.yaml".

    Returns
    -------
    dict
        Konfigürasyon parametrelerini içeren sözlük.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


class BaseTimeSeriesModel:
    """
    Zaman serisi tahmini veya anomali tespiti için temel derin öğrenme modeli sınıfı.
    Tüm modeller (ör. LSTMModel, CNN1DModel) bu sınıftan türetilecek ve
    kendi `build_model` metodlarını implement edeceklerdir.
    """

    def __init__(self, sequence_length: int, num_features: int, model_name: str = "BaseModel"):
        """
        Parameters
        ----------
        sequence_length : int
            Girdi zaman serisi dizisinin uzunluğu (ör. window_size).
        num_features : int
            Her bir zaman adımındaki özellik sayısı (genellikle PCA sonrası 1).
        model_name : str, optional
            Modelin adı, varsayılan "BaseModel".
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model_name = model_name
        self.model = None

    def build_model(self):
        """
        Keras model mimarisini oluşturur ve derler (compile).
        Alt sınıflar tarafından mutlaka ezilmelidir (override).
        """
        raise NotImplementedError("Alt sınıflar `build_model` metodunu tanımlamalıdır.")

    def summary(self):
        """Model mimarisi özetini yazdırır."""
        if self.model is not None:
            self.model.summary()
        else:
            logging.warning(f"[{self.model_name}] Model henüz oluşturulmadı (build_model() çağrılmalı).")

    def fit_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, config_path: str = "config.yaml", **kwargs):
        """
        Keras modelini eğitir. `batch_size=32` kuralı dahil olmak üzere
        hiperparametreler `config.yaml` üzerinden alınır.

        Early stopping, `validation_loss` (val_loss) takibiyle etkinleştirilir;
        `patience` değeri `config.yaml` içindeki `early_stopping_patience`
        anahtarından okunur (varsayılan: 5).

        Parameters
        ----------
        X_train : np.ndarray
            Eğitim verisi.
        y_train : np.ndarray
            Eğitim etiketleri.
        X_val : np.ndarray, optional
            Doğrulama (validation) özellikleri. Early stopping için zorunludur;
            sağlanmadığında early stopping devre dışı kalır.
        y_val : np.ndarray, optional
            Doğrulama (validation) etiketleri.
        config_path : str
            Konfigürasyon dosyasının yolu.
        **kwargs
            Modelin `fit` metoduna iletilecek ek parametreler (örn: ek callbacks).

        Returns
        -------
        tf.keras.callbacks.History
            Eğitim işlemine ait loss ve metrik geçmişi.
        """
        if self.model is None:
            raise RuntimeError(f"[{self.model_name}] Model oluşturulmamış! Önce `build_model()` çalıştırılmalı.")

        config = load_config(config_path)
        dl_params = config.get("deep_learning_params", {})

        batch_size = dl_params.get("batch_size", 32)
        epochs = dl_params.get("epochs", 50)
        early_stopping_patience = dl_params.get("early_stopping_patience", 5)

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        # Early stopping: yalnızca doğrulama verisi mevcutsa etkinleştirilir
        callback_list = kwargs.pop("callbacks", [])
        if validation_data is not None:
            early_stopping_callback = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
            callback_list = [early_stopping_callback] + list(callback_list)
            logging.info(
                "[%s] EarlyStopping etkin: monitor='val_loss', patience=%d, restore_best_weights=True",
                self.model_name,
                early_stopping_patience,
            )
        else:
            logging.warning(
                "[%s] Doğrulama verisi sağlanmadı; EarlyStopping devre dışı.",
                self.model_name,
            )

        logging.info(f"[{self.model_name}] Model eğitimi başlatılıyor... (Epochs: {epochs}, Batch Size: {batch_size})")

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            **kwargs
        )

        logging.info(f"[{self.model_name}] Eğitim süreci tamamlandı.")
        return history

    def predict_proba(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Girdi dizileri için ham sigmoid olasılık skorlarını döndürür.

        Parameters
        ----------
        X : np.ndarray
            Tahmin yapılacak girdi dizileri, şekil: (n_samples, sequence_length, num_features).
        batch_size : int, optional
            Tahmin sırasında kullanılacak toplu iş boyutu. Varsayılan: 32.

        Returns
        -------
        np.ndarray
            Her örnek için [0, 1] aralığında anomali olasılık skoru,
            şekil: (n_samples,).
        """
        if self.model is None:
            raise RuntimeError(
                f"[{self.model_name}] Model oluşturulmamış! Önce `build_model()` çalıştırılmalı."
            )

        logging.info(
            "[%s] predict_proba çalıştırılıyor: %d örnek, batch_size=%d",
            self.model_name,
            len(X),
            batch_size,
        )

        raw_scores = self.model.predict(X, batch_size=batch_size, verbose=0)
        # Keras çıktısı (n_samples, 1) şeklindedir; düzleştirilerek (n_samples,) döndürülür
        return raw_scores.flatten()

    def predict_model(
        self,
        X: np.ndarray,
        y_true: np.ndarray = None,
        threshold: float = 0.5,
        batch_size: int = 32,
        config_path: str = "config.yaml",
    ) -> dict:
        """
        Model tahminlerini ve sonuç özetini döndürür.

        Ham sigmoid skorları `threshold` eşiğiyle ikili (0/1) etiketlere
        dönüştürülür. Eşik değeri, config.yaml içindeki
        `prediction_threshold` anahtarından okunur; anahtar yoksa
        `threshold` parametresi kullanılır.

        Parameters
        ----------
        X : np.ndarray
            Tahmin yapılacak girdi dizileri, şekil: (n_samples, sequence_length, num_features).
        y_true : np.ndarray, optional
            Gerçek etiketler. Sağlanırsa sonuç sözlüğüne `y_true` alanı eklenir.
        threshold : float, optional
            Sigmoid skorunu 0/1 etikete çevirmek için kullanılan eşik.
            config.yaml'da `prediction_threshold` tanımlıysa oradan okunur.
            Varsayılan: 0.5.
        batch_size : int, optional
            Tahmin sırasında kullanılacak toplu iş boyutu. Varsayılan: 32.
        config_path : str, optional
            Konfigürasyon dosyasının yolu.

        Returns
        -------
        dict
            Aşağıdaki anahtarları içeren sonuç sözlüğü:

            - ``model_name``  : Modelin adı (str).
            - ``probabilities``: Ham sigmoid skorları, şekil (n_samples,) (np.ndarray).
            - ``predictions``  : İkili tahmin etiketleri (0 veya 1), şekil (n_samples,) (np.ndarray).
            - ``threshold``    : Kullanılan eşik değeri (float).
            - ``n_samples``    : Tahmin yapılan örnek sayısı (int).
            - ``n_anomalies``  : Anomali olarak etiketlenen örnek sayısı (int).
            - ``y_true``       : Gerçek etiketler — yalnızca `y_true` sağlandığında (np.ndarray).
        """
        config = load_config(config_path)
        dl_params = config.get("deep_learning_params", {})
        # config.yaml'da tanımlı eşik değeri, parametre eşiğini geçersiz kılar
        threshold = dl_params.get("prediction_threshold", threshold)

        probabilities = self.predict_proba(X, batch_size=batch_size)
        predictions = (probabilities >= threshold).astype(int)

        n_anomalies = int(predictions.sum())
        logging.info(
            "[%s] Tahmin tamamlandı: %d örnek, eşik=%.2f, anomali=%d",
            self.model_name,
            len(X),
            threshold,
            n_anomalies,
        )

        result = {
            "model_name": self.model_name,
            "probabilities": probabilities,
            "predictions": predictions,
            "threshold": threshold,
            "n_samples": len(X),
            "n_anomalies": n_anomalies,
        }

        if y_true is not None:
            result["y_true"] = np.asarray(y_true)

        return result


class CNN1DModel(BaseTimeSeriesModel):
    """
    config.yaml parametrelerine bağlı 1D Evrişimli Sinir Ağı (1D-CNN) modeli.

    Mimari:
        Conv1D → MaxPooling1D → Conv1D → GlobalAveragePooling1D →
        Dense (ReLU) → Dropout → Dense (Sigmoid çıkış)

    Tüm hiperparametreler (filtre sayısı, kernel boyutu, dropout oranı vb.)
    config.yaml içindeki `deep_learning_params` bloğundan okunur; herhangi
    bir hard-coded değer kullanılmaz.
    """

    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        config_path: str = "config.yaml",
    ):
        """
        Parameters
        ----------
        sequence_length : int
            Girdi penceresinin uzunluğu (ör. window_size).
        num_features : int
            Her zaman adımındaki özellik sayısı (PCA sonrası genellikle 1).
        config_path : str
            Konfigürasyon dosyasının yolu. Varsayılan: "config.yaml".
        """
        super().__init__(
            sequence_length=sequence_length,
            num_features=num_features,
            model_name="1D-CNN",
        )
        config = load_config(config_path)
        dl_params = config["deep_learning_params"]

        # Mimariyle ilgili hiperparametreler — tamamı config.yaml'dan okunur
        self.filters = dl_params["cnn_filters"]
        self.kernel_size = dl_params["cnn_kernel_size"]
        self.pool_size = dl_params["cnn_pool_size"]
        self.dense_units = dl_params["cnn_dense_units"]
        self.dropout_rate = dl_params["cnn_dropout_rate"]
        self.learning_rate = dl_params["learning_rate"]

        logging.info(
            "[1D-CNN] Hiperparametreler yüklendi: "
            "filters=%d, kernel_size=%d, pool_size=%d, "
            "dense_units=%d, dropout_rate=%.2f, learning_rate=%.4f",
            self.filters,
            self.kernel_size,
            self.pool_size,
            self.dense_units,
            self.dropout_rate,
            self.learning_rate,
        )

    def build_model(self) -> tf.keras.Model:
        """
        1D-CNN mimarisini oluşturur, derler (compile) ve döndürür.

        Mimarinin katmanları:
            1. Conv1D  : İlk evrişim bloğu — yerel örüntüleri yakalar.
            2. MaxPooling1D : Özellik haritalarını boyutsal olarak küçültür.
            3. Conv1D  : İkinci evrişim bloğu — daha soyut örüntüler öğrenir.
            4. GlobalAveragePooling1D : Zamansal boyutu düzleştirir.
            5. Dense   : Tam bağlantılı öğrenme katmanı.
            6. Dropout : Aşırı öğrenmeyi (overfitting) önler.
            7. Dense   : Sigmoid aktivasyonlu ikili sınıflandırma çıkış katmanı.

        Returns
        -------
        tf.keras.Model
            Derlenmiş Keras modeli.
        """
        input_layer = layers.Input(
            shape=(self.sequence_length, self.num_features),
            name="input_sequence",
        )

        # --- Evrişim bloğu 1 ---
        x = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation="relu",
            padding="same",
            name="conv1d_block1",
        )(input_layer)
        x = layers.MaxPooling1D(
            pool_size=self.pool_size,
            padding="same",
            name="maxpool_block1",
        )(x)

        # --- Evrişim bloğu 2 ---
        x = layers.Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            activation="relu",
            padding="same",
            name="conv1d_block2",
        )(x)
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

        # --- Tam bağlantılı katmanlar ---
        x = layers.Dense(self.dense_units, activation="relu", name="dense_hidden")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout")(x)
        output_layer = layers.Dense(1, activation="sigmoid", name="output")(x)

        self.model = models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=self.model_name,
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        logging.info("[1D-CNN] Model başarıyla oluşturuldu ve derlendi.")
        return self.model


class LSTMModel(BaseTimeSeriesModel):
    """
    config.yaml parametrelerine bağlı Uzun Kısa Süreli Bellek (LSTM) modeli.

    Mimari:
        LSTM → Dropout → Dense (ReLU) → Dense (Sigmoid çıkış)

    Hiperparametreler config.yaml içindeki `deep_learning_params` bloğundan okunur.
    """

    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        config_path: str = "config.yaml",
    ):
        """
        Parameters
        ----------
        sequence_length : int
            Girdi penceresinin uzunluğu (ör. window_size).
        num_features : int
            Her zaman adımındaki özellik sayısı (PCA sonrası genellikle 1).
        config_path : str
            Konfigürasyon dosyasının yolu. Varsayılan: "config.yaml".
        """
        super().__init__(
            sequence_length=sequence_length,
            num_features=num_features,
            model_name="LSTM",
        )
        config = load_config(config_path)
        dl_params = config["deep_learning_params"]

        self.lstm_units = dl_params["lstm_units"]
        self.dropout_rate = dl_params["lstm_dropout_rate"]
        self.dense_units = dl_params["lstm_dense_units"]
        self.learning_rate = dl_params["learning_rate"]

        logging.info(
            "[LSTM] Hiperparametreler yüklendi: "
            "lstm_units=%d, dropout_rate=%.2f, dense_units=%d, learning_rate=%.4f",
            self.lstm_units,
            self.dropout_rate,
            self.dense_units,
            self.learning_rate,
        )

    def build_model(self) -> tf.keras.Model:
        """
        LSTM mimarisini oluşturur, derler ve döndürür.
        """
        input_layer = layers.Input(
            shape=(self.sequence_length, self.num_features),
            name="input_sequence",
        )

        x = layers.LSTM(
            units=self.lstm_units,
            return_sequences=False,
            name="lstm_layer",
        )(input_layer)

        x = layers.Dropout(self.dropout_rate, name="dropout")(x)
        x = layers.Dense(self.dense_units, activation="relu", name="dense_hidden")(x)
        output_layer = layers.Dense(1, activation="sigmoid", name="output")(x)

        self.model = models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=self.model_name,
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        logging.info("[LSTM] Model başarıyla oluşturuldu ve derlendi.")
        return self.model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    config_path: str = "config.yaml"
) -> dict:
    """
    config.yaml'da belirtilen tüm derin öğrenme modellerini eğitir ve eğitilmiş
    model nesnelerini içeren bir sözlük döndürür.

    Parameters
    ----------
    X_train : np.ndarray
        Eğitim özellikleri, şekil: (n_samples, sequence_length, num_features).
    y_train : np.ndarray
        Eğitim etiketleri, şekil: (n_samples,).
    X_val : np.ndarray, optional
        Doğrulama özellikleri.
    y_val : np.ndarray, optional
        Doğrulama etiketleri.
    config_path : str
        Konfigürasyon dosyasının yolu.

    Returns
    -------
    dict
        Eğitilmiş modeller. Anahtarlar model isimleri (örn. "LSTM", "1D-CNN"),
        değerler BaseTimeSeriesModel alt sınıf örnekleridir.
    """
    config = load_config(config_path)
    models_to_run = config.get("deep_learning_params", {}).get("models_to_run", ["LSTM", "1D-CNN"])
    
    sequence_length = X_train.shape[1]
    num_features = X_train.shape[2]
    
    trained_models = {}
    
    for model_name in models_to_run:
        logging.info(f"--- {model_name} Modeli Başlatılıyor ---")
        if model_name == "LSTM":
            model = LSTMModel(sequence_length, num_features, config_path)
        elif model_name == "1D-CNN":
            model = CNN1DModel(sequence_length, num_features, config_path)
        else:
            logging.warning(f"Bilinmeyen model türü: {model_name}. Atlanıyor.")
            continue
            
        model.build_model()
        model.fit_model(X_train, y_train, X_val, y_val, config_path)
        trained_models[model_name] = model
        logging.info(f"--- {model_name} Eğitimi Tamamlandı ---\n")
        
    return trained_models
