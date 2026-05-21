"""
Derin Öğrenme Modelleri (Deep Learning Models) modülü.
Bu modül TensorFlow/Keras importlarını ve tüm derin öğrenme modelleri
(LSTM, 1D-CNN vb.) için ortak olan temel sınıf (base class) yapısını içerir.
"""

import logging
import numpy as np

# Keras ve TensorFlow importları
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

