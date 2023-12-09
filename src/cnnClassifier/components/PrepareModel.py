import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adamax
from cnnClassifier.entity.config_entity import ModelConfig
from pathlib import Path


class PrepareModel:
    def __init__(self, config: ModelConfig):
        self.model = None
        self.config = config

    def prepare_model(self):
        model = Sequential([
            Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[224, 224, 3]),
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            MaxPooling2D(pool_size=2, strides=2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(pool_size=2, strides=2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model
        self.save_model(path=self.config.model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
