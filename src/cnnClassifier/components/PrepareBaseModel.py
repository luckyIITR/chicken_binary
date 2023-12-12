import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adamax
from cnnClassifier.entity.config_entity import BaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: BaseModelConfig):
        self.model = None
        self.config = config

    def prepare_model(self):
        input_shape = (self.config.img_size[0], self.config.img_size[1], self.config.channels)
        learning_rate = self.config.learning_rate
        loss = self.config.loss

        model = Sequential([
            Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
            # BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            MaxPooling2D(pool_size=2, strides=2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(pool_size=2, strides=2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(Adamax(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
        model.summary()
        self.model = model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)