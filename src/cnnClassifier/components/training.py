from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
import keras
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adamax
import pickle
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.history = None
        self.config = config
        self.model = None
        self.train_gen = None
        self.valid_gen = None
        self.test_gen = None

    def get_base_model(self):
        self.model = keras.models.load_model(
            self.config.base_model_path
        )
        self.model.compile(Adamax(learning_rate=self.config.learning_rate), loss=self.config.loss, metrics=['accuracy'])
        return self.model

    def split_csv_data(self):
        df = pd.read_csv(self.config.csv_dir)
        # change column name
        df.columns = ["filepaths", "labels"]
        df['filepaths'] = df['filepaths'].apply(lambda x: os.path.join(self.config.data_dir, x))

        # only extracting two types of images
        filter = (df['labels'] == 'Coccidiosis') | (df['labels'] == 'Healthy')
        df = df[filter]

        # Create train df
        strat = df['labels']
        train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=strat)

        # valid and test dataframe
        strat = dummy_df['labels']
        valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=strat)

        self.save_train_valid_test_df(train_df, valid_df, test_df)

        return train_df, valid_df, test_df

    def save_train_valid_test_df(self, train_df, valid_df, test_df):
        # Save generators
        path = os.path.join(self.config.data_gen_path, 'train_df.pkl')
        with open(path, 'wb') as f:
            pickle.dump(train_df, f)

        path = os.path.join(self.config.data_gen_path, 'valid_df.pkl')
        with open(path, 'wb') as f:
            pickle.dump(valid_df, f)

        path = os.path.join(self.config.data_gen_path, 'test_df.pkl')
        with open(path, 'wb') as f:
            pickle.dump(test_df, f)

    def create_train_valid_test_generator(self, train_df, valid_df, test_df):
        '''
        This function takes train, validation, and test dataframe and fit them into image data generator, because model takes data from image data generator.
        Image data generator converts images into tensors.
        '''

        # define model parameters
        batch_size = self.config.batch_size
        img_size = self.config.img_size
        channels = self.config.channels
        color = self.config.color
        img_shape = (img_size[0], img_size[1], channels)

        ts_length = len(test_df)
        test_batch_size = max(
            sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
        test_steps = ts_length // test_batch_size

        # This function which will be used in image data generator for data augmentation, it just take the image and
        # return it again.
        def scalar(img):
            return img

        tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True, rescale = 1./255)
        ts_gen = ImageDataGenerator(preprocessing_function=scalar, rescale = 1./255)

        train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode="binary",
                                               color_mode=color, shuffle=True, batch_size=batch_size)
        valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode="binary",
                                               color_mode=color, shuffle=True, batch_size=batch_size)
        # Note: we will use custom test_batch_size, and make shuffle= false
        test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                              class_mode="binary",
                                              color_mode=color, shuffle=False, batch_size=test_batch_size)
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.test_gen = test_gen

        return train_gen, valid_gen, test_gen

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.history = self.model.fit(x=self.train_gen, epochs=self.config.epochs, verbose=0, callbacks=callback_list,
                                      validation_data=self.valid_gen, validation_steps=None, shuffle=False)

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
