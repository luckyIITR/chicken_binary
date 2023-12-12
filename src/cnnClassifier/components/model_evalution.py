import tensorflow as tf
import pickle
from keras.preprocessing.image import ImageDataGenerator
from cnnClassifier.entity.config_entity import EvaluationConfig
import os
from pathlib import Path
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.ts_length = None

    def load_train_valid_test_df(self):
        # load generators
        path = os.path.join(self.config.dataframe_path, 'train_df.pkl')
        with open(path, 'rb') as f:
            train_df = pickle.load(f)

        path = os.path.join(self.config.dataframe_path, 'valid_df.pkl')
        with open(path, 'rb') as f:
            valid_df = pickle.load(f)

        path = os.path.join(self.config.dataframe_path, 'test_df.pkl')
        with open(path, 'rb') as f:
            test_df = pickle.load(f)

        self.ts_length = len(test_df)

        return train_df, valid_df, test_df

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
        test_batch_size = self.config.batch_size
        test_steps = ts_length // test_batch_size

        # This function which will be used in image data generator for data augmentation, it just take the image and
        # return it again.
        def scalar(img):
            return img

        tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True, rescale = 1./255)
        ts_gen = ImageDataGenerator(preprocessing_function=scalar, rescale = 1./255)

        train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode="binary",
                                               color_mode=color, shuffle=False, batch_size=batch_size)
        valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                               class_mode="binary",
                                               color_mode=color, shuffle=False, batch_size=batch_size)
        # Note: we will use custom test_batch_size, and make shuffle= false
        test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                              class_mode="binary",
                                              color_mode=color, shuffle=False, batch_size=test_batch_size)
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.test_gen = test_gen

        return train_gen, valid_gen, test_gen

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(self.config.trained_model_path)

        ts_length = self.ts_length
        test_batch_size = self.config.batch_size
        test_steps = ts_length // test_batch_size

        train_score = model.evaluate(self.train_gen, steps=test_steps, verbose=1)
        valid_score = model.evaluate(self.valid_gen, steps=test_steps, verbose=1)
        test_score = model.evaluate(self.test_gen, steps=test_steps, verbose=1)

        print("Train Loss: ", train_score[0])
        print("Train Accuracy: ", train_score[1])
        print('-' * 20)
        print("Validation Loss: ", valid_score[0])
        print("Validation Accuracy: ", valid_score[1])
        print('-' * 20)
        print("Test Loss: ", test_score[0])
        print("Test Accuracy: ", test_score[1])

        scores = {"Train Loss": train_score[0], "Train Accuracy:": train_score[1],
                  "Validation Loss": valid_score[0], "Validation Accuracy:": valid_score[1],
                  "Test Loss": test_score[0], "Test Accuracy:": test_score[1]}

        save_json(path=Path("scores.json"), data=scores)