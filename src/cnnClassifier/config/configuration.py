from cnnClassifier.utils.common import create_directories, read_yaml
from cnnClassifier.constants import *
from cnnClassifier.entity.config_entity import DataIngestionConfig, BaseModelConfig, CallbacksConfig, TrainingConfig
import os


class ConfigurationManager:
    """
        responsible for creating the Directories needed for data
        and return entity -> DataIngestionConfig
    """

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            data_dir=config.data_dir,
            csv_dir=config.csv_dir
        )

        return data_ingestion_config

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        params_config = self.params

        create_directories([config.base_model_dir])

        base_model_config = BaseModelConfig(
            base_model_dir=Path(config.base_model_dir),
            base_model_path=Path(config.base_model_path),
            img_size=params_config.img_size,
            channels=params_config.channels,
            loss=params_config.loss,
            learning_rate=params_config.learning_rate
        )

        return base_model_config

    def get_callback_config(self) -> CallbacksConfig:
        config = self.config.callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])
        param_config = self.params

        callback_config = CallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
            batch_size=param_config.batch_size,  # set batch size for training
            epochs=param_config.epochs,  # number of all epochs in training
            patience=param_config.patience,  # number of epochs to wait to adjust lr if monitored value does not
            # improve
            stop_patience=param_config.stop_patience,
            # number of epochs to wait before stopping training if monitored value does not improve
            threshold=param_config.threshold,
            # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
            factor=param_config.factor,  # factor to reduce lr by
            ask_epoch=param_config.ask_epoch,  # number of epochs to run before asking if you want to halt training
        )

        return callback_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params

        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chicken-fecal-images")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            base_model_path=Path(training.base_model_path),
            data_dir=Path(training.data_dir),
            csv_dir=Path(training.csv_dir),
            data_gen_path=Path(training.data_gen_path),
            img_size=params.img_size,
            channels=params.channels,
            color=params.color,
            epochs=params.epochs,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            loss=params.loss,
        )

        return training_config
