from cnnClassifier.utils.common import create_directories, read_yaml
from cnnClassifier.constants import *
from cnnClassifier.entity.config_entity import DataIngestionConfig, ModelConfig


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

    def get_model_config(self) -> ModelConfig:
        config = self.config.model
        params_config = self.params

        create_directories([config.model_dir])

        model_config = ModelConfig(
            model_dir=Path(config.model_dir),
            model_path=Path(config.model_path),
            img_size=params_config.img_size,
            channels=params_config.channels,
            batch_size=params_config.batch_size,  # set batch size for training
            epochs=params_config.epochs,  # number of all epochs in training
            patience=params_config.patience,
            # number of epochs to wait to adjust lr if monitored value does not improve
            stop_patience=params_config.stop_patience,
            # number of epochs to wait before stopping training if monitored value does not improve
            threshold=params_config.threshold,
            # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
            factor=params_config.factor,  # factor to reduce lr by
            ask_epoch=params_config.ask_epoch  #
        )

        return model_config
