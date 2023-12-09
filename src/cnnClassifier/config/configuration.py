from cnnClassifier.utils.common import create_directories, read_yaml
from cnnClassifier.constants import *
from cnnClassifier.entity.config_entity import DataIngestionConfig


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
