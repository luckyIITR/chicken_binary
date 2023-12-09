from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
import zipfile
import subprocess
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import os


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        url = self.config.source_URL
        filename = self.config.local_data_file
        if not os.path.exists(self.config.local_data_file):
            print(f"Downloading {url} to {filename} using wget...")
            try:
                subprocess.run(['wget', url, '-O', filename], check=True)
                print(f"Download completed: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Download failed with error: {e}")
            logger.info(f"{filename} download!")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
