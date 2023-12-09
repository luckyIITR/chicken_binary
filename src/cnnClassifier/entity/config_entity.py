from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    data_dir: Path
    csv_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    model_dir: Path
    model_path: Path
    img_size: list
    channels: int
    batch_size: int  # set batch size for training
    epochs: int  # number of all epochs in training
    patience: int  # number of epochs to wait to adjust lr if monitored value does not improve
    stop_patience: int  # number of epochs to wait before stopping training if monitored value does not improve
    threshold: float  # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
    factor: float  # factor to reduce lr by
    ask_epoch: int  # number of epochs to run before asking if you want to halt training
