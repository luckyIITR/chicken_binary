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
class BaseModelConfig:
    base_model_dir: Path
    base_model_path: Path
    img_size: list
    channels: int
    loss: str
    learning_rate: float


@dataclass(frozen=True)
class CallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

    batch_size: int  # set batch size for training
    epochs: int  # number of all epochs in training
    patience: int  # number of epochs to wait to adjust lr if monitored value does not improve
    stop_patience: int  # number of epochs to wait before stopping training if monitored value does not improve
    threshold: float  # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
    factor: float  # factor to reduce lr by
    ask_epoch: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path  # to store the trained model
    base_model_path: Path  # load the base model
    data_dir: Path  # load data and csv
    csv_dir: Path
    img_size: list
    channels: int
    color: str
    epochs: int
    batch_size: int
    data_gen_path: Path  # to save the data generators
    learning_rate: int
    loss: str
