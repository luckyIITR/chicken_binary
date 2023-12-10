from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.callbacks import MyCallback, PrepareCallback
from cnnClassifier.components.training import Training
from cnnClassifier import logger

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        model = training.get_base_model()
        train_df, valid_df, test_df = training.split_csv_data()
        train_gen, valid_gen, test_gen = training.create_train_valid_test_generator(train_df, valid_df, test_df)

        callbacks_config = config.get_callback_config()
        prepare_callbacks = PrepareCallback(config=callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        callback_list = [MyCallback(model, train_gen, callbacks_config)] + callback_list

        training.train(
            callback_list=callback_list
        )


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e