from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.PrepareModel import PrepareModel
from cnnClassifier import logger

STAGE_NAME = "Prepare model"


class PrepareModelTrainingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        model_config = config.get_model_config()
        model_obj = PrepareModel(config=model_config)
        model_obj.prepare_model()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
