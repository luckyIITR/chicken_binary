from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evalution import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        val_config = config.get_evaluation_config()
        evaluation = Evaluation(val_config)
        train_df, valid_df, test_df = evaluation.load_train_valid_test_df()
        evaluation.create_train_valid_test_generator(train_df, valid_df, test_df)
        evaluation.evaluation()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e