'''
Module for testing the churn_library script.

Author: Samuel Castán
Date: January 8, 2023
'''


import logging
import churn_library as cls
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        datafame = import_data(constants.CSV_PATH)
        logging.info('SUCCESS - Testing import_data')
    except FileNotFoundError:
        logging.error("ERROR - Testing import_data: The file wasn't found")

    try:
        assert datafame.dataframe.shape[0] > 0
        assert datafame.dataframe.shape[1] > 0
    except AssertionError:
        logging.error(
            "ERROR - Testing import_data: The file doesn't appear to have rows and columns")


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        assert isinstance(constants.QUANT_COLUMNS, list)
        assert isinstance(constants.CAT_COLUMNS, list)
        perform_eda(
            quant_columns=constants.QUANT_COLUMNS,
            cat_columns=constants.CAT_COLUMNS)
        logging.info("SUCCESS - Testing perform_eda")

    except AssertionError:
        logging.error(
            "ERROR - Testing perform_eda: Method inputs must be list type")

    except KeyError:
        logging.info(
            "ERROR - Testing perform_eda: Column names doesn't match dataframe column names")

    except FileNotFoundError:
        logging.info(
            'ERROR - Testing perform_eda: Destination directory path does not exist')


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''

    try:
        assert isinstance(constants.CAT_COLUMNS, list)
        encoder_helper(cat_columns=constants.CAT_COLUMNS)

        logging.info("SUCCESS - Testing encoder_helper")

    except AssertionError:
        logging.error(
            "ERROR - Testing encoder_helper: Method input must be list type")

    except KeyError:
        logging.info(
            "ERROR - Testing encoder_helper: Column names doesn't match dataframe column names")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        assert isinstance(constants.RESPONSE, str)
        perform_feature_engineering(
            predictors=constants.PREDICTORS,
            response=constants.RESPONSE)
        logging.info('SUCCESS - Testing perform_feature_engineering')

    except AssertionError:
        logging.error(
            'ERROR - Testing perform_feature_engienering: RESPONSE must be a string')

    except KeyError:
        logging.error(
            'ERROR - Testing perform_feature_engineering: Column(s) name(s) does not \
                match with dataframe')

def test_train_models(train_models):
    '''
    test train_models
    '''

    try:
        train_models()
        logging.info('SUCCESS - Testing train_models')
        logging.info('SUCCESS - PROCESS FINISHED SUCCESFULLY')
    except FileNotFoundError:
        logging.info(
            'ERROR - Testing train_models: Destination directory path to save \
                results does not exist')


if __name__ == "__main__":

    # Create Model object for testing
    model = cls.Model()
    # Test import_data
    test_import(model.import_data)
    # Test perform_eda
    test_eda(model.perform_eda)
    # Test encoder_helper
    test_encoder_helper(model.encoder_helper)
    # Test perform_feature_engineering
    test_perform_feature_engineering(model.perform_feature_engineering)
    # Test model training, prediction, storing results & models
    test_train_models(model.train_models)
