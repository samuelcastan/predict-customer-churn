'''


Author: Samuel Castán
Date: 23 December 2022
'''

# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import constants

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


class Model():
    '''
    Class that encompasess the whole process of reading, preprocessing, training, predicting and
    evaluating a classification algorithm.

    Attributes:
        dataframe:
        X:
        Y:
        X_train:
        Y_train:
        X_test:
        X_test

    Methods:

    '''

    def __init__(self):
        self.dataframe = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth

        input:
            pth: a path to the .csv
        output:
            self.datafame: pandas dataframe
        '''

        df = pd.read_csv(pth)

        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        df = df.iloc[:, 1:]

        self.dataframe = df

        del df

        return self

    def clean_columns(self):
        '''
        Helper function to remove column name whitespaces and add underscores

        input:
            none
        output:
            none
        '''

        column_names = list(self.dataframe.columns)

        column_names = [
            column_name.replace(
                ' ', '').replace(
                '-', '_') for column_name in column_names]

        self.dataframe.columns = column_names

        del column_names

        return self

    def perform_eda(self, quant_columns, cat_columns):
        '''
        performs eda based on self.dataframe stored in object and save plots into ./images/eda

        input:
                self.dataframe: pandas dataframe
                quant_columns: list containing the quantitative variables
                cat_columns: list containing the qualitative variables

        output:
                None
        '''

        for quant_column in quant_columns:
            plt.figure(figsize=(20, 10))
            sns.histplot(data=self.dataframe, x=quant_column)
            plt.title(f'Histogram for {quant_column}')
            plt.savefig(f"./images/eda/histograms/{quant_column}.png")
            plt.close()

        for cat_column in cat_columns:
            plt.figure(figsize=(20, 10))
            self.dataframe[cat_column].value_counts(
                'normalize').plot(kind='bar')
            plt.title(f'Distribution of proportions for {cat_column}')
            plt.savefig(f"./images/eda/barplots/{cat_column}.png")
            plt.close()

        plt.figure(figsize=(20, 10))
        sns.heatmap(
            self.dataframe.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.title('Lineal correlation heatmap')
        plt.savefig('./images/eda/heatmap/heatmap.png')
        plt.close()

    def encoder_helper(self, cat_columns):
        '''
        helper function to turn each categorical column (list provided in constants.py) into a new column with
        propotion of churn for each category - associated with cell 16 from the notebook

        input:
                self.dataframe: pandas dataframe
                cat_columns: list of columns that contain categorical features to apply one-hot enconding

        output:
                self.dataframe: pandas dataframe with new columns updated
        '''

        self.dataframe = pd.get_dummies(self.dataframe, columns=cat_columns)

        return self

    def perform_feature_engineering(self, predictors, response):
        '''
        Normalizes the predictor variables and creates training and testing attributes.

        input:
            self.dataframe: pandas dataframe
            predictors: list of colum names used to predict the response variable
            response: string of response variable

        output:
            self.X: All attributes instances
            self.y: All response variables instances
            self.X_train: X training data
            self.X_test: X testing data
            self.y_train: y training data
            self.y_test: y testing data
        '''

        self.X = normalize(self.dataframe[predictors])
        self.y = self.dataframe[response]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)

        return self


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


if __name__ == '__main__':

    # Create Model object
    model = Model()

    # Import CSV
    model.import_data(constants.CSV_PATH)

    # Perform EDA
    model.perform_eda(
        quant_columns=constants.QUANT_COLUMNS,
        cat_columns=constants.CAT_COLUMNS)

    # Perform one-hot enconding on categorical variables
    model.encoder_helper(cat_columns=constants.CAT_COLUMNS)

    # Remove whitespaces and add underscores to new columns
    model.clean_columns()

    # Perform feature engineering
    model.perform_feature_engineering(
        predictors=constants.PREDICTORS,
        response=constants.RESPONSE)
