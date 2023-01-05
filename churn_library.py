'''


Author: Samuel Castán
Date: 23 December 2022
'''

# import libraries
import os
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import constants


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
        self.y_train_preds_lr = None
        self.y_test_preds_lr = None
        self.y_train_preds_rf = None
        self.y_test_preds_rf = None

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
        helper function to onek-hot enconde the categorical columns

        input:
            cat_columns: list of categorical columns to one-hot enconde

        output:
            None
        '''

        self.dataframe = pd.get_dummies(self.dataframe, columns=cat_columns)

        return self

    def perform_feature_engineering(self, predictors, response):
        '''
        Normalizes the predictor variables and creates training and testing attributes.

        input:
            predictors: list of colum names used to predict the response variable
            response: string of response variable

        output:
           None
        '''

        self.X = normalize(self.dataframe[predictors])
        self.y = self.dataframe[response]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)

        return self

    def train_models(self):
        '''
        train, store model results: images + scores, and store models

        input:
            None
        output:
            None
        '''

        rfc = RandomForestClassifier(random_state=constants.RANDOM_STATE)
        cv_rfc = GridSearchCV(
            estimator=rfc,
            param_grid=constants.RANDOM_FORREST_GRID,
            cv=5)

        lrc = LogisticRegression(
            solver=constants.LOGISTIC_SOLVER,
            max_iter=constants.LOGISTIC_ITER)

        # Train models
        cv_rfc.fit(self.X_train, self.y_train)
        lrc.fit(self.X_train, self.y_train)

        # Make predicitinos
        self.y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
        self.y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)

        self.y_train_preds_lr = lrc.predict(self.X_train)
        self.y_test_preds_lr = lrc.predict(self.X_test)

        # Store results
        self.classification_report_image()

        # Store models with best estimators
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')

        # Get and store results
        self.classification_report_image()

        return self

    def classification_report_image(self):
        '''
        Saves classification report, ROC curve with its AUC for training and testing data.

    input:
            None
        output:
            None
        '''

        prediction_results = {
            'Random Forest': [
                self.y_train_preds_rf,
                self.y_test_preds_rf],
            'Logistic Regression': [
                self.y_train_preds_lr,
                self.y_test_preds_lr]}

        for model, results in prediction_results.items():

            # Train dataset
            plt.rc('figure', figsize=(5, 5))
            plt.text(
                0.01, 1.25, str(f'{model} Train'), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.05, str(
                    classification_report(
                        self.y_train, results[0])), {
                    'fontsize': 10}, fontproperties='monospace')
            # Test dataset
            plt.text(
                0.01, 0.6, str(f'{model} Test'), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.7, str(
                    classification_report(
                        self.y_test, results[1])), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.savefig(
                f'./images/results/{model}_classification_report.png',
                bbox_inches='tight')
            plt.close()


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

    # Train, predict, store results and save models
    model.train_models()
