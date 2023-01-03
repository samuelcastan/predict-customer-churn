# library doc string
'''


Author: Samuel Castán
Date: 23 December 2022
'''

# import libraries
import os
import constants
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class Model():
    '''
    Class that encompasess the whole process of reading, preprocessing, training, predicting and evaluating a classification algorithm.
    '''

    def __init__(self):
        self.dataframe = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.Y_pred = None

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


        def encoder_helper(self, category_lst):
            '''
            helper function to turn each categorical column (list provided in constants.py) into a new column with
            propotion of churn for each category - associated with cell 16 from the notebook

            input:
                    self.dataframe: pandas dataframe
                    category_lst: list of columns that contain categorical features

            output:
                    self.dataframe: pandas dataframe with new columns updated
            '''
            
            pass



def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
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


if __name__ == '__main__':

    # Create Model object
    model = Model()
    
    # Import CSV
    model.import_data(constants.CSV_PATH)
    
    # Perform EDA
    model.perform_eda(quant_columns=constants.QUANT_COLUMNS, cat_columns=constants.CAT_COLUMNS)

    # Perform one-hot enconding on categorical variables
    #model.encoder_helper(constants.CAT_COLUMNS)