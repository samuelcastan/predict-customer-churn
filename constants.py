'''
Constants used in churn_library.py

Author: Samuel Castan
Date: 2022/12/24
'''

CSV_PATH = 'data/bank_data.csv'

RANDOM_STATE = 42

DICT_REPLACE_COL_NAMES = {
    ' ':'',
    '-':'_'
}

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

KEEP_COLS = ['Customer_Age',
             'Dependent_count',
             'Months_on_book',
             'Total_Relationship_Count',
             'Months_Inactive_12_mon',
             'Contacts_Count_12_mon',
             'Credit_Limit',
             'Total_Revolving_Bal',
             'Avg_Open_To_Buy',
             'Total_Amt_Chng_Q4_Q1',
             'Total_Trans_Amt',
             'Total_Trans_Ct',
             'Total_Ct_Chng_Q4_Q1', 
             'Avg_Utilization_Ratio',
             'Gender_Churn',
             'Education_Level_Churn',
             'Marital_Status_Churn',
             'Income_Category_Churn',
             'Card_Category_Churn']

TEST_SIZE = 0.30

LOGISTIC_SOLVER = 'lbfgs'

LOGISTIC_ITER = 3000

CROSS_VALIDATION = 5

RANDOM_FORREST_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
