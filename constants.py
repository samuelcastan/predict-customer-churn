'''
Constants used in churn_library.py and churn_script_logging_and_tests.py

Author: Samuel Castan
Date: 2022/12/24
'''

CSV_PATH = 'data/bank_data.csv'

RANDOM_STATE = 42

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

DICT_REPLACE_COL_NAMES = {
    ' ': '',
    '-': '_'
}

RESPONSE = 'Churn'

PREDICTORS = ['Customer_Age',
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
              'Gender_F',
              'Gender_M',
              'Education_Level_College',
              'Education_Level_Doctorate',
              'Education_Level_Graduate',
              'Education_Level_HighSchool',
              'Education_Level_Post_Graduate',
              'Education_Level_Uneducated',
              'Education_Level_Unknown',
              'Marital_Status_Divorced',
              'Marital_Status_Married',
              'Marital_Status_Single',
              'Marital_Status_Unknown',
              'Income_Category_$120K+',
              'Income_Category_$40K_$60K',
              'Income_Category_$60K_$80K',
              'Income_Category_$80K_$120K',
              'Income_Category_Lessthan$40K',
              'Income_Category_Unknown',
              'Card_Category_Blue',
              'Card_Category_Gold',
              'Card_Category_Platinum',
              'Card_Category_Silver']

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
