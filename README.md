# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project looks to develop the end-to-end process of a classification problem:
- Load data
- Transform data and prepare date for modelling
- Train algorithm
- Classify testing and training data
- Evaluate new predictions
- Store results and model

The main objective of the classification problem is to predict if a bank customer will churn based on certain characteristics.

The current workflow trains two models: logistic regression and a random forest

## Files and data description

Repository structure
```
.
├── requirements.txt
├── churn_notebook.ipynb
├── churn_library.py
├── churn_script_logging_and_tests.py  (script for running, testing and logging the main script)
├── constants.py  
├── data (file directory to store the input data for the model)
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── barplots
│   │   ├── histograms
│   │   ├── heatmap
│   └── results
│       ├── feature_importance.png
│       ├── Logistic Regression_classification_report.png
│       ├── Random Forest_classification_report.png
│       └── roc_auc.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
└── README.md
```

Main files:
- requirements: Contains the library dependencies that the entire project uses.
- churn_notebook: This is where the solution was prototyped; use it as a reference if needed.
- churn_library.py: Main script where the end to end churn classification algorithm resides, this file generates all the outputs (EDA, model storing, model evaluations, etc.).
- churn_script_logging_and_tests: File to run, test and log the main script.
- constants.py: Here are all the constant values and data types that the main and testing script uses for its inputs. For example, the path for the input. data (```./data/bank_data.csv```)

Input files:
- data: where the source data lives (the code is written to read a .csv file).

Output directories:
- images: Here it is stored all the EDA visualizations (bar charts, histograms and heatmap) and model training and evaluation results.
- logs: It contains an unique file (churn_library.log) that is overwritted every time the churn_script_logging_and_tests.py is ran. Here you can monitor all the pipeline process of its success or failure with the corresponding annotations.
- models: Here are the pickled models stored.


## Running Files
- Make sure you first install (using pip) the dependencies in requirements.txt file using the following statement in your working environment:
```python
    pip install =r requirements.txt
```
- FYI - Dependencies used in this project:
```python
    scikit-learn==0.22       
    joblib==0.11
    pandas==0.23.3
    matplotlib==2.1.0      
    seaborn==0.11.2
    pylint==2.7.4
    autopep8==1.5.6
```
- To run the entire workflow just run the ```churn_library.py``` file.
```python
ipython churn_library.py
```
- If you made any changes to the constants file and/or the main file please use the testing and logging script to ensure the entire workflow runs succesfully.
```python
ipython churn_script_logging_and_tests.py
```


## Special thanks to:
- mohrosidi: https://github.com/mohrosidi/udacity_customer_churn. I look into his solution on HOW to turn function into a class. Starting from there I could take on with the project but with my own approach.
