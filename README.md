# WageEstimation-ML
Machine Learning project for estimating whether or not a person with a given set of features is being paid a fair living wage.

## Dataset
The following dataset will be used for this project: https://www.kaggle.com/datasets/ddmasterdon/income-adult

## Approach
K-Nearest-Neighbors is used to classify the fairness of wage based on this dataset. To increase the performance and accuracy of the model, it is tested using multiple subsets of features and using varying numbers of neighbors. To ensure there is no bias, cross validation is used to split the dataset into training and testing sets.

## Features
15 features are given in the dataset:
- Age (int)
- Workclass (int-enum)
- FNLWGT - (int) Final Wage Total
- Education (str) 
- Education Num - (int) scale 1-16
- Marital Status (str-enum) 
- Occupation (str) 
- Relationship (str-enum) 
- Race (str-enum)
- Sex (str-enum)
- Capital Gain (int) 
- Capital Loss (int) 
- Hours Per Week (int)
- Native country (str)
- Salary >=50k (bool)
