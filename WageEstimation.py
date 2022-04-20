import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# import dataset
data = pd.read_csv("adult_data.csv")
data = pd.DataFrame(data)

# encode independent variable
data['salary'].replace(['<=50K', '>50K'], [0,1], inplace=True)

# drop unnecessary columns
data = data.drop(columns=['education', 'fnlwgt', 'relationship', 'native-country', 'capital-loss'])

# Encode remaining categorical variables
encoding = dict()
for column in data.columns:
    current_encoding = dict()
    if data.dtypes[column]==np.object:
        current_encoding = dict(enumerate(data[column].astype('category').cat.categories))
        data[column] = data[column].astype('category').cat.codes
        current_encoding = dict([(x, y) for y, x in current_encoding.items()])
        encoding[column] = current_encoding

# Create DV and IV dataframes
y = pd.DataFrame(data.salary)
X = pd.DataFrame(data.drop(columns=['salary']))

cv_classifier = KNeighborsClassifier(n_neighbors=20)

scores = cross_val_score(cv_classifier, X, y, cv=10, scoring='accuracy')
print(scores.mean()) #0.8429016142782777



# From the heatmap, we see that occupation, race and workclass have low correlation to living wage
X = X.drop(columns=['occupation'])
scores = cross_val_score(cv_classifier, X, y, cv=10, scoring='accuracy')
print(scores.mean()) #0.8451128284700445


X = X.drop(columns=['race'])
scores = cross_val_score(cv_classifier, X, y, cv=10, scoring='accuracy')
print(scores.mean()) #0.846505142436565


X = X.drop(columns=['workclass'])
scores = cross_val_score(cv_classifier, X, y, cv=10, scoring='accuracy')
print(scores.mean()) #0.84591129558888

