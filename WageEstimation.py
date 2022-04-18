import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

print(y)
print(X)
