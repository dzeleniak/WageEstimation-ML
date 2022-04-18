import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Data import
data = pd.read_csv("adult_data.csv")
data = pd.DataFrame(data)

# Data separation
y = data.salary
X = data.drop(columns=['salary'])

# CV Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)