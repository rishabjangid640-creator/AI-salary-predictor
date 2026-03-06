import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv("employee_data.csv")

print(data)
encoder = LabelEncoder()

data['skills'] = encoder.fit_transform(data['skills'])
data['education'] = encoder.fit_transform(data['education'])

data = data.drop(['first_name','last_name','emp_id'], axis=1)
X = data.drop('salary', axis=1)
y = data['salary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions)
new_employee = np.array([[28,4,2,1]])

salary = model.predict(new_employee)

print("Predicted Salary:", salary)
