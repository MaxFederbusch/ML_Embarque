import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

linear = LinearRegression()
linear.fit(X,y)

joblib.dump(linear, "linear.joblib")

logistic = LogisticRegression()
logistic.fit(X,y)

joblib.dump(logistic, "logistic.joblib")