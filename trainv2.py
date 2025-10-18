import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib, os

data = pd.read_csv("data/iris.csv")
X = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
model = DecisionTreeClassifier(max_depth=2, random_state=1).fit(X_train, y_train)

print("Accuracy v2:", metrics.accuracy_score(y_test, model.predict(X_test)))
joblib.dump(model, "artifacts/model_v2.joblib")

