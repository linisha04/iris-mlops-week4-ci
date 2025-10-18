import unittest
import pandas as pd
import joblib
from sklearn import metrics

class TestModelEvaluation(unittest.TestCase):
    def test_model_accuracy(self):
        model = joblib.load("artifacts/model_v1.joblib")
        data = pd.read_csv("data/iris.csv")
        X = data[['sepal_length','sepal_width','petal_length','petal_width']]
        y = data['species']
        preds = model.predict(X)
        acc = metrics.accuracy_score(y, preds)
        self.assertGreaterEqual(acc, 0.8, "Model accuracy too low")

if __name__ == "__main__":
    unittest.main()

