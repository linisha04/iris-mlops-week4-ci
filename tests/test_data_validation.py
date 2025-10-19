import pandas as pd
import unittest
#comment
class TestDataValidation(unittest.TestCase):
    def test_data_shape(self):
        df = pd.read_csv("data/iris.csv")
        self.assertEqual(df.shape[1], 5, "Data must have 5 columns")
    
    def test_no_missing_values(self):
        df = pd.read_csv("data/iris.csv")
        self.assertFalse(df.isnull().any().any(), "Data contains missing values")

if __name__ == '__main__':
    unittest.main()

