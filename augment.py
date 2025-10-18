import pandas as pd, numpy as np
data = pd.read_csv("data/iris.csv")
aug = data.copy()
aug[['sepal_length','sepal_width','petal_length','petal_width']] += np.random.normal(0,0.2,aug[['sepal_length','sepal_width','petal_length','petal_width']].shape)
data_new = pd.concat([data, aug], ignore_index=True)
data_new.to_csv("data/iris.csv", index=False)


