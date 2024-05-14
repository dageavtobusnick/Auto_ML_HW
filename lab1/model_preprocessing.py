from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import pandas as pd
import numpy as np

def preproc(data):
  categorical_columns_selector = selector(dtype_include=object)
  categorical_columns = categorical_columns_selector(data)
  data_categorical = data[categorical_columns]
  encoder = OrdinalEncoder().set_output(transform="pandas")
  data_encoded = encoder.fit_transform(data_categorical)
  data[categorical_columns]=data_encoded
  scaler = StandardScaler()
  scaler.fit(data)
  scaled_data=scaler.transform(data)
  return scaled_data

train = pd.read_csv("./train/X.csv")
test = pd.read_csv("./test/X.csv")
train.head()
test.head()
np.savetxt("./train/X_scaled.csv",preproc(train), delimiter=",")
np.savetxt("./test/X_scaled.csv",preproc(test) , delimiter=",")
