from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

model= load('./model.joblib') 
X=np.loadtxt("./test/X_scaled.csv", delimiter=",")
Y= pd.read_csv("./test/Y.csv")
Y.head()
Y=Y.to_numpy()
approx = model.predict(X)
print("Mean absolute error: ", mean_absolute_error(Y,approx))
