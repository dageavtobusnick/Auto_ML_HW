import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump

X=np.loadtxt("./train/X_scaled.csv", delimiter=",")
Y= pd.read_csv("./train/Y.csv")
Y.head()
Y=Y.to_numpy()
reg= LinearRegression()
reg.fit(X,Y)
dump(reg, './model.joblib', compress=9) 
