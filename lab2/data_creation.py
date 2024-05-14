from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import pandas as pd
import os

api = KaggleApi()
api.authenticate()
api.dataset_download_files('lovishbansal123/sales-of-a-supermarket',unzip=True)
data = pd.read_csv("./supermarket_sales.csv")
data.head()
X_train,X_test,Y_train,Y_test=train_test_split(data.loc[:, data.columns !='Rating'],data['Rating'],train_size=0.67, random_state=42)
os.makedirs('./train', exist_ok=True)
os.makedirs('./test', exist_ok=True)
X_train.to_csv('./train/X.csv')
Y_train.to_csv('./train/Y.csv')
X_test.to_csv('./test/X.csv')
Y_test.to_csv('./test/Y.csv')
