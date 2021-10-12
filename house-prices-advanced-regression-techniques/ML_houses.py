# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn.impute import SimpleImputer

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

#reading data 
train=pd.read_csv('train.csv')
valid=pd.read_csv('test.csv')


#variables
#X=pd.concat([train,valid])
X_valid = pd.get_dummies(valid)

y=train['SalePrice']
X=train.drop(['SalePrice'],axis=1)
X = pd.get_dummies(X)

#eliminate variables not in valid array
valid_columns=X_valid.columns
train_columns=X_train.columns

final_columns=[]
for column in train_columns:
    if column in valid_columns:
        final_columns.append(column)

#new variables 
X=X.loc[:,final_columns]
X_valid =X_valid.loc[:,final_columns]


#imputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
#y=imp_mean.fit_transform(np.array(y).reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=9,test_size=0.33)


#model 
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_test, y_test)], 
             verbose=False)

y_pred=my_model.predict(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
#print(cross_val_score(knn, X_train, y_train, cv=4))
#print(cross_val_score(my_model, X, y, cv=2))

#subbmission 
y_valid=my_model.predict(X_valid)

X_valid['SalePrice']=y_valid
data=X_valid.loc[:,['Id','SalePrice']]

data.to_csv('submission.csv',index=False)

