import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, recall_score, precision_score, auc, f1_score, plot_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 7
np.random.seed(RANDOM_STATE) 

##enter file path, customer id, and churnoutcome columns here.
filepath = "G:\DAMI\Production\datasets\Telco Data.csv"
customerid = 'customerID'
churnoutcome = 'Churn'
##importing the dataset
churndataset = pd.read_csv(filepath, sep=",", header=0, skipinitialspace=True)
churndf = pd.DataFrame(churndataset)
##replacing all yes with 1, and all No with 0 for processing
churndf.replace(('Yes', 'No'), (1, 0), inplace=True)
##dropping null values, although the data we are working with has none.
churndf = churndf.dropna()

##drop customerid for processing
churndf = churndf.drop(customerid, axis=1)

## pulling categorical features for labelencoding. Decision trees don't really need onehot.
categorical_feature_mask = churndf.dtypes==object
categorical_cols = churndf.columns[categorical_feature_mask].tolist()
churndf[categorical_cols]= churndf[categorical_cols].astype(str)
lechurndf = churndf
le = LabelEncoder()
for i in categorical_cols:
    le.fit(lechurndf[i])
    lechurndf[i] = le.transform(lechurndf[i])


## Splitting the data from the class label
lechurndf_x = lechurndf.drop(churnoutcome, axis=1)
lechurndf_y = lechurndf[churnoutcome]
## splitting the data into train test
x_train, x_test, y_train, y_test = train_test_split(lechurndf_x, lechurndf_y, test_size=0.3, random_state=RANDOM_STATE)

##implementing random tree
param_grid_rf = {'n_estimators': np.arange(10, 2000, 10), 
                'max_features': ['auto','sqrt'], 
                'max_depth': np.arange(10, 2000, 5), 
                'criterion': ['gini', 'entropy'], 
                'bootstrap':[True, False]}
rf = RandomForestClassifier()
rf_random_grid = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf, cv=5, verbose=0)
rf_random_grid.fit(x_train, y_train)
churnpredict = rf_random_grid.predict(x_test)
print(rf_random_grid.best_params_)
print(confusion_matrix(y_test, churnpredict))
print("Accuracy Random Forest: %.2f" % (accuracy_score(y_test, churnpredict)*100) )
print("Recall Random Forest:",recall_score(y_test, churnpredict)*100)
print("Precision Random Forest:",precision_score(y_test, churnpredict)*100)
print("AUC Score :",roc_auc_score(y_test, churnpredict)*100)
print("Classification Report:",classification_report(y_test, churnpredict))