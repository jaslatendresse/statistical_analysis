import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

RSEED = 50

data = pd.read_csv('data/qt51.csv')

"""
PREPROCESS THE DATA
"""

#Encode categorical features
subsystem_dummies = pd.get_dummies(data.subsystem)

#create new dataframe with the encoded categorical features
new_data = pd.concat([data, subsystem_dummies], axis=1)

#Remove useless features if needed 
new_data = new_data.drop(['comp'], axis=1)
new_data = new_data.drop(['subsystem'], axis=1)

"""
TRANSFORM INTO BINARY CLASSIFICATION
here, we are interested if the file is buggy or not, not the number of bugs
so if post_bugs > 0 --> we encode to 1, else 0
"""
new_data['post_bugs'] = [1 if bugs > 0 else 0 for bugs in new_data.post_bugs]

corr_matrix = new_data.corr().abs()

#Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
new_data.drop(to_drop, axis=1, inplace=True)

X = new_data.drop('post_bugs', axis=1)

# Extracting class 
y = new_data['post_bugs']

X, X_test, y, y_test = train_test_split(X, y, test_size=0.4, random_state=RSEED)

# Handling dataset imbalance with SMOTE - oversample minority class
oversample = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=RSEED)
X_train, y_train = oversample.fit_resample(X, y)

features = list(X_train.columns)

def search_rf():
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    rf_random.fit(X_train, y_train)

    print(rf_random.best_params_)

def search_dt():
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    random_grid = {'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
                }
    
    dt = DecisionTreeClassifier()

    dt_random = RandomizedSearchCV(estimator = dt, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    dt_random.fit(X_train, y_train)

    print(dt_random.best_params_)

search_rf()
#search_dt()