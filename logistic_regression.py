import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import itertools

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

"""
CORRELATION ANALYSIS
"""
def correlation_analysis():
    #create correlation matrix
    corr_matrix = new_data.corr().abs()

    #Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    to_drop = to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

    new_data.drop(to_drop, axis=1, inplace=True)

"""
DATA IMBALANCE ANALYSIS
"""
def check_imbalance():
    #We check for the class feature, in our case post_bugs
    print(new_data['post_bugs'].value_counts())

"""
DEAL WITH DATA IMBALANCE 
Here, we will oversample the minority class using SMOTE
"""

#Extract the class from dataset
X = new_data.drop('post_bugs', axis=1) #Our features
y = new_data['post_bugs'] #Our class

#Split into train and test data
X, X_test, y, y_test = train_test_split(X, y, test_size=0.4, random_state=RSEED)

#Oversample
oversample = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=RSEED)
X_train, y_train = oversample.fit_resample(X, y)

features = list(X_train.columns)

logistic_regression= LogisticRegression()
lg = logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

train_predictions = lg.predict(X_train)
train_probs = lg.predict_proba(X_train)[:, 1]

predictions = lg.predict(X_test)
probs = lg.predict_proba(X_test)[:, 1]

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(y_test, 
                                     [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test, 
                                      [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(y_test, predictions)
    results['precision'] = precision_score(y_test, predictions)
    results['roc'] = roc_auc_score(y_test, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(y_train, train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves')
    plt.show()


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

evaluate_model(predictions, probs, train_predictions, train_probs)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))



"""
Now if we wanted to do predictions on a new set of files:
new_files = pd.read_csv('new_files')
y_pred = logistic_regression.predict(new_files)
print(new_file)
print(y_pred)
"""

