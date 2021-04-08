import pandas as pd
import numpy as np 
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
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

"""
DECISION TREE 
optimal hyperparams
{'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 50}
"""

dt = DecisionTreeClassifier(random_state=RSEED, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=50)
dt.fit(X_train, y_train)

train_dt_predictions = dt.predict(X_train)
train_dt_probs = dt.predict_proba(X_train)[:, 1]

dt_predictions = dt.predict(X_test)
dt_probs = dt.predict_proba(X_test)[:, 1]

"""
MODEL EVALUATION 
"""

plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

def dt_feature_importance():
    print('Decision Tree - Feature Importance')
    fi = pd.DataFrame({
        'feature': list(X_train.columns),
        'importance': dt.feature_importances_}).\
            sort_values('importance', ascending=False)

    print(fi.head())

def dt_accuracy():
    print(f'DT - Model Accuracy: {dt.score(X_train, y_train)}')

#############################
correlation_analysis()
check_imbalance()

print('\n')

print('Decision Tree Evaluation')
evaluate_model(dt_predictions, dt_probs, train_dt_predictions, train_dt_probs)
plt.savefig('figures/decision_tree_roc_auc_curve.png')

print('\n')

print('Decision Tree Confusion Matrix')
cm = confusion_matrix(y_test, dt_predictions)
plot_confusion_matrix(cm, classes = ['Not buggy', 'Buggy'], title = 'Decision Tree Model - Buggy File Confusion Matrix')
plt.savefig('figures/decision_tree_cm.png')

print('\n')

dt_accuracy()

print('\n')

dt_feature_importance()
#############################

"""
Now if we wanted to do predictions on a new set of files:
new_files = pd.read_csv('new_files')
y_pred = dt.predict(new_files)
print(new_file)
print(y_pred)
"""
