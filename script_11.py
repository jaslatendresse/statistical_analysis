import pandas as pd
import scipy 
import numpy as np
import math
from scipy.stats import ranksums, wilcoxon

data = pd.read_csv('qt51.csv')

def get_data_info():
    #names(data)
    for col in data.columns:
        print(col)

    #summary(data)
    print(data.describe())

    #median
    print(data[['size']].median())

    #mean
    print(data[['size']].mean())

#Wilcoxon test - 1 sample
def wilcoxon_test():
    """
    Null hypothesis: there is no difference in complexity of files with change_churn <= 100 (s1) and files with 
    change_churn > 100 (s2)

    First, compute the difference between the complexity and then compute the test. 
    """
    s1 = data.loc[data['change_churn'] <= 100, 'complexity']
    s2 = data.loc[data['change_churn'] > 100, 'complexity']

    diff = s1 - s2

    w, p = wilcoxon(diff)
    print(w,p)

#Rank sum test - 2 samples
def ranksum_test():
    """
    We want to see if the average complexity of files with change_churn <= 100 (s1) is different from the one
    of files with change_churn > 100 (s2)

    Null hypothesis: s1 and s2 are not significantly different.
    """

    sample1 = data.loc[data['change_churn'] <= 100, 'complexity']
    sample2 = data.loc[data['change_churn'] > 100, 'complexity']

    s, pvalue = ranksums(sample1, sample2)

    print(s,pvalue)

#Cliff's delta https://github.com/neilernst/cliffsDelta 

#Dealing with categorical data
def encode():
    #get the categorical column - in our case 'subsystem'
    subsystem_dummies = pd.get_dummies(data.subsystem)

    #create new dataframe with the encoded categorical features
    new_data = pd.concat([data, subsystem_dummies], axis=1)
"""
We want to predict if a file will be buggy or not. 
The class label for this is "post_bugs" 

https://docs.scipy.org/doc/scipy/reference/stats.html

"""