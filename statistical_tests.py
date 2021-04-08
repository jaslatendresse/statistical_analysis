import pandas as pd
import scipy 
import numpy as np
import math
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from scipy.stats import ranksums, wilcoxon

"""
https://docs.scipy.org/doc/scipy/reference/stats.html
"""

data = pd.read_csv('data/qt51.csv')

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




