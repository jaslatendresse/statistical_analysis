# Preparation for SOEN 6591 final exam

You need to submit the following material: 

1. a text file, pdf or a word doc file that contains your answer to each of the questions in your questionnaire. You can put your answers in text, screenshot of the output when you run your analysis or any form that you would like to support your answer to each question. The name of the file should be answer.txt or answer.pdf or answer.doc or answer.docx.

1. For each question, you need to attach a .txt or an .r file names question_X.txt with all your code. For example, for question 1, your text file name is question_1.txt.

## Normal Distribution 
* Normally distributed data
* Mean, median, and mode are all equal
* Use of parametric tests
* Normality test: check if data is normally distributed
  * Kolmogorov-Smirnov
  * Shapiro-Wilk

![Capture d’écran, le 2021-04-08 à 16 37 22](https://user-images.githubusercontent.com/17911957/114093094-b257ba00-9888-11eb-8047-60b6158175c3.png)

Most software engineering data is not normally distributed

## Non-normal Distribution
* Mean can be biased towards the larger or smaller values so it is usually better to use the median

![Capture d’écran, le 2021-04-08 à 16 38 23](https://user-images.githubusercontent.com/17911957/114093226-d74c2d00-9888-11eb-8ea2-bbb4c8ff1675.png)

## Some definitions

**Mean** -- Average value of a distribution

**Median** -- Middle value of a distribution (ordered from small to large) 

**Mode** -- Value that appears the most often in the distribution

**Probability density function** -- If you pick a random point in the distribution, what is the probability of picking that point?

## Significantly Different

Statistical tests to figure out if two distributions are statistically different
* T-test (normal distributions)
* Wilcoxon test (non-normal distributions)

### T-Test

Tells you how significant the difference between distributions are --> tells you if those differences (measures in means) could have happened by chance. 

1. Null hypothesis H0 = distr1 and distr2 are not significantly different
2. Do T-test with software which returns p-value --> the probaility that H0 is true. 

If we get p-value = 1 --> 100% chance the null hypothesis is true. 

We need a threshold: alpha = 0.05
* If p-value < alpha: reject H0, distr1 and distr2 are significantly different
* If p-value > alpha: accept H0, distr1 and distr2 are not significantly different

Example: we have a dataset with the following data and we want to know: does the average age of engineers differ from the rest of the population? 

![Capture d’écran, le 2021-04-08 à 17 20 50](https://user-images.githubusercontent.com/17911957/114098003-c4d4f200-988e-11eb-8ee2-0048c666009b.png)

To solve: 
1. create sample with engineers and calculate mean age
2. create sample with rest of the population and calculate mean age
3. do the t-test

### Wilcoxon Test

Same as T-test but for non-normal distributions. The Wilcoxon test is a nonparametric statistical test that compares two paired groups, and comes in two versions the Rank Sum test or the Signed Rank test. The goal of the test is to determine if two or more sets of pairs are different from one another in a statistically significant manner.

**Wilcoxon = one sample**

Example: 
* Null hypothesis: there is no difference in complexity of files with change_churn <= 100 (s1) and files with change_churn > 100 (s2)
* compute the differences in complexity for both samples, will end up with something like this: 
* diff = [d1, d2, d3, …] and then compute the wilcoxon test. 

**Rank sum = 2 samples**

Example:
* We want to see if the average complexity of files with change_churn <= 100 (s1) is different from the one of files with change_churn > 100 (s2)

## Effect Size
* Between 0 and 1
* How different the distribution are
* Two distributions can have the p-value, but different effect size. 

For normal distribution: cohen’s d

For non-normal: Cliff’s delta https://github.com/neilernst/cliffsDelta 

## Trends

Assume we have points distributed over time. We want to see if there is a significant trend (upwards or downwards)

**Stochastic trends** -  A stochastic system has a random probability distribution or pattern that may be analysed statistically but may not be predicted precisely.
Dickey-Fuller tests → Dickey–Fuller test tests the null hypothesis that a unit root is present in an autoregressive model.

**Deterministic trends** - A deterministic system is a system in which no randomness is involved in the development of future states of the system.
* Cox-Stuart test → identify if trends in data are significant. Works similar with p-value.
* You want to assess whether there is an increasing or decreasing trend 
* To perform the test of Cox-Stuart, the number of observations must be even
* https://www.r-bloggers.com/2009/08/trend-analysis-with-the-cox-stuart-test-in-r/ 

Example: You want to assess whether there is an increasing or decreasing trend of the number of daily customers of a restaurant. We have the number of customers in 15 days:

Customers: 5, 9, 12, 18, 17, 16, 19, 20, 4, 3, 18, 16, 17, 15, 14

In our case we have 15 observations, so we remove the observation at position (N+1)/2 (here the observation with value = 20). Now we have 14 observations, and we can then proceed. 

## Data Preprocessing

To do before applying model to dataset

### Categorical feature encoding
* Most classification models cannot deal with categorical data. We need to encode the categorical data with numerical values. 
  * Pandas dummies
* Sometimes we do not care about the numbers (for example, number of bugs in a file: we just want to know if the file is buggy or not) we just want a binary classification
  * We encode the class to binary values (see decision_tree.py line 31)

### Correlation Analysis
* Correlation affects interpretability of the model
  * Affects feature importance in the classification process
* Check for variables that are correlated and remove one of them (usually keep the one that is easiest to measure)

### Redundancy Analysis
* Checks if a combination of metrics in the data can measure another metric
  * if we find that x1 = x2 + x3, then we can remove x1 because we found that it can be expressed by a combination of x2 and x3. 

### Data Imbalance
* One class is more represented than the other. 
  * For example, we have workflow runs that are classified as “success” or “failure”. Now, if we have 2000 successful runs and 300 failed runs, then our data is imbalanced. 
* Oversample the minority class (creates dummy instances at random)
* Undersample the majority class (remove instances at random)

## Classification

We want to classify something with a label. For example “is this file buggy?” 
* Regression → how many bugs does this file have?
* Ranking → what is the buggiest file?
* Prediction → will this file be buggy? 

### Regression

Given a set of data points, we want to identify the relation between x and y. Essentially, regression is the “best guess” at using a set of data to make some kind of prediction. It’s fitting a set of points to a graph. “Estimate a certain value of x for a given value of y”

Example (linear regression): predict how much snow you think will fall this year. 

y = -2.2923x + 4624.4.

![Capture d’écran, le 2021-04-08 à 17 42 54](https://user-images.githubusercontent.com/17911957/114100136-da97e680-9891-11eb-8692-4306d92b06eb.png)

![Capture d’écran, le 2021-04-08 à 17 43 06](https://user-images.githubusercontent.com/17911957/114100160-e1265e00-9891-11eb-85dc-91d13c3051ce.png)

We want to find a line that minimizes the distance between the line and the data points → minimize the residuals (modelling error)

#### Linear Regression

It is used when we want to predict the value of a variable based on the value of another variable. The variable we want to predict is called the dependent variable (or sometimes, the outcome variable).

#### Logistic Regression

In statistics, the logistic model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc.

Ex: the probability of a file containing a bug. 

When to use logistic regression? https://medium.com/@bemali_61284/random-forest-vs-logistic-regression-16c0c8e2484c 

#### Non-Linear Regression

In statistics, nonlinear regression is a form of regression analysis in which observational data are modeled by a function which is a nonlinear combination of the model parameters and depends on one or more independent variables. The data are fitted by a method of successive approximations.

One example of how nonlinear regression can be used is to predict population growth over time. A scatterplot of changing population data over time shows that there seems to be a relationship between time and population growth, but that it is a nonlinear relationship, requiring the use of a nonlinear regression model.

![Capture d’écran, le 2021-04-08 à 17 45 21](https://user-images.githubusercontent.com/17911957/114100383-319dbb80-9892-11eb-81ea-5c5da59e14ab.png)

### Decision Trees
* Decision tree algorithm falls under the category of supervised learning. They can be used to solve both regression and classification problems.
* Decision tree uses the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree.
* We can represent any boolean function on discrete attributes using the decision tree.

![Capture d’écran, le 2021-04-08 à 17 49 51](https://user-images.githubusercontent.com/17911957/114100792-d28c7680-9892-11eb-9c65-e6f25d23bfe7.png)

### Random Forest

The random forest is a model made up of many decision trees. Rather than just simply averaging the prediction of trees (which we could call a “forest”), this model uses two key concepts that gives it the name random:

1. Random sampling of training data points when building trees
2. Random subsets of features considered when splitting nodes

The random forest combines hundreds or thousands of decision trees, trains each one on a slightly different set of the observations, splitting nodes in each tree considering a limited number of the features. The final predictions of the random forest are made by averaging the predictions of each individual tree.

Example in which we could use decision tree or random forest
* Binary classification → predicting someone's health based on features
  * Is the person healthy or unhealthy? (class label)
  * Age, gender, medical history (features fed to the classifier)

## Model Evaluation

### Confusion Matrix

![Capture d’écran, le 2021-04-08 à 17 53 14](https://user-images.githubusercontent.com/17911957/114101134-4cbcfb00-9893-11eb-9072-c2e87f97c9e3.png)

### Precision
* Always look at the test data precision
* Ratio of the number of true positives divided by the sum of the true positives and false positives.
* Describes how good a model is at predicting the positive class

### Recall
* Always look at the test data recall
* Ratio of the number of true positives divided by the sum of the true positives and false negatives
* Refers to the percentage of total relevant results correctly classified by your algorithm
* Model's ability to find all the data points of interest in a dataset

### ROC
* Always look at the test data ROC
* An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds.

### Accuracy
* How closely the measured value of a quantity corresponds to its “true” value

### Feature Importance
* Feature of classification models that tells you how important a certain feature is in predicting the outcome you want to predict. 


## Regression VS Classification

Fundamentally, classification is about predicting a label and regression is about predicting a quantity.
* Predictive modeling is about the problem of learning a mapping function from inputs to outputs called function approximation.
* Classification is the problem of predicting a discrete class label output for an example.
* Regression is the problem of predicting a continuous quantity output for an example.

https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/#:~:text=Fundamentally%2C%20classification%20is%20about%20predicting,is%20about%20predicting%20a%20quantity.&text=That%20classification%20is%20the%20problem,quantity%20output%20for%20an%20example.















