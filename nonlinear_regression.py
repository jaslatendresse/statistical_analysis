import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

data = pd.read_csv('data/position_salaries.csv')

data.drop(['position'], axis=1)

x = np.array(data['level'])
y = np.array(data['salary'])
degree = 3

df = pd.DataFrame(columns=['y', 'x'])
df['x'] = x
df['y'] = y

weights = np.polyfit(x, y, degree)
model = np.poly1d(weights)
results = smf.ols(formula='y ~ model(x)', data=df).fit()

print(results.summary())