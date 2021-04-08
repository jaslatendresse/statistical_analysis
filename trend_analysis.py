import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fcdata = pd.read_csv('data/flotation-cell.csv', index_col=0)

#Next, we will select the data we want to adjust:
selected = fcdata.loc[('16/12/2004 20:16:00' < fcdata.index) & (fcdata.index < '16/12/2004 20:25:00'),'Feed rate']

coefficients, residuals, _, _, _ = np.polyfit(range(len(selected.index)),selected,1,full=True)
mse = residuals[0]/(len(selected.index))
nrmse = np.sqrt(mse)/(selected.max() - selected.min())
print('Slope ' + str(coefficients[0]))
print('NRMSE: ' + str(nrmse))

#Slope -1.72979024566 --> negative trend
#NMRSE: 0.274160734073

plt.plot(selected)
plt.plot([coefficients[0]*x + coefficients[1] for x in range(len(selected))])
plt.show()


