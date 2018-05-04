import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pandas.tools.plotting import scatter_matrix


dataframe = pd.read_csv("../Dataset/train_all.csv")
ataframe = pd.read_csv("../Dataset/reg_train.csv")



sns.pairplot(dataframe)
plt.show()

sns.pairplot(ataframe)
plt.show()

corr = dataframe.corr()
sns.heatmap(corr, annot=True , linewidths=.5)
plt.show()

sns.heatmap(ataframe.corr(), annot=True , linewidths=.5)
plt.show()