import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../output.csv', index_col=['instant'])
sns.distplot(df['cnt'])
plt.show()
