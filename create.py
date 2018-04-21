import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df = pd.read_csv('lyrics.csv',encoding='utf-8')
# print df.head()

sns.countplot(df.genre)
plt.xlabel('Genre')
plt.title('Genre distribution')
plt.show()
