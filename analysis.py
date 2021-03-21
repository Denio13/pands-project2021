#Author: Denis Sarf




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


iris_data = pd.read_csv('iris.data')

iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

#you can specific the number to show here
iris_data.head(10)

plt.show()
plt.savefig('charts/chart1.png')



sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Length"

sns.boxplot(x="species", y="sepal_length", data=iris_data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()
plt.savefig('charts/chart.png')



iris_data.min()
