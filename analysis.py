#!/usr/bin/env python
# coding: utf-8

# # Analysis of Iris dataset
# ***
# *Denis Sarf*
#                                                         
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems" as an example of linear discriminant analysis.
# This famous iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.
# 
# The dataset contains a set of 150 records under 5 attributes:
# 
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm 
# 5. species: 
#       - Iris Versicolour
#       - Iris Setosa 
#       - Iris Virginica

# ![irises.png](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

# ### Importing the libaries for this project: Pandas, Numpy, Matplotlib, Seaborn.
# 
# Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools.
# 
# NumPy is the fundamental package for scientific computing with Python.
# 
# Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
# 
# Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

# In[259]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # 1. Dataset
# Import the **iris.data** using the panda library and examine first few rows of data

# In[260]:


iris_data = pd.read_csv('iris.data')
# setting variable in dataset
iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']
# output the first 10 lines
iris_data.head(10)


# In[261]:


#shape
print(iris_data.shape)


# # 2. Presenting the Summary of  Dataset Statistics

# In[262]:


# Summarize the all Dataset
Summary = iris_data.describe()

#Then the program created a text file and outputs results of this describe command 
with open('Summary_Dataset.txt', 'w') as f: 
    #to the new created text file.
    f.write(str(Summary))#to the new created text file.


# ## 2.1 Summarize the Petal Length

# In[263]:


# Method 1 : creating a text file and saving a summary of petal_length variable using
# describe() command to the same text file.
with open('Summary_petal_length.txt', 'w') as f: 
    f.write(str(petal_length.describe()))


# ## 2.2 Summarize the Petal Width

# In[264]:


# Method 1 :creating a text file and saving a summary  of petal_width variable using
#  describe() command to the same text file.
with open('Summary_petal_width.txt', 'w') as f: 
    f.write(str(petal_width.describe()))


# ## 2.3 Summarize the Sepal Width

# In[265]:


# Method 2:creating a text file and saving a summary  of sepal_width variable using
#  describe() command to the same text file.
sepal_width = iris_data['sepal_width']
sepal_width.describe().to_string('Summary_sepal_width.txt', index = True, header = True)


# ## 2.4 Summarize the Sepal Length

# In[266]:


# Method 2:creating a text file and saving a summary  of sepal_length variable using
#  describe() command to the same text file.
sepal_length = iris_data['sepal_length']
sepal_length.describe().to_string('Summary_sepal_length.txt', index = True, header = True)


#  # 3. Specifications of each variable.

# ## 3.1 Specifications for petal_length variable

# In[285]:


# creating histogram
plt.hist(iris_data['petal_length'], label = 'iris(setosa, versicolor, virginica)', color ='b')
# adding legend
plt.legend()
# defining x-axis
plt.xlabel('size in cm')
# defining y-axis
plt.ylabel('count')
# adding title 
plt.title('Petal_length')
plt.show()
# saving the graphs
plt.savefig('Petal_length.png')
plt.clf()


# ## 3.2 Specifications for petal_width variable

# In[286]:


# creating histogram
plt.hist(iris_data['petal_width'], label = 'iris(setosa, versicolor, virginica)', color ='r')
# adding legend
plt.legend()
# defining x-axis
plt.xlabel('size in cm')
# defining y-axis
plt.ylabel('count')
# adding title 
plt.title('Petal_width')
plt.show()
# saving the graph
plt.savefig('Petal_width.png')
plt.clf()


# ## 3.3 Specifications for sepal_length variable

# In[287]:


# creating histogram
plt.hist(iris_data['sepal_length'], label = 'iris(setosa, versicolor, virginica)', color ='g')
# adding legend
plt.legend()
# defining x-axis
plt.xlabel('size in cm')
# defining y-axis
plt.ylabel('count')
# adding title 
plt.title('Sepal_length')
plt.show()
# saving the graph
plt.savefig('Sepal_length.png') 
plt.clf()


# ## 3.4 Specifications for sepal_width variable

# In[288]:


# creating histogram
plt.hist(iris_data['sepal_width'], label = 'iris(setosa, versicolor, virginica)', color ='c')
# adding legend
plt.legend()
# defining x-axis
plt.xlabel('size in cm')
# defining y-axis
plt.ylabel('count')
# adding title 
plt.title('Sepal_width')
plt.show()
# saving the graph
plt.savefig('Sepal_width.png') 
plt.clf()


# # 4. Scatter Plot of Iris Dataset (Relationship between variables)

# In[281]:


# Scatter plot of the dataset
sns.pairplot(iris_data,hue='species')
plt.show()


# # 4.1 Scatter Plot
# ### The plot shows the relationship between sepal lenght and width of plants

# In[289]:


# use the function regplot to make a scatterplot
sns.set_style('white')
sns.FacetGrid(iris_data,hue='species',height=5).map(plt.scatter,'sepal_length','sepal_width').add_legend()
plt.suptitle('Sepal_length - Sepal_width')
plt.show()
# saving the graph
plt.savefig('Sepal_length-Sepal_width.png')
plt.clf()


# ### The plot shows the relationship between petal lenght and width of plants

# In[290]:


# use the function regplot to make a scatterplot
sns.set_style('white')
sns.FacetGrid(iris_data,hue='species',height=5).map(plt.scatter,'petal_length','petal_width').add_legend()
plt.suptitle('Petal_length - Petal_width')
plt.show()
# saving the graph
plt.savefig('Petal_length-Petal_width.png')
plt.clf()


# ## 4.2 Violin Plot It is used to visualize the distribution of data and its probability distribution.

# In[292]:



# set a grey background 
sns.set(style='darkgrid')

# Sepal length of all species
sns.violinplot(y = iris_data['species'], x = iris_data["sepal_length"])
plt.show()
# saving the graph
plt.savefig('Sepal_length_Violin_plot.png')
plt.clf()
# Sepal width  of all species
sns.violinplot(y = iris_data['species'], x = iris_data["sepal_width"])
plt.show()
# saving the graph
plt.savefig('Sepal_width_Violin_plot.png')
plt.clf()
# Petal length of all species
sns.violinplot(y = iris_data['species'], x = iris_data["petal_length"])
plt.show()
# saving the graph
plt.savefig('Petal_length_Violin_plot.png')
plt.clf()
# Petal width  of all species
sns.violinplot(y = iris_data['species'], x = iris_data["petal_width"])
plt.show()
# saving the graph
plt.savefig('Petal_width_Violin_plot.png')
plt.clf()


# # 5. Correlation
#  The seaborn library allows to draw a correlation matrix through the  *pairplot()*  function.

# In[293]:


# with regression
sns.pairplot(iris_data, kind="reg")
plt.show()
 
# without regression
sns.pairplot(iris_data, kind="scatter")
plt.show()


# # 6. Investigating the data: Min, Max, Mean, Median and Standard Deviation

# In[294]:


#Get the minimum value of all the column in python pandas
iris_data.min()


# In[295]:


#Get the maximum value of all the column in python pandas
iris_data.max()


# In[296]:


#Get the mean value of all the column in python pandas
iris_data.mean()


# In[297]:


#Get the standard deviation value of all the column in python pandas
iris_data.std()


# # 7. Multivariate Plots
# A scatterplot matrix is a matrix associated to n numerical arrays (data variables), X1,X2,…,Xn , of the same length. The cell (i,j) of such a matrix displays the scatter plot of the variable Xi versus Xj.

# In[298]:


#create plot
scatter_matrix(iris_data)
plt.show()


# # 8. Box Plot
# A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution.

# In[299]:


# Plotting the features using boxes
plt.style.use('ggplot')
plt.subplot(2,2,1)
sns.boxplot(x = 'species', y = 'sepal_length', data = iris_data)
plt.subplot(2,2,2)
sns.boxplot(x = 'species', y = 'sepal_width', data = iris_data)
plt.subplot(2,2,3)
sns.boxplot(x = 'species', y = 'petal_length', data = iris_data)
plt.subplot(2,2,4)
sns.boxplot(x = 'species', y = 'petal_width', data = iris_data)


# # 9. lmplot() function in seaborn
# 
# Seaborn’s lmplot is a 2D scatterplot with an optional overlaid regression line. Logistic regression for binary classification is also supported with lmplot . It is intended as a convenient interface to fit regression models across conditional subsets of a dataset.

# In[300]:


# This graph is plotting the species separately
sns.lmplot(x = 'sepal_length', y = 'sepal_width', data = iris_data, hue = 'species', col = 'species')


# # 10. Plot 2D views of the iris dataset

# In[307]:


# The indices of the features that we are plotting
x_index = 0
y_index = 1

# This formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()


# # References
# 
# Background info:
# 
# - https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# - https://archive.ics.uci.edu/ml/datasets/iris
# 
# Docs:
# 
# - https://www.python.org/
# 
# - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# 
# - https://matplotlib.org/stable/gallery/color/named_colors.html
# 
# - http://holoviews.org/gallery/demos/bokeh/boxplot_chart.html
# 
# - https://code.visualstudio.com/docs/python/data-science-tutorial
# 
# Summary values:
# 
# - https://stackoverflow.com/questions/33889310/r-summary-equivalent-in-numpy
# 
# Python iris project:
# 
# - https://rajritvikblog.wordpress.com/2017/06/29/iris-dataset-analysis-python/
# 
# Statistics in Python:
# 
# - http://www.scipy-lectures.org/packages/statistics/index.html#statistics
# 
# - https://aaaanchakure.medium.com/data-visualization-a6dccf643fbb
# 
# Python - IRIS Data visualization and explanation:
# 
# - https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
# 
# - https://www.kaggle.com/biphili/seaborn-matplotlib-iris-data-visualization-code-1
# 
# Iris Data Visualization using Python:
# 
# - https://www.kaggle.com/aschakra/iris-data-visualization-using-python
# 
# 
# Machine Learning Tutorial:
# 
# - https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
# 
# - https://diwashrestha.com/2017/09/18/machine-learning-on-iris/
# 
# - https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html
# 
# - https://www.youtube.com/watch?v=rNHKCKXZde8
# 
# IRIS DATASET ANALYSIS PYTHON | GTHUB:
# 
# - https://github.com/search?q=iris+dataset
# 
# Iris Dataset Analysis (Classification) | Machine Learning | Python:
# 
# - https://www.youtube.com/watch?v=hd1W4CyPX58
# 
# - https://www.youtube.com/watch?v=pTjsr_0YWas&t=66s
# 
# Jupyter Notebook
# 
# - https://jupyter.org/
# 
# - https://sqlbak.com/blog/wp-content/uploads/2020/12/Jupyter-Notebook-Markdown-Cheatsheet2.pdf
# 
