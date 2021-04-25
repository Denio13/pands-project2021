#  Programming and scripting project 2021 
*pands-project2021*
*Denis Sarf*
***
# **Analysis of Iris dataset**

## Table of contents
* [1. Task](#1-task)
* [2. Iris dataset](#2-iris-dataset)
  * [2.1 Iris dataset history](#21-iris-dataset-history)
  * [2.2 Importing the libaries for this project: Pandas, Numpy, Matplotlib, Seaborn.](#22-importing-the-libaries-for-this-project-pandas-numpy-matplotlib-seaborn)
* [3. Dataset](#3-dataset)
* [4. Presenting the Summary of Dataset Statistics](#4-presenting-the-summary-of--dataset-statistics)
  * [4.1 Summarize the Petal Length](#41-summarize-the-petal-length)
  * [4.2 Summarize the Petal Width](#42-summarize-the-petal-width)
  * [4.3 Summarize the Sepal Width](#43-summarize-the-sepal-width)
  * [4.4 Summarize the Sepal Length](#44-summarize-the-sepal-length)
* [5. Specifications of each variable.](#5-specifications-of-each-variable.)
  * [5.1 Specifications for petal_length variable](#51-specifications-for-petal_length-variable)
  * [5.2 Specifications for petal_width variable](#52-specifications-for-petal_width-variable)
  * [5.3 Specifications for sepal_length variable](#53-specifications-for-sepal_length-variable)
  * [5.4 Specifications for sepal_width variable](#54-specifications-for-sepal_width-variable)
* [6. Scatter Plot of Iris Dataset (Relationship between variables)](#6-scatter-plot-of-iris-dataset-relationship-between-variables))
  * [6.1 Scatter Plot](#61-scatter-plot)
  * [6.2 Violin Plot It is used to visualize the distribution of data and its probability distribution](#62-violin-plot-it-is-used-to-visualize-the-distribution-of-data-and-its-probability-distribution)
* [7. Correlation](#7-correlation)
* [8. Investigating the data: Min, Max, Mean, Median and Standard Deviation](#8-investigating-the-data-min-max-mean-median-and-standard-deviation)
* [9. Multivariate Plots](#9-multivariate-plots)
* [10. Box Plot](#10-box-plot)
* [11. lmplot() function in seaborn](#11-implot-function-in-seaborn)
* [12. Plot 2D views of the iris dataset](#12-plot-2d-views-of-the-iris-dataset)  
* [13. References](#13-references)
  * [13.1 Background info](#131-background-info)
  * [13.2 Documentation](#132-documentation)
  * [Dataset analysis approach by others](#dataset-analysis-approach-by-others)
  * [13.1 Background info](#131-background-info)
  * [13.2 Documentation](#132-documentation)
  * [13.3 Summary values](#133-summary-values)
  * [13.4 Iris Data Visualization using Python](#134-iris-data-visualization-using-python)
  * [13.5 Machine Learning Tutorial](#135-machine-learning-tutorial)
  * [13.6 Iris Dataset Analysis (Classification) | Machine Learning | Python](#136-iris-dataset-analysis-classification-machine-learning-python)
  * [13.7 Jupyter Notebook](#137-jupyter-notebook)


## **1. Task**

Detailed project description can be found on [GitHub](https://github.com/Denio13/pands-project2021/blob/main/Project%202021.pdf)

## **2. Iris dataset**

### **2.1 Iris dataset history**
                                                        
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper [“The Use of Multiple Measurements in Taxonomic Problems”](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x) as an example of linear discriminant analysis.
This famous iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

The dataset contains a set of 150 records under 5 attributes:

1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. species: 
      - Iris Versicolour
      - Iris Setosa 
      - Iris Virginica

![irises.png](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)

### **2.2 Importing the libaries for this project: Pandas, Numpy, Matplotlib, Seaborn.**

Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools.

NumPy is the fundamental package for scientific computing with Python.

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## **3. Dataset**
Import the **iris.data** using the panda library and examine first few rows of data


```python
iris_data = pd.read_csv('iris.data')
# setting variable in dataset
iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']
# output the first 10 lines
iris_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5.4</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
#shape
print(iris_data.shape)
```

    (149, 5)
    

## **4. Presenting the Summary of  Dataset Statistics**


```python
# Summarize the all Dataset
Summary = iris_data.describe()

#Then the program created a text file and outputs results of this describe command 
with open('Summary_Dataset.txt', 'w') as f: 
    #to the new created text file.
    f.write(str(Summary))#to the new created text file.
```

### **4.1 Summarize the Petal Length**


```python
# Method 1 : creating a text file and saving a summary of petal_length variable using
# describe() command to the same text file.
with open('Summary_petal_length.txt', 'w') as f: 
    f.write(str(petal_length.describe()))
```

### **4.2 Summarize the Petal Width**


```python
# Method 1 :creating a text file and saving a summary  of petal_width variable using
#  describe() command to the same text file.
with open('Summary_petal_width.txt', 'w') as f: 
    f.write(str(petal_width.describe()))
```

### **4.3 Summarize the Sepal Width**


```python
# Method 2:creating a text file and saving a summary  of sepal_width variable using
#  describe() command to the same text file.
sepal_width = iris_data['sepal_width']
sepal_width.describe().to_string('Summary_sepal_width.txt', index = True, header = True)
```

### **4.4 Summarize the Sepal Length**

```python
# Method 2:creating a text file and saving a summary  of sepal_length variable using
#  describe() command to the same text file.
sepal_length = iris_data['sepal_length']
sepal_length.describe().to_string('Summary_sepal_length.txt', index = True, header = True)
```

## **5. Specifications of each variable.**

### **5.1 Specifications for petal_length variable**


```python
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
```


    
![Petal_length](charts/Petal_length.png)
    


### **5.2 Specifications for petal_width variable**


```python
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
```


    
![Petal_width](charts/Petal_width.png)


### **5.3 Specifications for sepal_length variable**


```python
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
```


    
![Sepal_length](charts/Sepal_length.png)


### **5.4 Specifications for sepal_width variable**


```python
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
```


    
![Sepal_width](charts/Sepal_width.png)


## **6. Scatter Plot of Iris Dataset (Relationship between variables)**


```python
# Scatter plot of the dataset
sns.pairplot(iris_data,hue='species')
plt.show()
```


![Scatter_Plot](charts/Scatter_Plot.png)




### **6.1 Scatter Plot**
### The plot shows the relationship between sepal lenght and width of plants


```python
# use the function regplot to make a scatterplot
sns.set_style('white')
sns.FacetGrid(iris_data,hue='species',height=5).map(plt.scatter,'sepal_length','sepal_width').add_legend()
plt.suptitle('Sepal_length - Sepal_width')
plt.show()
# saving the graph
plt.savefig('Sepal_length-Sepal_width.png')
plt.clf()
```


![Sepal_length-Sepal_width](charts/Sepal_length-Sepal_width.png)


### The plot shows the relationship between petal lenght and width of plants


```python
# use the function regplot to make a scatterplot
sns.set_style('white')
sns.FacetGrid(iris_data,hue='species',height=5).map(plt.scatter,'petal_length','petal_width').add_legend()
plt.suptitle('Petal_length - Petal_width')
plt.show()
# saving the graph
plt.savefig('Petal_length-Petal_width.png')
plt.clf()
```


![Petal_length-Petal_width](charts/Petal_length-Petal_width.png)


### **6.2 Violin Plot It is used to visualize the distribution of data and its probability distribution**


```python

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
```


    
![Sepal_length_Violin_plot](charts/Sepal_length_Violin_plot.png)
    



    
![Sepal_width_Violin_plot](charts/Sepal_width_Violin_plot.png)
    



    
![Petal_length_Violin_plot](charts/Petal_length_Violin_plot.png)
    



    
![Petal_width_Violin_plot](charts/Petal_width_Violin_plot.png)
    




## **7. Correlation**
The seaborn library allows to draw a correlation matrix through the  *pairplot()*  function.


```python
# with regression
sns.pairplot(iris_data, kind="reg")
plt.show()
 
# without regression
sns.pairplot(iris_data, kind="scatter")
plt.show()
```


    
![Correlation_with_regression](charts/Correlation_with_regression.png)
    



    
![Correlation_without_regression](charts/Correlation_without_regression.png)
    


## **8. Investigating the data: Min, Max, Mean, Median and Standard Deviation**


```python
#Get the minimum value of all the column in python pandas
iris_data.min()
```




    sepal_length            4.3
    sepal_width               2
    petal_length              1
    petal_width             0.1
    species         Iris-setosa
    dtype: object




```python
#Get the maximum value of all the column in python pandas
iris_data.max()
```




    sepal_length               7.9
    sepal_width                4.4
    petal_length               6.9
    petal_width                2.5
    species         Iris-virginica
    dtype: object




```python
#Get the mean value of all the column in python pandas
iris_data.mean()
```




    sepal_length    5.848322
    sepal_width     3.051007
    petal_length    3.774497
    petal_width     1.205369
    dtype: float64




```python
#Get the standard deviation value of all the column in python pandas
iris_data.std()
```




    sepal_length    0.828594
    sepal_width     0.433499
    petal_length    1.759651
    petal_width     0.761292
    dtype: float64



## **9. Multivariate Plots**
A scatterplot matrix is a matrix associated to n numerical arrays (data variables), X1,X2,…,Xn , of the same length. The cell (i,j) of such a matrix displays the scatter plot of the variable Xi versus Xj.


```python
#create plot
scatter_matrix(iris_data)
plt.show()
```

![Multivariate_Plots](charts/Multivariate_Plots.png)
    


## **10. Box Plot**
A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution.


```python
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
```


![Box_Plot](charts/Box_Plot.png)
    


## **11. lmplot() function in seaborn**

Seaborn’s lmplot is a 2D scatterplot with an optional overlaid regression line. Logistic regression for binary classification is also supported with lmplot . It is intended as a convenient interface to fit regression models across conditional subsets of a dataset.


```python
# This graph is plotting the species separately
sns.lmplot(x = 'sepal_length', y = 'sepal_width', data = iris_data, hue = 'species', col = 'species')
```



![lmplot_function](charts/lmplot_function.png)
    



## **12. Plot 2D views of the iris dataset**


```python
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

```


    
![Plot_2D](charts/Plot_2D.png)
    


## **13. References**

### **13.1 Background info**

- [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [The Iris Dataset — A Little Bit of History and Biology](https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5)

- [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)

### **13.2 Documentation**

- [Python Documentation](https://www.python.org/) 
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) 

- [Matplotlib Documentation](https://matplotlib.org/stable/gallery/color/named_colors.html)

- [Boxplot chart](http://holoviews.org/gallery/demos/bokeh/boxplot_chart.html) 

- [Visual Studio Code - Python](https://code.visualstudio.com/docs/python/data-science-tutorial)

### **13.3 Summary values**

- [Summary() equivalent in numpy](https://stackoverflow.com/questions/33889310/r-summary-equivalent-in-numpy) 

- [IRIS DATASET ANALYSIS (PYTHON)](https://rajritvikblog.wordpress.com/2017/06/29/iris-dataset-analysis-python/)

- [Statistics in Python](http://www.scipy-lectures.org/packages/statistics/index.html#statistics) 

- [Data Visualization using matplotlib and seaborn](https://aaaanchakure.medium.com/data-visualization-a6dccf643fbb) 

- [Python - IRIS Data visualization and explanation](https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation) 

- [Seaborn Matplotlib Iris Data Visualization](https://www.kaggle.com/biphili/seaborn-matplotlib-iris-data-visualization-code-1) 

### **13.4 Iris Data Visualization using Python**

- [Iris Data Visualization using Python](https://www.kaggle.com/aschakra/iris-data-visualization-using-python)  

### **13.5 Machine Learning Tutorial**

- [A Complete Guide to K-Nearest-Neighbors with Applications in Python and R](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/)  

- [Plot 2D views of the iris dataset](https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html)   

- [Scikit Learn - Iris Dataset](https://www.youtube.com/watch?v=rNHKCKXZde8)  

### **13.6 Iris Dataset Analysis (Classification) | Machine Learning | Python**

- [Getting started in scikit-learn with the famous iris dataset](https://www.youtube.com/watch?v=hd1W4CyPX58)  

### **13.7 Jupyter Notebook**

- [Jupyter Notebook](https://jupyter.org/)  

- [Jupyter Notebook Markdown Cheatsheet](https://sqlbak.com/blog/wp-content/uploads/2020/12/Jupyter-Notebook-Markdown-Cheatsheet2.pdf)  


