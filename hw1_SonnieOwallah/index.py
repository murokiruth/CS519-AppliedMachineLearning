## Importing pandas
import pandas as pd

## Importing matplotlib
import matplotlib.pyplot as plt

## Reading Iris dataset
data = pd.read_csv("iris.data", header=None)
print(data.head())
print(data.columns)

## Calculating and printing number or rows and columns
data.shape
print(f"Number of rows : {data.shape[0]}")
print(f"Number of columns : {data.shape[1]}")

## Getting all the values of the last column
lastColumn = data.iloc[:,-1]
print(lastColumn)

## Printing the distinct values of the last column
lastColumnDistinct = lastColumn.unique()
print(f"The distict values of the last column are : {lastColumnDistinct}")

## Rows where the last column has "Iris-setosa"
irisSetosa = data[data.iloc[:,-1] == "Iris-setosa"]

## Where the last column has "Iris-setosa", Calculating the no. of rows,the avg value of the 1st column, the max value of the 2nd column, & the min value of the 3rd column 
print(f"Where the last column has 'Iris-setosa', the number of rows is : {irisSetosa.shape[0]} ")
print(f"Where the last column has 'Iris-setosa', the Avg value of 1st column is : {irisSetosa.iloc[:,0].mean():.2f} ")
print(f"Where the last column has 'Iris-setosa', the Max value of the second column is : {irisSetosa.iloc[:,1].max()} ")
print(f"Where the last column has 'Iris-setosa', the Min value of 3rd column is : {irisSetosa.iloc[:,2].min()} ")

##Scatter plot
#Renaming columns to help with reference
data.columns = ["col1","col2","col3","col4","col5" ]

#Defining Markers
markers = ["d","o","*" ]
categories = data["col5"].unique()


for i, category in enumerate(data["col5"].unique()):
    subset = data[data["col5"] == category]
    plt.scatter(
        subset["col1"],
        subset["col2"],
        label=category,
        marker=markers[i % len(markers)], 
        alpha=0.7,
    )


##Scatter plot title and labels
plt.title("Iris Scatter Plot Colored by Column 5")
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.legend()
plt.show()
