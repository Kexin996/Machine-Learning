# Representing Data and Engineering Features

## Categorical Variables

### One-Hot-Encoding (Dummy Variables)

* replace a categorical variable with either 0 or 1
* binary

![](<.gitbook/assets/Screen Shot 2022-06-12 at 6.37.57 PM.png>)

* process:
  * find out the unique data to by using value\_counts()
    * we can also use it to find similar categories which might be able to be merged

```
data.value_counts()

data_dummy = pd.get_dummies(data)
# it is a dataframw
# it will not touch the continuous data
# don't forget to separate the output and input data
```

* **in pandas, column slicing will include the end of range**
* make sure training set and test set share **the same number of variables and meanings**



### Numbers Can Encode Category

* pandas cannot recognize this, we need to use **scikit-learn's OneHotEncoder by pointing out the variables or convert the numbers to string in dataframe**

****

###

```
df['category'] = df['category'].astype(str)
```
