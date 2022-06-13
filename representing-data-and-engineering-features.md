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

```
df['category'] = df['category'].astype(str)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# sparse = False: returns an array
# sparse = True: turns a sparse matrix
encoder.fit(which_bin)
# OneHotEncoder only works for categorical vairables in numbers

```



### Binning, Discretization, Linear Models, and Trees

* binning (discretization): making **linear models on continuous data** more powerful

```
bins = np.linespace(start,end,numberofbins+1)
# why numberofbins+1: it is the number of entries

which_bin = np.digitize(data, bins = bins)
# then, the data will be transformed to the bin it belongs to
# the bin starts at 1

# we can also use OneHotEncoder for this
```

* **any model will predict the same results based on the bin**
* linear model ---> more flexible
  * benefits a lot ---> **increases modeling power**
* decision tres ---> less flexible
  * binning has no beneficial effect for trees
    * trees themselves know the best binning
    * trees can look at all the data at once



### Interactions and Polynomials

* one way to add slope to the models:
  * add back the original features to the bins
  * the new slope is shared among the bins
* add ing polynomials of the original features ---> each bin has individual slope

```
from sklearn.preprocessing import PolynomialFeatures

# include polynomials up to x ** 10:
# the default "include_bias=True" adds a feature that's constantly 1

poly = PolynomialFeatures(degree=10, include_bias=False) 
poly.fit(X)
X_poly = poly.transform(X)

# it will include each x ** n for 1 =< n <= the_power

 poly.get_feature_names()
 # return the name of the features 
```

* smooth curve
* high degree, little data ---> extreme



###

