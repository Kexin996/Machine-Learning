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
# it is a dataframe
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
* adding polynomials of the original features ---> each bin has individual slope

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



### Univariate Nonlinear Transformations

* the models work best for **gaussian distribution**&#x20;
  * log / ex ---> **use them to make the data distribution normal**&#x20;
* important for simple models: **Bayes / linear regression**



## Automatic Feature Selection

* supervised
* when we are **not sure about the features being selected**



### Univariate Statistics (ANOVA)

* analysis of variance
* **select the variables with high confidence**
* independent of models

```
SelectKBest ---> select a fixed number of features
SelectPercentile ---> select a fixed percentage of features
```



### Model-Based Feature Selection

* need measure of importance for each feature
  * decision trees: feature\_importances\_
* **the model for feature selection doesn't have to be the same for the final models**
* consider all features at once
* more power&#x20;

```
from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestClassifier 

select = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold="median")

# select will selects all the feature > threshold (median)
```

### Iterative Feature Selection

* two types:
  * adding feature one by one starting at 0
  * removing features one by one starting from all features
* recursive feature elimination: type 2
  * discard the least important features
  * the results are **better than model-based feature selection and anova**
  * **longer time than model-based selection** since we need to train final\_features time

```
from sklearn.feature_selection import RFE

# select from randomforest
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
                 n_features_to_select=40)
select.fit(X_train, y_train)

# we can also use the model inisde rfe to predict
# only for the features that we selected
# in this case the model is random forest
select.score(X_test, y_test)
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)

# once we reach the correct features
# linear regression and random forest produce the same accuracy
```



## Utilizing Expert Knowledge

* adding **more encoded features about the nature of the task**
* use **common sense**
  * e.g. during holiday ---> more flight
* how do we use time series to predict:
  * **learn from the past and predict for the feature**&#x20;
  *
* **trees cannot extrapolate the data outside the range of training data**
  * **it only predicts the closet in the training data ---> the last point**
  * **why expert knowledge is important or trees:**
    * **the regression cannot recognize some labels, such as date**
    * **but if we change it to hours / weekdays, the trees can "retrieve" it**
* linear regression is better for visualization ---> we have **coefficients**

```
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True,
                                          include_bias=False)
# the interaction_only == true:
# we only count the product less than 2
# for example, we will have x_0, x_0 * x_1
# but excluseds x_0 **2, x_0 **2 * x_1
```

* time conversion:
  * [https://pyformat.info/](https://pyformat.info/)

