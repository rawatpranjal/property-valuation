
# **Introduction**

* One of the most fascinating datasets that I've come across is the Ames Housing dataset. It was originally collated by Dr. Dean De Cock (Truman State Univ) for the undergraduate statistical methods course, and later became very popular through Kaggle.
* It contains rich information captured at the point of sale of houses in Ames (Iowa, US) from 2006 to 2010. Among others we have the location and various area measurements of the house, the immediate context of sale, the condition of rooms, etc..
* One of the popular "tasks" on this dataset to try and predict the Sale Price from the attributes of the house - its location, size, quality, etc. But there can be other interesting "tasks".
* While Kaggle split the data into train and test sets, the original dataset is larger and more complete.



```
PATH = '/content/drive/My Drive/Projects/Machine Learning/Real World Problems/Housing Prices/AmesHousing.xls'
import pandas as pd
df = pd.read_excel(PATH)
df.columns = [i.replace(' ', '') for i in df.columns]
df[['PID', 'SalePrice', 'Neighborhood', 'GrLivArea', 'YrSold', 'MoSold']].sample(frac = 1).head(10)
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
      <th>PID</th>
      <th>SalePrice</th>
      <th>Neighborhood</th>
      <th>GrLivArea</th>
      <th>YrSold</th>
      <th>MoSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1646</th>
      <td>527302060</td>
      <td>176000</td>
      <td>NWAmes</td>
      <td>1479</td>
      <td>2007</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2912</th>
      <td>923226150</td>
      <td>146500</td>
      <td>Mitchel</td>
      <td>1652</td>
      <td>2006</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2752</th>
      <td>906380170</td>
      <td>194000</td>
      <td>CollgCr</td>
      <td>1220</td>
      <td>2006</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2799</th>
      <td>907285100</td>
      <td>237000</td>
      <td>CollgCr</td>
      <td>1995</td>
      <td>2006</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1517</th>
      <td>909175050</td>
      <td>179500</td>
      <td>SWISU</td>
      <td>2192</td>
      <td>2008</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1559</th>
      <td>911370410</td>
      <td>392500</td>
      <td>Crawfor</td>
      <td>1652</td>
      <td>2008</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1712</th>
      <td>528172050</td>
      <td>245700</td>
      <td>NridgHt</td>
      <td>1614</td>
      <td>2007</td>
      <td>5</td>
    </tr>
    <tr>
      <th>222</th>
      <td>905105200</td>
      <td>137900</td>
      <td>Sawyer</td>
      <td>892</td>
      <td>2010</td>
      <td>6</td>
    </tr>
    <tr>
      <th>619</th>
      <td>534476320</td>
      <td>128900</td>
      <td>NAmes</td>
      <td>1050</td>
      <td>2009</td>
      <td>6</td>
    </tr>
    <tr>
      <th>966</th>
      <td>916460110</td>
      <td>170000</td>
      <td>Timber</td>
      <td>2014</td>
      <td>2009</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# **The Target: Sale Price in $**

Two observations: 
1. A majority of houses are sold for less, a minority of houses are sold for a really, really high amount. Sale Price exhibits high left skew, and thus needs to be log-transformed. 
2. Post log-transformation, we detect outliers by plotting MEAN -+2 STD vertical bars. 


```
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,6)

sns.distplot(df.SalePrice, hist=True, kde = False)
plt.title('Distribution of Sale Price')
plt.show()
```


![png](Ames_Housing_files/Ames_Housing_5_0.png)



```
df['logSalePrice'] = np.log(df.SalePrice)
sns.distplot(df.logSalePrice, hist=True, kde = False)
plt.title('Distribution of Log Sale Price')

# Outlier Thresholds
μ = df.logSalePrice.mean()
σ = df.logSalePrice.std()
plt.axvline(μ - 3 * σ, color='black', linestyle = '--')
plt.axvline(μ + 3 * σ, color='black', linestyle = '--')

plt.show()
```


![png](Ames_Housing_files/Ames_Housing_6_0.png)


* We remove outliers below and above these statistical thresholds. 
* Additionally, by the recommendation of the author we remove houses with "GrLivArea" greater than 4000. 


```
old_examples = df.shape[0]
df = df[(df.logSalePrice > μ - 3 * σ) &
        (df.logSalePrice < μ + 3 * σ) & 
        (df.GrLivArea<4000)]
new_examples = df.shape[0]
print(old_examples, new_examples)
```

    2930 2907


* A total of 23 records are dropped, and we are left with 2907 records. 


```
sns.distplot(df.logSalePrice, hist=True, kde = False)
plt.title('Cleaned-Up Log Sale Price')
μ = df.logSalePrice.mean()
σ = df.logSalePrice.std()
plt.axvline(μ - 3 * σ, color='black', linestyle = '--')
plt.axvline(μ + 3 * σ, color='black', linestyle = '--')
plt.show()
```


![png](Ames_Housing_files/Ames_Housing_10_0.png)


# **Variation of Price by Location**
* SalePrices vary a lot by Neighbourhood & Section (from PIN)



```
import seaborn as sns
import matplotlib.pyplot as plt
def plot_categorical(var):
    plt.figure(figsize=(16, 6))
    my_order = df[['SalePrice', var]].groupby(by=var).mean().sort_values('SalePrice').index
    chart = sns.violinplot(x=var, y='SalePrice', data=df, order=my_order)
    chart.axhline(y = df.SalePrice.mean(), color='black', linewidth=2) 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right',fontweight='light',fontsize='x-large')
    return chart

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
def regression_model(var, return_weights = False):
    x, y = pd.get_dummies(df[var]), df.SalePrice
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 42)
    
    model = MLPRegressor(hidden_layer_sizes=(20,), alpha = 2, learning_rate_init=0.01, max_iter=1000, random_state = 42)
    model.fit(x_train, y_train)
    if return_weights == True:
        return model.coefs_
    print(r2_score(y_train, model.predict(x_train)), r2_score(y_val, model.predict(x_val)))

    plt.figure(figsize=(16, 6))
    plt.plot(y_train, y_train)
    plt.scatter(y_train, model.predict(x_train))
    plt.show()

```


```
plot_categorical('MSZoning')
regression_model('MSZoning')
```


```
df['Zone'] = df.MSZoning.astype(str) + '~' + df.MSSubClass.astype(str)
plot_categorical('Zone')
regression_model('Zone')
```


```
plot_categorical('Neighborhood')
regression_model('Neighborhood')
```


```
df.PID = df.PID.astype(str) # Postal Identification Number
df['PID1']=df.PID.str[0:1] # township
df['PID2']=df.PID.str[1:3] # section number
df['PID3']=df.PID.str[3:6] # quarter section (N, E, S, W, etc)
df['PID4']=df.PID.str[6:10] # parcel number within section
```


```
plot_categorical('PID2')
regression_model('PID2')
```


```
plot_categorical('PID3')
regression_model('PID3')
```


```
plot_categorical('PID4')
regression_model('PID4')
```


```
plot_categorical('Condition1')
regression_model('Condition1')
```


```
plot_categorical('Condition2')
regression_model('Condition2')
```


```
keyLocationAttributes = ['Neighborhood', 'PID2', 'Zone']
coef = regression_model(keyLocationAttributes, return_weights=True)
```


```
for i in coef:
    print(i.shape)
```

    (106, 100)
    (100, 1)



```
eM = coef[0]
eM.shape
```




    (106, 100)



# Context of Sale
* Does not seem to have any significant effect. 


```
plot_categorical('SaleCondition')
regression_model('SaleCondition')
```

    0.08989407979589081 0.14527501146496014



![png](Ames_Housing_files/Ames_Housing_26_1.png)



![png](Ames_Housing_files/Ames_Housing_26_2.png)



```
plot_categorical('SaleType')
regression_model('SaleType')
```

    0.10674419934127866 0.13913094584412988



![png](Ames_Housing_files/Ames_Housing_27_1.png)



![png](Ames_Housing_files/Ames_Housing_27_2.png)



```
plot_categorical('MoSold')
regression_model('MoSold')
```

    0.0009556132307242393 -0.014616219508186035



![png](Ames_Housing_files/Ames_Housing_28_1.png)



![png](Ames_Housing_files/Ames_Housing_28_2.png)



```
plot_categorical('YrSold')
regression_model('YrSold')
```

    -0.019410817390614676 -0.044146156948597026



![png](Ames_Housing_files/Ames_Housing_29_1.png)



![png](Ames_Housing_files/Ames_Housing_29_2.png)



```
df['YrMoSold'] = df.YrSold.astype(str) + df.MoSold.astype(str)
plot_categorical('YrMoSold')
regression_model('YrMoSold')
```

    0.006909498505049849 -0.059339581739229486



![png](Ames_Housing_files/Ames_Housing_30_1.png)



![png](Ames_Housing_files/Ames_Housing_30_2.png)



```
keyLocationAttributes = ['SaleType', 'SaleCondition']
coef = regression_model(keyLocationAttributes, return_weights=True)
```

# House Size & Area
* Key takeaways - The Right Definition of Floor Area should be applied. Price is dependent on the right area. 


```
df.select_dtypes(include = ['float', 'int']).head().columns
```




    Index(['Order', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
           'OverallCond', 'YearBuilt', 'YearRemod/Add', 'MasVnrArea', 'BsmtFinSF1',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
           'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
           'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
           'MiscVal', 'MoSold', 'YrSold', 'SalePrice'],
          dtype='object')




```
df['PorchArea'] = df.OpenPorchSF +df.EnclosedPorch +df['3SsnPorch'] + df.ScreenPorch
df['FloorArea1'] = df.TotalBsmtSF + df['1stFlrSF'] + df['2ndFlrSF']
df['FloorArea2'] = df.TotalBsmtSF + df['1stFlrSF'] + df['2ndFlrSF'] + df.EnclosedPorch + df.ScreenPorch + df['3SsnPorch']
df['FloorArea3'] = df.BsmtFinSF1 + 2 * df['1stFlrSF'] + df['2ndFlrSF'] + df.WoodDeckSF + df.PorchArea + 2* df.GarageArea 
df['OutdoorArea']  = df.LotArea - df['1stFlrSF']
df['FloorAreaRatio'] = df['FloorArea3']/df['PorchArea']
#areaFeatures = ['LotArea', 'OutdoorArea', 'FloorArea1','FloorArea2','FloorArea3', 'GrLivArea', 'GarageArea']
areaFeatures = ['LotArea', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'FloorArea3', 'FloorArea2', 'FloorArea1', 'PorchArea', 'OutdoorArea', 'FloorAreaRatio'
       ]
df[areaFeatures] = df[areaFeatures].fillna(0)
df[areaFeatures].head()
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
      <th>LotArea</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>FloorArea3</th>
      <th>FloorArea2</th>
      <th>FloorArea1</th>
      <th>PorchArea</th>
      <th>OutdoorArea</th>
      <th>FloorAreaRatio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>726</th>
      <td>7879</td>
      <td>0.0</td>
      <td>495.0</td>
      <td>0.0</td>
      <td>225.0</td>
      <td>720.0</td>
      <td>720</td>
      <td>0</td>
      <td>0</td>
      <td>720</td>
      <td>0.0</td>
      <td>0</td>
      <td>523</td>
      <td>115</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2573.0</td>
      <td>1555.0</td>
      <td>1440.0</td>
      <td>638</td>
      <td>7159</td>
      <td>4.032915</td>
    </tr>
    <tr>
      <th>2843</th>
      <td>8088</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>498.0</td>
      <td>498.0</td>
      <td>498</td>
      <td>0</td>
      <td>0</td>
      <td>498</td>
      <td>216.0</td>
      <td>0</td>
      <td>0</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1528.0</td>
      <td>1096.0</td>
      <td>996.0</td>
      <td>100</td>
      <td>7590</td>
      <td>15.280000</td>
    </tr>
    <tr>
      <th>2880</th>
      <td>9000</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>430.0</td>
      <td>480.0</td>
      <td>480</td>
      <td>0</td>
      <td>0</td>
      <td>480</td>
      <td>308.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1626.0</td>
      <td>960.0</td>
      <td>960.0</td>
      <td>0</td>
      <td>8520</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>709</th>
      <td>5925</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>600.0</td>
      <td>600.0</td>
      <td>600</td>
      <td>368</td>
      <td>0</td>
      <td>968</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1568.0</td>
      <td>1568.0</td>
      <td>1568.0</td>
      <td>0</td>
      <td>5325</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>5000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>334</td>
      <td>0</td>
      <td>0</td>
      <td>334</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>668.0</td>
      <td>334.0</td>
      <td>334.0</td>
      <td>0</td>
      <td>4666</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(15, 15))
sns.heatmap(df[['SalePrice']+ areaFeatures].corr(), annot=True, cmap = 'Blues')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5acf6b9208>




![png](Ames_Housing_files/Ames_Housing_35_1.png)



```
areaFeatures = ['TotalBsmtSF', 'GrLivArea', 'GarageArea','FloorArea1']
```


```
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
def regression_model2(var, return_weights = False):
    model = MLPRegressor(hidden_layer_sizes=(100, ), alpha = 3, learning_rate_init=0.01, max_iter=20000, random_state = 42)
    y = df.SalePrice
    df[var] = (df[var]-df[var].mean(axis = 0))/(df[var].std(axis = 0))
    x, y = df[var], df.SalePrice
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 42)
    model.fit(x_train, y_train)
    print(r2_score(y_train, model.predict(x_train)), r2_score(y_val, model.predict(x_val)))
    if return_weights == True:
        return model.coefs_
        
    plt.figure(figsize=(16, 6))
    plt.plot(y_train, y_train)
    plt.scatter(y_train, model.predict(x_train))
    plt.show()

    for i in var:
        plt.scatter(x_train[i], model.predict(x_train))
        plt.title(i)
        plt.show()

```


```
areaFeatures = ['TotalBsmtSF', 'GrLivArea', 'GarageArea','FloorArea1']
regression_model2(areaFeatures)
```

    0.7269140738653929 0.7390095585649867



![png](Ames_Housing_files/Ames_Housing_38_1.png)



![png](Ames_Housing_files/Ames_Housing_38_2.png)



![png](Ames_Housing_files/Ames_Housing_38_3.png)



![png](Ames_Housing_files/Ames_Housing_38_4.png)



![png](Ames_Housing_files/Ames_Housing_38_5.png)



```
f = list(df.columns)
f.drop('Order', 'PID')
```




    Index(['Order', 'PID', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',
           'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
           'YearRemod/Add', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
           'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'SalePrice', 'PorchArea', 'FloorArea1', 'FloorArea2',
           'FloorArea3', 'OutdoorArea', 'FloorAreaRatio'],
          dtype='object')




```
var = areaFeatures
model = MLPRegressor(hidden_layer_sizes=(100, ), alpha = 3, learning_rate_init=0.01, max_iter=10000, random_state = 42)
y = df.SalePrice
df[var] = (df[var]-df[var].mean(axis = 0))/(df[var].std(axis = 0))
x, y = df[var], df.SalePrice
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 42)
model.fit(x_train, y_train)
print(r2_score(y_train, model.predict(x_train)), r2_score(y_val, model.predict(x_val)))
```

    0.6911405032509836 0.6901373049147072



```
yhat_train = model.predict(x_train)
yhat_test = model.predict(x_val)
```


```
yhat = model.predict(x[areaFeatures])
```


```
var = ['Neighborhood', 'PID2', 'Zone']
x2 = pd.DataFrame([[]])
for i in var:
    x2 = pd.concat([x2, pd.get_dummies(df[i])], axis = 1)
y2 = yhat
x2.shape, y2.shape

```




    ((2918, 106), (2918,))




```
x_train2, x_val2, y_train2, y_val2 = train_test_split(x2, y2, test_size = 0.3, random_state = 42)
model2 = MLPRegressor(hidden_layer_sizes=(100, ), alpha = 3, learning_rate_init=0.01, max_iter=10000, random_state = 42)
model2.fit(x_train2, y_train2)
print(r2_score(y_train2, model2.predict(x_train2)), r2_score(y_val2, model2.predict(x_val2)))
```

    0.04855021237855084 -0.035088942405500756




# Interior Components


```

```
