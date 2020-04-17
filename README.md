
# Import Data


```
import pandas as pd
import numpy as np
import io
from google.colab import files
uploaded = files.upload()
df = pd.read_excel(io.BytesIO(uploaded['AmesHousing.xls']))
```



     <input type="file" id="files-d213bb60-12d8-4633-b0db-f40e14fedbe1" name="files[]" multiple disabled />
     <output id="result-d213bb60-12d8-4633-b0db-f40e14fedbe1">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving AmesHousing.xls to AmesHousing.xls



```
cols = list(df.columns)
new_cols = []
for i in cols:
    new_cols.append(i.replace(' ', ''))
df.columns = new_cols
```

# SalePrice & Outliers


```
df.SalePrice = np.log(df.SalePrice)
df.sort_values(by = 'SalePrice', inplace = True)
outlier_idx = []
```


```
import seaborn as sns
sns.distplot(df.SalePrice)
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm





    <matplotlib.axes._subplots.AxesSubplot at 0x7f5ad91d97f0>




![png](test_files/test_5_2.png)



```
outlier_idx = [181, 1553, 1760, 1767]
df = df[(df.GrLivArea<4000)& ~(df.index.isin(outlier_idx))]
```


```
df.SalePrice.tail()
```




    1637    13.290564
    432     13.321214
    44      13.323927
    1063    13.329378
    2445    13.345507
    Name: SalePrice, dtype: float64



# Location
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



![png](test_files/test_23_1.png)



![png](test_files/test_23_2.png)



```
plot_categorical('SaleType')
regression_model('SaleType')
```

    0.10674419934127866 0.13913094584412988



![png](test_files/test_24_1.png)



![png](test_files/test_24_2.png)



```
plot_categorical('MoSold')
regression_model('MoSold')
```

    0.0009556132307242393 -0.014616219508186035



![png](test_files/test_25_1.png)



![png](test_files/test_25_2.png)



```
plot_categorical('YrSold')
regression_model('YrSold')
```

    -0.019410817390614676 -0.044146156948597026



![png](test_files/test_26_1.png)



![png](test_files/test_26_2.png)



```
df['YrMoSold'] = df.YrSold.astype(str) + df.MoSold.astype(str)
plot_categorical('YrMoSold')
regression_model('YrMoSold')
```

    0.006909498505049849 -0.059339581739229486



![png](test_files/test_27_1.png)



![png](test_files/test_27_2.png)



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




![png](test_files/test_32_1.png)



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



![png](test_files/test_35_1.png)



![png](test_files/test_35_2.png)



![png](test_files/test_35_3.png)



![png](test_files/test_35_4.png)



![png](test_files/test_35_5.png)



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
