
# **Introduction**


Accurate property valuation is a common problem faced by every city government. It is a requirement for the collection of property tax and necessary for the development of an area. Poor valuation will give rise to illicit activities, under the table transactions and underutilization. 

Most houses are not sold, thus their market value must be extrapolated from sold houses. This becomes a fertile problem for the application of statistical methods and machine learning. We tackle this problem using two large datasets on Residential Sales & Property Characteristics obtained from Cook County in the US.

1. Sale of Houses from 2013-2019, with Characteristics. (500k Records)
2. Residential Property Characteristics of all houses. (2Mn Records)

[A full description of the data](https://datacatalog.cookcountyil.gov/stories/s/p2kt-hk36).


```
# Load Data
PATH = '/content/drive/My Drive/Projects/Machine Learning/Real World Problems/Residential Property Valuation/Data/'
import pandas as pd, numpy as np
df = pd.read_csv(PATH + 'sales.csv').sample(frac=0.1)

# Rename Columns
df.columns = ['PIN'] + [''.join([j for j in i.title() if j.isalnum()]) for i in df.columns[1:]]
firstCols = ['PIN', 'PropertyClass', 'SaleDate', 'SalePrice', 'EstimateBuilding', 'PureMarketFilter', 'BuildingSquareFeet', 'LandSquareFeet' ]
df = df[firstCols + [i for i in df.columns if i not in firstCols]]

df.head()
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
      <th>PIN</th>
      <th>PropertyClass</th>
      <th>SaleDate</th>
      <th>SalePrice</th>
      <th>EstimateBuilding</th>
      <th>PureMarketFilter</th>
      <th>BuildingSquareFeet</th>
      <th>LandSquareFeet</th>
      <th>NeighborhoodCode</th>
      <th>TownCode</th>
      <th>TypeOfResidence</th>
      <th>Apartments</th>
      <th>WallMaterial</th>
      <th>RoofMaterial</th>
      <th>Rooms</th>
      <th>Bedrooms</th>
      <th>Basement</th>
      <th>BasementFinish</th>
      <th>CentralHeating</th>
      <th>OtherHeating</th>
      <th>CentralAir</th>
      <th>Fireplaces</th>
      <th>AtticType</th>
      <th>AtticFinish</th>
      <th>HalfBaths</th>
      <th>DesignPlan</th>
      <th>CathedralCeiling</th>
      <th>ConstructionQuality</th>
      <th>Renovation</th>
      <th>SiteDesirability</th>
      <th>Garage1Size</th>
      <th>Garage1Material</th>
      <th>Garage1Attachment</th>
      <th>Garage1Area</th>
      <th>Garage2Size</th>
      <th>Garage2Material</th>
      <th>Garage2Attachment</th>
      <th>Garage2Area</th>
      <th>Porch</th>
      <th>OtherImprovements</th>
      <th>...</th>
      <th>EstimateLand</th>
      <th>DeedNo</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>CensusTract</th>
      <th>TotalBuildingSquareFeet</th>
      <th>MultiPropertyIndicator</th>
      <th>PropertyAddress</th>
      <th>ModelingGroup</th>
      <th>FullBaths</th>
      <th>Age</th>
      <th>Use</th>
      <th>NumberOfUnits</th>
      <th>PercentOwnership</th>
      <th>CondoClassFactor</th>
      <th>MultiFamilyIndicator</th>
      <th>LargeLot</th>
      <th>ConditionDesirabilityAndUtility</th>
      <th>OHareNoise</th>
      <th>Floodplain</th>
      <th>RoadProximity</th>
      <th>CondoStrata</th>
      <th>SaleYear</th>
      <th>SaleQuarter</th>
      <th>SaleHalfYear</th>
      <th>SaleQuarterOfYear</th>
      <th>SaleMonthOfYear</th>
      <th>SaleHalfOfYear</th>
      <th>MostRecentSale</th>
      <th>AgeSquared</th>
      <th>AgeDecade</th>
      <th>AgeDecadeSquared</th>
      <th>LotSizeSquared</th>
      <th>ImprovementSizeSquared</th>
      <th>GarageIndicator</th>
      <th>NeigborhoodCodeMapping</th>
      <th>SquareRootOfLotSize</th>
      <th>SquareRootOfAge</th>
      <th>SquareRootOfImprovementSize</th>
      <th>TownAndNeighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>340910</th>
      <td>30204090100000</td>
      <td>203</td>
      <td>02/25/2015</td>
      <td>27000</td>
      <td>60190</td>
      <td>1</td>
      <td>1138.0</td>
      <td>4760.0</td>
      <td>101</td>
      <td>37</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>...</td>
      <td>15470</td>
      <td>1505601030</td>
      <td>-87.528277</td>
      <td>41.588988</td>
      <td>826202.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1531 KENILWORTH DR</td>
      <td>SF</td>
      <td>1.0</td>
      <td>65</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2015</td>
      <td>73</td>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>4225</td>
      <td>6.5</td>
      <td>42.25</td>
      <td>2.265760e+07</td>
      <td>1295044.0</td>
      <td>1.0</td>
      <td>101</td>
      <td>68.992753</td>
      <td>8.062258</td>
      <td>33.734256</td>
      <td>37101</td>
    </tr>
    <tr>
      <th>395458</th>
      <td>17284010340000</td>
      <td>211</td>
      <td>08/30/2016</td>
      <td>415000</td>
      <td>264420</td>
      <td>1</td>
      <td>3339.0</td>
      <td>2640.0</td>
      <td>30</td>
      <td>76</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>18.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>...</td>
      <td>92400</td>
      <td>1624310108</td>
      <td>-87.634863</td>
      <td>41.844273</td>
      <td>340300.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>312 W 27TH ST</td>
      <td>MF</td>
      <td>3.0</td>
      <td>137</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>211.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2016</td>
      <td>79</td>
      <td>40</td>
      <td>3</td>
      <td>8</td>
      <td>2</td>
      <td>1.0</td>
      <td>18769</td>
      <td>13.7</td>
      <td>187.69</td>
      <td>6.969600e+06</td>
      <td>11148921.0</td>
      <td>1.0</td>
      <td>30</td>
      <td>51.380930</td>
      <td>11.704700</td>
      <td>57.784081</td>
      <td>7630</td>
    </tr>
    <tr>
      <th>205507</th>
      <td>10232160640000</td>
      <td>203</td>
      <td>12/12/2013</td>
      <td>345000</td>
      <td>214540</td>
      <td>1</td>
      <td>1346.0</td>
      <td>4920.0</td>
      <td>81</td>
      <td>24</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>...</td>
      <td>40590</td>
      <td>1334635046</td>
      <td>-87.714261</td>
      <td>42.036871</td>
      <td>807200.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>8549 TRUMBULL AVE</td>
      <td>SF</td>
      <td>1.0</td>
      <td>51</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2013</td>
      <td>68</td>
      <td>34</td>
      <td>4</td>
      <td>12</td>
      <td>2</td>
      <td>1.0</td>
      <td>2601</td>
      <td>5.1</td>
      <td>26.01</td>
      <td>2.420640e+07</td>
      <td>1811716.0</td>
      <td>1.0</td>
      <td>81</td>
      <td>70.142712</td>
      <td>7.141428</td>
      <td>36.687873</td>
      <td>2481</td>
    </tr>
    <tr>
      <th>512443</th>
      <td>13292140240000</td>
      <td>203</td>
      <td>06/13/2016</td>
      <td>176500</td>
      <td>158500</td>
      <td>1</td>
      <td>1048.0</td>
      <td>3720.0</td>
      <td>200</td>
      <td>71</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>...</td>
      <td>40920</td>
      <td>1616562007</td>
      <td>-87.767986</td>
      <td>41.936351</td>
      <td>190402.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>3048 N PARKSIDE AVE</td>
      <td>SF</td>
      <td>1.0</td>
      <td>90</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2016</td>
      <td>78</td>
      <td>39</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>0.0</td>
      <td>8100</td>
      <td>9.0</td>
      <td>81.00</td>
      <td>1.383840e+07</td>
      <td>1098304.0</td>
      <td>1.0</td>
      <td>200</td>
      <td>60.991803</td>
      <td>9.486833</td>
      <td>32.372828</td>
      <td>71200</td>
    </tr>
    <tr>
      <th>374646</th>
      <td>17221050331021</td>
      <td>299</td>
      <td>10/28/2014</td>
      <td>770000</td>
      <td>430730</td>
      <td>1</td>
      <td>NaN</td>
      <td>34442.0</td>
      <td>12</td>
      <td>76</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>...</td>
      <td>119040</td>
      <td>1430122010</td>
      <td>-87.622874</td>
      <td>41.864777</td>
      <td>330100.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1336 S INDIANA AVE</td>
      <td>NCHARS</td>
      <td>NaN</td>
      <td>10</td>
      <td>1</td>
      <td>36.0</td>
      <td>0.03</td>
      <td>299.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2014</td>
      <td>72</td>
      <td>36</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>0.0</td>
      <td>100</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.186251e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12</td>
      <td>185.585560</td>
      <td>3.162278</td>
      <td>NaN</td>
      <td>7612</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>



**Ensuring Data Integrity**
* Condominiums are shared properties, and for which we have lesser data so we will handle them separately. 
* We remove all Sales not done through the Free Market, this removes extreme values.
* We remove all Records where Sale was very far from its previous estimated valuation. This takes care of distress sales and other extremities. 

This brings the Median Sale Price to $215,000 for 2013-2019, which is not far from its Zillow estimate



```
# Remove Condominiums
df = df[df.PropertyClass != 299]

# Retain Pure Market Sales
df = df[df.PureMarketFilter == 1]

# Remove Sales too far from Last Valuation
df['DiffSaleEstm'] = df['SalePrice'] - df['EstimateBuilding']
μ, σ = df.DiffSaleEstm.mean(), df.DiffSaleEstm.std()
df = df[(df.DiffSaleEstm > μ - 3 * σ) & (df.DiffSaleEstm < μ + 3 * σ)]

# Merge Codes
df = df.merge(pd.read_csv(PATH + 'townships.csv'), left_on = 'TownCode', right_on = 'township_code')
```

# **Helper Functions**


*PlotDist* - Plots Historgram with μ ± 3σ Bars.

*CatPlot* - Plots Mean LogSalePrice Distributions across ordinal values, ordered.

*DataGen* - Uses a list of variables to create training and test sets. Ensures numericals get standardized and ordinals get dummified. 

*NNReg* Function - Neural Net Regression between LogSalePrice and Selected Variables.

<!-- 

```
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)
def PlotDist(var):
    μ, σ = df[var].mean(), df[var].std()
    sns.distplot(df[df[var] < μ + 5 * σ][var], hist=True, kde = False)
    plt.axvline(μ - 3 * σ, color='black', linestyle = '--')
    plt.axvline(μ + 3 * σ, color='black', linestyle = '--')
    plt.title(f'Distribution of {var}')
    plt.show()
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm



```
def CatPlot(var, sortVar = 'TSalePrice'):
    dfTemp = df.sample(frac = 0.05)
    df_order = dfTemp[['TSalePrice', var]].groupby(by=var).mean().sort_values(sortVar)
    chart = sns.violinplot(x=var, y='TSalePrice', data=dfTemp, order=df_order.index)
    chart.axhline(y = df.TSalePrice.mean(), color='black', linewidth=2) 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right',fontweight='light',fontsize='x-large')
    plt.title(f'Distribution of TSalePrice with {var}')
    df_order = df_order.reset_index()
    if df_order[var].nunique() > 10:
        print(f'Distinct Values Taken: {df[var].nunique()}') 
        print(f'Top 5 {var}: {df_order[var].head().unique()}')
        print(f'Bottom 5 {var}: {df_order[var].tail().unique()}')
    plt.show()

```


```
from sklearn.model_selection import train_test_split
def DataGen(var):
    global df
    y = df.TSalePrice
    x = df[var].copy()
    for i in x.select_dtypes(exclude = 'object'):
        ε = 0.0000001
        x[i] = (x[i]-x[i].mean(axis = 0)+ε)/(x[i].std(axis = 0)+ε)
    for i in x.select_dtypes(include = 'object').columns:
        x = pd.concat([x, pd.get_dummies(x[i])], axis = 1)
        x = x.drop(i, axis = 1)
    return train_test_split(x, y, test_size = 0.3, random_state = 42)
```


```
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from warnings import filterwarnings
filterwarnings('ignore')
def NNReg(var, H = (100, ), N = 200, α = 0.01, L1 = 2, plot = False):
    x_train, x_val, y_train, y_val = DataGen(var)
    #N_h1, N_h2 = round(x_train.shape[1] * 1.5), round(x_train.shape[1] * 0.5)
    model = MLPRegressor(hidden_layer_sizes = H, alpha = L1, learning_rate_init = α, max_iter = N, random_state = 42, verbose = False)
    model.fit(x_train, y_train)
    yhat = model.predict(x_train)
    print(f'Train R Square: {r2_score(y_train, yhat):0.3f}')
    print(f'Test R Square: {r2_score(y_val, model.predict(x_val)):0.3f}')
    if plot == True: 
        plt.figure(figsize=(16, 6))
        plt.scatter(y_val, y_val, c = 'black',  marker='.', label = 'Actual Values vs Actual Values')
        plt.scatter(model.predict(x_val), y_val, c = 'red', marker='.', label = 'Actual Values vs Predicted Values')
        plt.title('Regression Model for logSalePrice')
        plt.ylabel('True Values')
        plt.xlabel('Predictions')
        plt.legend()
        plt.show()
    return model
```
-->

# **Transformation of Sale Price**

Like most Financial Variables, it is right skewed so we apply Box-Cox Transformation with Optimal λ



```
PlotDist('SalePrice')
```


![png](residential-property-valuation_files/residential-property-valuation_12_0.png)



```
from scipy import stats
df['TSalePrice'], λ, (ci_l, ci_u) = stats.boxcox(df['SalePrice'], lmbda = None, alpha = 0.05)
print(f"Estimated λ: {λ:0.2f}")
PlotDist('TSalePrice')
```

    Estimated λ: 0.18



![png](residential-property-valuation_files/residential-property-valuation_13_1.png)


# **Market Value by Location** 


**Administrative Blocks**
* There exist a total of 38 Townships in Cook County, including Chicago City
* The 14 digit [Property Index Numbers](https://www.cookcountyclerk.com/service/about-property-index-number-pin) can be used to break down geography by administrative blocks



```
from PIL import Image
Image.open(PATH + 'map.png')
```




![png](residential-property-valuation_files/residential-property-valuation_16_0.png)




```
plt.rcParams["figure.figsize"] = (16,6)
CatPlot('township_name')
```

    Distinct Values Taken: 38
    Top 5 township_name: ['Calumet' 'Thornton' 'Bloom' 'Hyde Park' 'Lake']
    Bottom 5 township_name: ['Northfield' 'River Forest' 'Lake View' 'New Trier' 'North Chicago']



![png](residential-property-valuation_files/residential-property-valuation_17_1.png)



```
df['PIN'] = df["PIN"].astype(str).str.pad(14, side ='left', fillchar='0') # Permanent Identification Number
df['PIN1']=df.PIN.str[0:2]; print('No of Area/Sequential-Townships:', df.PIN1.nunique())  
df['PIN2']=df.PIN.str[0:4]; print('No of SubArea:', df.PIN2.nunique())
df['PIN3']=df.PIN.str[0:7]; print('No of Blocks:', df.PIN3.nunique())
df['PIN4']=df.PIN.str[0:10]; print('No of Parcels/Homes:', df.PIN4.nunique())
```

    No of Area/Sequential-Townships: 33
    No of SubArea: 862
    No of Blocks: 21018
    No of Parcels/Homes: 31907



```
CatPlot('PIN2')
```

    Distinct Values Taken: 862
    Top 5 PIN2: ['3229' '2906' '3230' '2909' '2824']
    Bottom 5 PIN2: ['1830' '1704' '1432' '0527' '1428']



![png](residential-property-valuation_files/residential-property-valuation_19_1.png)


**Geographical Clusters**
* GIS Data can be used to produce a map of House locations and a Map of Cook County
* Using Lat/Long co-ordinates for Sale location we can obtain clusters/hubs of Sale activity.
* These clusters/hubs can be new geographical divisions and act as ordinal features

<!-- 
```
# Load Street Map
#%pip install geopandas
import geopandas as geo
from shapely.geometry import Point, Polygon
streetMapPATH = '/content/drive/My Drive/Projects/Machine Learning/Real World Problems/Residential Property Valuation/Data/tl_2018_17031_addrfeat.shp'
streetMap = geo.read_file(streetMapPATH)

# Cluster Detection - 8, 16, 32 Clusters
df_cluster = df[['PIN', 'DeedNo', 'TSalePrice', 'Longitude', 'Latitude']].dropna()
from sklearn.cluster import KMeans
df_cluster = df[['PIN', 'DeedNo', 'TSalePrice', 'Longitude', 'Latitude']].dropna()
df_cluster['xy_kmeans8'] = KMeans(n_clusters=8, random_state=42).fit(df_cluster[['Latitude', 'Longitude']]).labels_
df_cluster['xy_kmeans16'] = KMeans(n_clusters=16, random_state=42).fit(df_cluster[['Latitude', 'Longitude']]).labels_
df_cluster['xy_kmeans32'] = KMeans(n_clusters=32, random_state=42).fit(df_cluster[['Latitude', 'Longitude']]).labels_
```


```
# Geodataframes 
df_cluster_centroids = df_cluster.groupby('xy_kmeans16').mean()
geometry1 = [Point(xy) for xy in zip(df_cluster['Longitude'], df_cluster['Latitude'])]
geometry2 = [Point(xy) for xy in zip(df_cluster_centroids['Longitude'], df_cluster_centroids['Latitude'])]
gdf1 = geo.GeoDataFrame(df_cluster, crs = {'init':'epsg:4326'}, geometry=geometry1)
gdf2 = geo.GeoDataFrame(df_cluster_centroids, crs = {'init':'epsg:4326'}, geometry=geometry2)

# Plot
fig, ax = plt.subplots(figsize = (20,20))
streetMap.plot(ax = ax, alpha=0.5, color = 'grey')
gdf1.sample(frac = 0.1).plot(ax = ax, markersize = 1, color = 'red', label = 'Sold Houses')
gdf2.plot(ax = ax, markersize = 20, color = 'blue', label = 'Cluster Centers')
plt.legend()
plt.show()
```


![png](residential-property-valuation_files/residential-property-valuation_22_0.png)



```
# Add features to dataset
df = df.merge(df_cluster.drop('geometry', axis = 1), on = ['PIN', 'DeedNo', 'TSalePrice'])
```


```
# Scaling Price & Lat/Long
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
X = df_cluster[['Latitude', 'Longitude', 'TSalePrice']]
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
```
-->

```
CatPlot('xy_kmeans16')
```

    Distinct Values Taken: 16
    Top 5 xy_kmeans16: [11 15 10  3  6]
    Bottom 5 xy_kmeans16: [12  5  2  4  0]



![png](residential-property-valuation_files/residential-property-valuation_25_1.png)


**Location Model**

* We are able to explain above 60% of variation in Sale Price with Location Attributes
* However there is still variation within same locations. 
* Location cannot explain extreme Sales


```
# Other measures
for i in ['Floodplain', 'RoadProximity', 'OHareNoise']:
    df[i].fillna(0, inplace = True)

LOCATION = ['township_name', 'PIN2', 'xy_kmeans16', 'Floodplain', 'RoadProximity', 'OHareNoise']
df.Floodplain.fillna(0, inplace = True)
modelLOCATION = NNReg(LOCATION, H = (500,), N = 200, α = 0.1, L1 = 1, plot = True)
```

    Train R Square: 0.641
    Test R Square: 0.635



![png](residential-property-valuation_files/residential-property-valuation_27_1.png)



```

```
