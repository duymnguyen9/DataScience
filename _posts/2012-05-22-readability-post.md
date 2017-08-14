---
layout: post
title: "Testing Readability with a Bunch of Text"
date: 2012-05-22
excerpt: "A ton of text to test readability."
comments: true
---


### Import Libraries 


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
%matplotlib inline
```

### read csv file


```python
train_df = pd.read_csv('train.csv')
train_df.info()
train_df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
test_df = pd.read_csv('test.csv')
```


```python
df = train_df.append(test_df)
```

## Data Analysis

###### SalePrice Distribution:


```python
saleprice = train_df['SalePrice']
plt.figure(figsize = (10,5)); plt.title = 'Normal Distribution'
sns.distplot(saleprice, fit=st.norm)
plt.figure(figsize = (10,5)); plt.title = 'log Distribution'
sns.distplot(saleprice, fit=st.lognorm)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc6b9b70>




![png](housepriceviz_files/housepriceviz_8_1.png)



![png](housepriceviz_files/housepriceviz_8_2.png)


- From the 2 ditributions above, Sale Price seems to be fit better with log distribution.
- There are possible outliers need to be remove

###### Missing Data Overview:

Lets take a look at missing data and the percentage of missing data


```python
Total = train_df.isnull().sum()
percent = np.round((((train_df.isnull().sum())/train_df.shape[0])*100),2)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent']).sort_values('Total', ascending=False)
missing_data[missing_data['Total']>0]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453</td>
      <td>99.52</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406</td>
      <td>96.30</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369</td>
      <td>93.77</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179</td>
      <td>80.75</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690</td>
      <td>47.26</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259</td>
      <td>17.74</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38</td>
      <td>2.60</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38</td>
      <td>2.60</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>8</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.loc[(train_df['Alley'].notnull()),['Alley']] = 1
train_df.loc[(train_df['Alley'].isnull()),['Alley']] = 0
sns.boxplot(x=train_df['Alley'], y=train_df['SalePrice'])
x = plt.xticks(rotation = 90)
```


![png](housepriceviz_files/housepriceviz_13_0.png)



```python
train_df.loc[(train_df['Fence'].notnull()),['Fence']] = 1
train_df.loc[(train_df['Fence'].isnull()),['Fence']] = 0
sns.boxplot(x=train_df['Fence'], y=train_df['SalePrice'])
x = plt.xticks(rotation = 90)
```


![png](housepriceviz_files/housepriceviz_14_0.png)



```python
train_df.loc[(train_df['Fence'].notnull()),['Fence']] = 1
train_df.loc[(train_df['Fence'].isnull()),['Fence']] = 0
sns.boxplot(x=train_df['Fence'], y=train_df['SalePrice'])
x = plt.xticks(rotation = 90)
```


![png](housepriceviz_files/housepriceviz_15_0.png)



```python
train_df.loc[(train_df['FireplaceQu'].isnull()),['FireplaceQu']] = 0
sns.boxplot(x=train_df['FireplaceQu'], y=train_df['SalePrice'])
x = plt.xticks(rotation = 90)
```


![png](housepriceviz_files/housepriceviz_16_0.png)


Combining the missing data table above with data dictionary, one can safely remove PoolQC, MiscFeature, 

###### Counting and Listing quantitative variables and qualitative variables:


```python
category_var = []
numeric_var = []
category_var = [column for column in train_df.columns if  train_df.dtypes[column] == 'object']
numeric_var = [column for column in train_df.columns if train_df.dtypes[column] != 'object']
```

- From data dictionary, we can see that MSSubClass, OverallQual and OverallCond are actually a categorical varible.
- In addition, we need to remove id and SalePrice columns


```python
category_var.append('MSSubClass')
category_var.append('OverallQual')
category_var.append('OverallCond')
numeric_var.remove('MSSubClass')
numeric_var.remove('OverallQual')
numeric_var.remove('OverallCond')
numeric_var.remove('SalePrice')
numeric_var.remove('Id')

print 'Number of category variable: ', len(category_var)
print 'Number of numeric variable: ', len(numeric_var)
```

    Number of category variable:  46
    Number of numeric variable:  33
    

###### Category Variable Analysis:

Convert catergory raw variable to pandas category type variable, this is for compatible purposes. Certain tasks works well if the data type is category


```python
for element in category_var:
    train_df[element] = train_df[element].astype('category')
    if train_df[element].isnull().any():
        train_df[element] = train_df[element].cat.add_categories(['MISSING'])
        train_df[element] = train_df[element].fillna('MISSING')
```

Boxplot for all category variables:


```python
def boxplot(x,y, **kawrgs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation = 90)

data = pd.melt(train_df, id_vars='SalePrice', value_vars=category_var)
g = sns.FacetGrid(data, col = 'variable', col_wrap=2, sharex= False, sharey= False)
g = g.map(boxplot, 'value', 'SalePrice')
```


![png](housepriceviz_files/housepriceviz_26_0.png)



```python
#Some Comments
```

Determine the significance of each category variables:


```python
def anova(frame):
    pvals = []
    anv = pd.DataFrame()
    anv['feature'] = category_var
    for col in category_var:
        samples = []
        for element in frame[col].unique():
            s = frame[frame[col] == element]['SalePrice'].values
            samples.append(s)
        pval = st.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

```


```python
a = anova(train_df)
a['disparity'] = np.log(1./a['pval'].values)
plt.figure(figsize=(15,8))
sns.barplot(data = a, x ='feature', y = 'disparity')
x = plt.xticks(rotation=90)
```

    C:\Anaconda2\lib\site-packages\scipy\stats\stats.py:2966: RuntimeWarning: invalid value encountered in double_scalars
      msb = ssbn / float(dfbn)
    C:\Anaconda2\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in divide
      
    


![png](housepriceviz_files/housepriceviz_30_1.png)


    This is ANOVA test, it is the Analysis Of Variance.
    This Chart basically showed the variance for each variables based SalePrice aka how much does the variable change the SalePrice
    From the Chart, you can see that Neighborhood is a significant factor which make sense since in  real estate, location is one of the most important factor
    List of variable candidate for removing:
        - Ultilities
        - LandSlope
        - Street
        - Condition2
        - MiscFeature
        - Heating
        - Functional
        - LotConfig
        - PoolQC
        - ExterCond


```python
#Ordering columns based on the significant
def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    
qual_encoded = []
for q in category_var:  
    encode(train_df, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)
def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(10, 0.25*len(features)))
    sns.barplot(data = spr, x = 'spearman', y= 'feature', orient='h')
features = numeric_var + qual_encoded
spearman(train_df, features)
```

    ['MSZoning_E', 'Street_E', 'Alley_E', 'LotShape_E', 'LandContour_E', 'Utilities_E', 'LotConfig_E', 'LandSlope_E', 'Neighborhood_E', 'Condition1_E', 'Condition2_E', 'BldgType_E', 'HouseStyle_E', 'RoofStyle_E', 'RoofMatl_E', 'Exterior1st_E', 'Exterior2nd_E', 'MasVnrType_E', 'ExterQual_E', 'ExterCond_E', 'Foundation_E', 'BsmtQual_E', 'BsmtCond_E', 'BsmtExposure_E', 'BsmtFinType1_E', 'BsmtFinType2_E', 'Heating_E', 'HeatingQC_E', 'CentralAir_E', 'Electrical_E', 'KitchenQual_E', 'Functional_E', 'FireplaceQu_E', 'GarageType_E', 'GarageFinish_E', 'GarageQual_E', 'GarageCond_E', 'PavedDrive_E', 'PoolQC_E', 'Fence_E', 'MiscFeature_E', 'SaleType_E', 'SaleCondition_E', 'MSSubClass_E', 'OverallQual_E', 'OverallCond_E']
    


![png](housepriceviz_files/housepriceviz_32_1.png)


Spearman distribution is used to determine the relation when they are not linear

###### Numeric Variable Analysis:

Corellation Matrix:


```python
corr = train_df[numeric_var + ['SalePrice']].corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1157fc5d0>




![png](housepriceviz_files/housepriceviz_36_1.png)


There are some correlation between SalePrice and numeric variables

###### Creating  boundary to see diffrence between variables and SalePrice


```python
boundary = np.exp(np.log(train_df['SalePrice']).median())
boundary
```




    163000.00000000012




```python
saleprice_low = train_df[train_df['SalePrice']<boundary]
saleprice_high = train_df[train_df['SalePrice']>boundary]
```


```python
data = pd.DataFrame()
data['feature'] = numeric_var
data['difference'] = [(saleprice_high[f].fillna(0.).mean() - saleprice_low[f].fillna(0.).mean())/(saleprice_low[f].fillna(0.).mean()) for f in numeric_var]
plt.figure(figsize=(10,10))
sns.barplot(data=data, x='feature', y='difference')
x=plt.xticks(rotation=90)
```


![png](housepriceviz_files/housepriceviz_41_0.png)



```python
data[data['difference']>1]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>MasVnrArea</td>
      <td>2.137880</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2ndFlrSF</td>
      <td>1.227779</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HalfBath</td>
      <td>1.360203</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Fireplaces</td>
      <td>1.579835</td>
    </tr>
    <tr>
      <th>24</th>
      <td>WoodDeckSF</td>
      <td>1.124169</td>
    </tr>
    <tr>
      <th>25</th>
      <td>OpenPorchSF</td>
      <td>1.572749</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3SsnPorch</td>
      <td>2.172595</td>
    </tr>
    <tr>
      <th>28</th>
      <td>ScreenPorch</td>
      <td>1.113397</td>
    </tr>
    <tr>
      <th>29</th>
      <td>PoolArea</td>
      <td>6.432280</td>
    </tr>
  </tbody>
</table>
</div>




```python
These features above are features that showed the biggest differences between SalePrice Boundary
```

List of Things need to be done:
    - SalePrice Log
    - Data Cleaning
    - Feature Engineer
    - Modeling
    - Model Stacking


```python

```


```python

```

