---
layout: post
title: "Housing Prices Prediction"
date: 2017-08-15
excerpt: "Perform Data Analysis on Kaggle House Price Prediction"
tags: [sample post, images, test]
comments: true
---
# 1. Import Libraries 


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
%matplotlib inline
```

# 2. Reading Data into DataFrames


```python
train_df = pd.read_csv('train.csv')
train_df.info()
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
    

# 3. Data Analysis

## 3.1. Overview

    We will breaks down the tasks into 3 Parts:
        1. General Analysis
        2. Categorical variable Analysis
        3. Numerical variable Analysis
        4. Conclusion with future tasks

The features of the house can be broke down to: Bedroom, Bathroom, Basement, Gardening/Lot Features, House's history, Location

## 3.2. General Analysis

### 3.2.1 Missing Data Analysis

Let's dive into the amount of missing data we have on hands:


```python
missing = train_df.isnull().sum()
missing = missing[missing >0]
missing.sort_values(inplace=True, ascending=False)
missing = pd.DataFrame(missing, columns=['datamissing'])
missing.reset_index(inplace=True)
plt.figure(figsize=(9,7))
missingviz = sns.barplot(y=missing['index'], x=missing['datamissing'], orient="h")
missingviz.set_title('Train Data')
```







![png]({{ site.url }}/assets/postimg/housepriceprediction/{{ site.url }}/assets/postimg/housepriceprediction/output_11_1.png)


It is important to note that eventhough it is missing data, most of the values listed as NA because those houses did not contain those features. For example, There are missing data in Garage related features, this is due to the fact that some houses do not have Garage

Most of these nulls values will be taken care of it in the Data Cleaning part of this blog which will be up in couple weeks

### 3.2.2 Houses and SalePrice Distribution


```python
plt.figure(figsize=(6, 6));plt.title('Normal')
sns.distplot(train_df.SalePrice,kde=False,fit=st.norm)
plt.figure(figsize=(6, 6));plt.title('Log')
sns.distplot(train_df.SalePrice,kde=False,fit=st.lognorm)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc565710>




![png]({{ site.url }}/assets/postimg/housepriceprediction/output_15_1.png)



![png]({{ site.url }}/assets/postimg/housepriceprediction/output_15_2.png)


The distribution plots above showed that it is best to take log of the SalePrice which will give us a smoother and more accurate distribution


```python
train_df['SalePrice'].quantile(0.99)
```




    442567.0100000005



Any Houses that are higher than the 99% of the distribution, would be candidates for outliers


```python
print 'Number of Houses that have SalePrice higher than 500k: ', train_df['SalePrice'][train_df['SalePrice']>500000].count()
```

    Number of Houses that have SalePrice higher than 500k:  9
    

    - These Houses potentially can be removed because these house seems to be outliers
    - Lets take a look at the distribution after removing the outliers


```python
no_outliers = np.log(train_df['SalePrice'][train_df['SalePrice']<500000])

plt.figure(figsize=(6, 6));plt.title('Log')
sns.distplot(no_outliers,kde=False,fit=st.norm)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xd1a0d68>




![png]({{ site.url }}/assets/postimg/housepriceprediction/output_21_1.png)


The distribution seems to fit better with log now

We now quickly look at the rest of Sale related features mainly:
    - SaleType
    - SaleCondition


##### SaleType

SaleType: Type of sale
		
       WD 	    Warranty Deed - Conventional
       CWD	    Warranty Deed - Cash
       VWD	    Warranty Deed - VA Loan
       New	    Home just constructed and sold
       COD	    Court Officer Deed/Estate
       Con	    Contract 15% Down payment regular terms
       ConLw	  Contract Low Down payment and low interest
       ConLI	  Contract Low Interest
       ConLD	  Contract Low Down
       Oth	    Other


```python
SaleType = pd.DataFrame((train_df['SaleType'].value_counts()))
SaleType['Percent'] = (SaleType['SaleType']/1460)*100
print SaleType
plt.figure(figsize=(7, 7));plt.title('SaleType')
SaleTypeviz = sns.boxplot(x=train_df['SalePrice'], y= train_df['SaleType'])
```

           SaleType    Percent
    WD         1267  86.780822
    New         122   8.356164
    COD          43   2.945205
    ConLD         9   0.616438
    ConLw         5   0.342466
    ConLI         5   0.342466
    CWD           4   0.273973
    Oth           3   0.205479
    Con           2   0.136986
    


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_26_1.png)


Many houses were bought as Warranty Deed - Conventional (87%). In addition, houses were bought with Cash have a slightly higher distribution in SalePrice

##### SaleCondition:

    SaleCondition: Condition of sale
       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)


```python
SaleCondition = pd.DataFrame((train_df['SaleCondition'].value_counts()))
SaleCondition['Percent'] = (SaleCondition.SaleCondition/1460)*100
print SaleCondition
plt.figure(figsize=(7, 7));plt.title('SaleCondition')
SaleTypeviz = sns.boxplot(x=train_df['SalePrice'], y= train_df['SaleCondition'])
```

             SaleCondition    Percent
    Normal            1198  82.054795
    Partial            125   8.561644
    Abnorml            101   6.917808
    Family              20   1.369863
    Alloca              12   0.821918
    AdjLand              4   0.273973
    


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_30_1.png)


When houses were with SaleCondition of Partial, the distribution seems to be higher than other SaleConditions

##### Sale related features Conclusion:

Tasks that need to be take care of when cleaning data is:
    - Use Log for SalePrice
    - Remove outliers from SalePrice


### 3.2.2 Bedroom Analysis

    We will take a look at BedroomAbvGr feature in detail
    First, let's visualize how SalePrice affected by BedroomAbvGr features


```python
BedroomAbvGr = pd.DataFrame((train_df['BedroomAbvGr'].value_counts()))
BedroomAbvGr['Percent'] = (BedroomAbvGr.BedroomAbvGr/1460)*100
print BedroomAbvGr
```

       BedroomAbvGr    Percent
    3           804  55.068493
    2           358  24.520548
    4           213  14.589041
    1            50   3.424658
    5            21   1.438356
    6             7   0.479452
    0             6   0.410959
    8             1   0.068493
    


```python
plt.figure(figsize=(7, 7));plt.title('# of Bedroom')
BedroomAbvGrviz = sns.boxplot(x=train_df['BedroomAbvGr'], y= train_df['SalePrice'])
```


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_37_0.png)


One unsual attribute here listed above is houses with no bedroom. How could it be? Lets investigate


```python
train_df[train_df['BedroomAbvGr']==0]
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
      <th>53</th>
      <td>54</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>50271</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>385000</td>
    </tr>
    <tr>
      <th>189</th>
      <td>190</td>
      <td>120</td>
      <td>RL</td>
      <td>41.0</td>
      <td>4923</td>
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
      <td>8</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>286000</td>
    </tr>
    <tr>
      <th>634</th>
      <td>635</td>
      <td>90</td>
      <td>RL</td>
      <td>64.0</td>
      <td>6979</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>600</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>144000</td>
    </tr>
    <tr>
      <th>1163</th>
      <td>1164</td>
      <td>90</td>
      <td>RL</td>
      <td>60.0</td>
      <td>12900</td>
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
      <td>1</td>
      <td>2008</td>
      <td>WD</td>
      <td>Alloca</td>
      <td>108959</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>1214</td>
      <td>80</td>
      <td>RL</td>
      <td>NaN</td>
      <td>10246</td>
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
      <td>5</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>145000</td>
    </tr>
    <tr>
      <th>1270</th>
      <td>1271</td>
      <td>40</td>
      <td>RL</td>
      <td>NaN</td>
      <td>23595</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>260000</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 81 columns</p>
</div>



There are 2 possiblities here:
    - These houses have bedrooms in the basement which the data did not account for
    - They are all from Low Density Residential Area

### 3.2.3 Bathroom Analysis

There are # bathroom Related Features: BsmtFullBath, BsmtHalfBath, FullBath, HalfBath

Lets find the true total Full Bath and Half Bath which including both basement and above ground


```python
train_df['TotFullBath'] = train_df['BsmtFullBath'] + train_df['FullBath']
train_df['TotHalfBath'] = train_df['BsmtHalfBath'] + train_df['HalfBath']
```


```python
train_df['TotFullBath'].value_counts()
```




    2    750
    1    371
    3    319
    4     18
    6      1
    0      1
    Name: TotFullBath, dtype: int64




```python
train_df['TotHalfBath'].value_counts()
```




    0    855
    1    572
    2     29
    3      3
    4      1
    Name: TotHalfBath, dtype: int64




```python
pd.crosstab(train_df['TotFullBath'],train_df['TotHalfBath'])
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
      <th>TotHalfBath</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>TotFullBath</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>228</td>
      <td>129</td>
      <td>12</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>443</td>
      <td>293</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>172</td>
      <td>144</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



From the crosstab, we can see that most houses have 0 to 1 HalfBath, and 1 to 3 FullBath

How's SalePrice related to # of Bathroom


```python
train_df['Full-Half Bath'] = train_df['TotFullBath'].map(str) + '-' + train_df['TotHalfBath'].map(str) 
```


```python
train_df['Full-Half Bath'].value_counts()
```




    2-0    443
    2-1    293
    1-0    228
    3-0    172
    3-1    144
    1-1    129
    2-2     14
    1-2     12
    4-0     11
    4-1      6
    3-2      2
    1-3      2
    3-3      1
    6-0      1
    4-2      1
    0-4      1
    Name: Full-Half Bath, dtype: int64




```python
plt.figure(figsize=(7, 7));plt.title('Bathroom - SalePrice')
bathroomGrviz = sns.boxplot(x=train_df['Full-Half Bath'], y= train_df['SalePrice'])
```


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_52_0.png)


From the boxplot, we can see that houses with increasing amount of full bath have higher SalePrice

Conclusion for Bathroom related features:
    - As the amount of both fullbath and halfbath increase, the saleprice also increase. However there are also exception, such as there are few houses with more than 4 bath room with a very average saleprice. These rows can consider to be outliers and can be remove.



### 3.2.4 Garage Analysis

    1. Houses with Garage vs Houses with no Garage


```python
train_df['HasGarage'] = train_df['GarageArea'].apply(lambda x: 1 if x>0 else 0)
sns.boxplot(x=train_df['HasGarage'], y= train_df['SalePrice'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc82fd30>




![png]({{ site.url }}/assets/postimg/housepriceprediction/output_57_1.png)


Houses with no garage have significantly lower SalePrice compare to houses with Garage

    2. Garage Area vs SalePrice


```python
sns.regplot(train_df['GarageArea'],train_df['SalePrice'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc8d2ba8>




![png]({{ site.url }}/assets/postimg/housepriceprediction/output_60_1.png)


If we remove some of the outliers we can see that as Garage Area increase, so will SalePrice

    3. Garage Condition vs Garage Quality


```python
pd.crosstab(train_df['GarageCond'],train_df['GarageQual'])
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
      <th>GarageQual</th>
      <th>Ex</th>
      <th>Fa</th>
      <th>Gd</th>
      <th>Po</th>
      <th>TA</th>
    </tr>
    <tr>
      <th>GarageCond</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ex</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Fa</th>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Gd</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Po</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TA</th>
      <td>1</td>
      <td>24</td>
      <td>10</td>
      <td>0</td>
      <td>1291</td>
    </tr>
  </tbody>
</table>
</div>



Most of the Garages are with Typical/Average for both Quality and Condition




```python

sns.boxplot(x=train_df['GarageCond'], y= train_df['SalePrice']).set_title('Condition')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_65_1.png)



```python
sns.boxplot(x=train_df['GarageQual'], y= train_df['SalePrice']).set_title('Quality')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_66_1.png)


There are not enough Garage Condition or Quality that are not typical to see meaningful trend. For xxample, even with excellence condition or quality, the saleprice are not any higher compare to the least desire values (Good)

    4. Garage Cars


```python
train_df['GarageCars'].value_counts()
```




    2    824
    1    369
    3    181
    0     81
    4      5
    Name: GarageCars, dtype: int64




```python
sns.boxplot(x=train_df['GarageCars'], y= train_df['SalePrice']).set_title('GarageCars')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_70_1.png)


It is clear that the more cars can store in the garage will give u higher SalePrice, with the exception of 4 cars garage, this is due to the fact that we do not have enough data point for 4 cars to see meaningful distribution for 4 cars.

    5. Garage Type


```python
train_df['GarageType'].value_counts()
```




    Attchd     870
    Detchd     387
    BuiltIn     88
    Basment     19
    CarPort      9
    2Types       6
    Name: GarageType, dtype: int64




```python
sns.boxplot(x=train_df['GarageType'], y= train_df['SalePrice']).set_title('GarageType')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_74_1.png)


There are a trend of increasing Saleprice with Garage Type, which goes CarPort --> Basement --> 2Types --> Attachd --> Detach --> Builtin

### 3.2.5 Utility and Options Analysis:

    1. Fireplace Analysis:


```python
pd.crosstab(train_df['FireplaceQu'], train_df['Fireplaces'])
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
      <th>Fireplaces</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ex</th>
      <td>19</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fa</th>
      <td>28</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Gd</th>
      <td>324</td>
      <td>54</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Po</th>
      <td>20</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TA</th>
      <td>259</td>
      <td>53</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Most Houses have only one Fireplace


```python
sns.boxplot(x=train_df['Fireplaces'], y= train_df['SalePrice']).set_title('Fireplaces')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_80_1.png)



```python
sns.boxplot(x=train_df['FireplaceQu'][train_df['Fireplaces']==1], y= train_df['SalePrice']).set_title('FireplaceQu')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_81_1.png)


From the boxplot, you can see that the quality does matter, as the quality of the fireplace goes up, the saleprices goes up

    2. Utilities


```python
train_df['Utilities'].value_counts()
```




    AllPub    1459
    NoSeWa       1
    Name: Utilities, dtype: int64



We dont have enough variety in Data point beside AllPub. This columns is a good candidate to be dropped from the data frame

    4. Central Air


```python
train_df['CentralAir'].value_counts()
```




    Y    1365
    N      95
    Name: CentralAir, dtype: int64




```python
sns.boxplot(x=train_df['CentralAir'], y= train_df['SalePrice']).set_title('CentralAir')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_88_1.png)


Houses with central air have higher Saleprice Distribution

    5. Electrical


```python
train_df['Electrical'].value_counts()
```




    SBrkr    1334
    FuseA      94
    FuseF      27
    FuseP       3
    Mix         1
    Name: Electrical, dtype: int64




```python
sns.boxplot(x=train_df['Electrical'], y= train_df['SalePrice']).set_title('Electrical')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_92_1.png)


It seems like there are no significant impact on SalePrice from Eletrical type

### 3.2.6 Outside of the House Features Analysis:

Some of the features that will be looked are: Alley, Fences, PavedDrive, Pool

   1. Alley

Lets replace Null values to No Alley


```python
train_df.loc[train_df['Alley'].isnull(),['Alley']] = 'NoAlley'
```


```python
sns.boxplot(x=train_df['Alley'], y= train_df['SalePrice']).set_title('Alley')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_99_1.png)


One thing to note that there is a different between Gravel and Pave Alley, but the distribution for No Alley is way too broad to draw any conclusion from it

    2. Fences


```python
train_df['Fence'][train_df['Fence'].isnull()] = 'NoFence'
train_df['Fence'].value_counts()
```

    C:\Anaconda2\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    




    NoFence    1179
    MnPrv       157
    GdPrv        59
    GdWo         54
    MnWw         11
    Name: Fence, dtype: int64




```python
sns.boxplot(x=train_df['Fence'], y= train_df['SalePrice']).set_title('Fence')
```









![png]({{ site.url }}/assets/postimg/housepriceprediction/output_103_1.png)


Convert Fences to either have Fences or Not


```python
train_df['Fence'][train_df['Fence']!= 'NoFence'] = 1
train_df['Fence'][train_df['Fence']== 'NoFence'] = 0

```

    C:\Anaconda2\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    C:\Anaconda2\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    


```python
sns.boxplot(x=train_df['Fence'], y= train_df['SalePrice']).set_title('Fence')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_106_1.png)


surprisingly, Houses with fences actually cost less than houses with no fences

    3. Paved Drive


```python
train_df['PavedDrive'].value_counts()
```




    Y    1340
    N      90
    P      30
    Name: PavedDrive, dtype: int64




```python
sns.boxplot(x=train_df['PavedDrive'], y= train_df['SalePrice']).set_title('PavedDrive')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_110_1.png)


Houses with no PavedDrive seems to have lower distribution compare to the rest

    4. Pool


```python
train_df['HasPool'] = train_df['PoolArea'].apply(lambda x: 1 if x>0 else 0)
```


```python
sns.boxplot(x=train_df['HasPool'], y= train_df['SalePrice']).set_title('HasPool')
```








![png]({{ site.url }}/assets/postimg/housepriceprediction/output_114_1.png)


It seems that house that with SalePrice higher than 150k would have a pool

### 3.2.7 Location Features Analysis:

As many may knows, in Real Estate Location is very important. Let's us analyze location features to help gain insight on this data set


```python
plt.figure(figsize=(10,15))
a = sns.boxplot(x=train_df['SalePrice'], y= train_df['Neighborhood']).set_title('Neighborhood')
```


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_118_0.png)


The boxplot seems to be all over the place but if you look closely, you can see that each neighborhood occupied a clear distribution from each other

  

##### Separating catergorical and numeric variable


```python
quantitive_var = [f for f in train_df.columns if train_df.dtypes[f] != 'object']
quantitive_var.remove('SalePrice')
quantitive_var.remove('Id')
qualitative_var = [f for f in train_df.columns if train_df.dtypes[f] == 'object']
```


```python
qualitative_var.append('MSSubClass')
qualitative_var.append('OverallQual')
qualitative_var.append('OverallCond')
quantitive_var.remove('MSSubClass')
quantitive_var.remove('OverallQual')
quantitive_var.remove('OverallCond')

```

## 3.3 Categorical Variables Analysis

#### Categorical Variable


```python
#Convert qualitative data to category data
for element in qualitative_var:
    train_df[element] =  train_df[element].astype('category')
    if train_df[element].isnull().any():
        train_df[element] = train_df[element].cat.add_categories(['MISSING'])
        train_df[element] = train_df[element].fillna('MISSING')
        
```

The reason to convert data type of columns to category are: Performance, certain function works the way they meant to be


```python
#Determine the significant of each categorical variable based on SalePrice
def anova(frame):
    pvals = []
    anv = pd.DataFrame()
    anv['feature'] = qualitative_var
    for column in qualitative_var:
        samples =[]
        for element in frame[column].unique():
            s = frame[frame[column]==element]['SalePrice'].values
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

    C:\Anaconda2\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in divide
      
    


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_129_1.png)


Analysis of Variance (ANOVA) is a statistical method used to test differences between two or more means. As the chart showed above, Neighborhood has the most variance in values based on SalePrice


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
for q in qualitative_var:  
    encode(train_df, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)
```

    ['MSZoning_E', 'Street_E', 'Alley_E', 'LotShape_E', 'LandContour_E', 'Utilities_E', 'LotConfig_E', 'LandSlope_E', 'Neighborhood_E', 'Condition1_E', 'Condition2_E', 'BldgType_E', 'HouseStyle_E', 'RoofStyle_E', 'RoofMatl_E', 'Exterior1st_E', 'Exterior2nd_E', 'MasVnrType_E', 'ExterQual_E', 'ExterCond_E', 'Foundation_E', 'BsmtQual_E', 'BsmtCond_E', 'BsmtExposure_E', 'BsmtFinType1_E', 'BsmtFinType2_E', 'Heating_E', 'HeatingQC_E', 'CentralAir_E', 'Electrical_E', 'KitchenQual_E', 'Functional_E', 'FireplaceQu_E', 'GarageType_E', 'GarageFinish_E', 'GarageQual_E', 'GarageCond_E', 'PavedDrive_E', 'PoolQC_E', 'Fence_E', 'MiscFeature_E', 'SaleType_E', 'SaleCondition_E', 'Full-Half Bath_E', 'MSSubClass_E', 'OverallQual_E', 'OverallCond_E']
    


```python
#Correlation
```


```python
def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(10, 0.25*len(features)))
    sns.barplot(data = spr, x = 'spearman', y= 'feature', orient='h')
features = quantitive_var + qual_encoded
spearman(train_df, features)
```


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_133_0.png)


## 3.4. Numerical Variable Analysis


```python
f = pd.melt(train_df,value_vars= quantitive_var)
g = sns.FacetGrid(f, col='variable', col_wrap=3, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
```


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_135_0.png)


###### There are some features that are good candidates for log distribution


```python
#Before taking log
list1 = ['GrLivArea','1stFlrSF','2ndFlrSF','TotalBsmtSF','LotArea','LotFrontage','LotFrontage','GarageArea']

f = pd.melt(train_df,value_vars= list1)
g = sns.FacetGrid(f, col='variable', col_wrap=3, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
```


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_137_0.png)



```python
#After Taking Log
train2 = train_df[list]
for i in list1:
    train2[i] = np.log1p(train2[i])

f = pd.melt(train2, value_vars= list1)
g = sns.FacetGrid(f, col='variable', col_wrap=3, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
```


![png]({{ site.url }}/assets/postimg/housepriceprediction/output_138_0.png)


As the result of Log transformation, the distribution looks betters.

#### Correlation Matrix


```python
corr = train_df[quantitive_var + ['SalePrice']].corr()
```


```python
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18bff898>




![png]({{ site.url }}/assets/postimg/housepriceprediction/output_142_1.png)



```python
corr['SalePrice'][corr['SalePrice']>0.2]
```




    LotFrontage     0.351799
    LotArea         0.263843
    YearBuilt       0.522897
    YearRemodAdd    0.507101
    MasVnrArea      0.477493
    BsmtFinSF1      0.386420
    BsmtUnfSF       0.214479
    TotalBsmtSF     0.613581
    1stFlrSF        0.605852
    2ndFlrSF        0.319334
    GrLivArea       0.708624
    BsmtFullBath    0.227122
    FullBath        0.560664
    HalfBath        0.284108
    TotRmsAbvGrd    0.533723
    Fireplaces      0.466929
    GarageYrBlt     0.486362
    GarageCars      0.640409
    GarageArea      0.623431
    WoodDeckSF      0.324413
    OpenPorchSF     0.315856
    TotFullBath     0.582934
    TotHalfBath     0.250628
    HasGarage       0.236832
    SalePrice       1.000000
    Name: SalePrice, dtype: float64



From the table above, we can see that there are some strong correlation between numerical variables and SalePrice. The next step of the project would be posted in several days.

Here are some examples of what a post with images might look like. If you want to display two or three images next to each other responsively use `figure` with the appropriate `class`. Each instance of `figure` is auto-numbered and displayed in the caption.

