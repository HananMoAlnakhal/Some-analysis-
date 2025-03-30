<body><div class="alert  alert-info" style="text-align:center">
<b>Adult_Money.csv</b><br>
 this project is made by <strong>Hanan Mo. Alnakhal</strong>
</div>

this Data is a very limited version of the original data set on UCI, I found it on my laptop during the 2023-2024 war <br>
I did this project before at 2021 but could not find it after<br>
So here I'm making a more profissional version

# **SetUp**
<div class="alert-block alert-info">
</div>

## imported libraries:


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

 ## imported data


```python
ds=pd.read_csv(r"adult.csv",na_values="?")
ds.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>educational-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>Private</td>
      <td>110713</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>Self-emp-not-inc</td>
      <td>225456</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43</td>
      <td>Private</td>
      <td>118308</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>1977</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>Private</td>
      <td>84619</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36</td>
      <td>Private</td>
      <td>447346</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>



## colors Dictionary


```python
colorsDict = {'acid green': '#8ffe09',
            'adobe': '#bd6c48',
            'algae': '#54ac68',
            'algae green': '#21c36f',
            'almost black': '#070d0d',
            'amber': '#feb308',
            'amethyst': '#9b5fc0',
            'apple': '#6ecb3c',}#there are more colors but deleted for the sake of making shorter report

```

# **Data Analysis**
<div class="alert-block alert-info">
</div>

## **EDA exploring**
---

### General info

#### getting row information: 


```python
ds.shape
```




    (3803, 15)




```python
ds.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3803 entries, 0 to 3802
    Data columns (total 15 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   age              3803 non-null   int64 
     1   workclass        3612 non-null   object
     2   fnlwgt           3803 non-null   int64 
     3   education        3803 non-null   object
     4   educational-num  3803 non-null   int64 
     5   marital-status   3803 non-null   object
     6   occupation       3610 non-null   object
     7   relationship     3803 non-null   object
     8   race             3803 non-null   object
     9   gender           3803 non-null   object
     10  capital-gain     3803 non-null   int64 
     11  capital-loss     3803 non-null   int64 
     12  hours-per-week   3803 non-null   int64 
     13  native-country   3741 non-null   object
     14  income           3803 non-null   object
    dtypes: int64(6), object(9)
    memory usage: 445.8+ KB
    


```python
ds.describe()
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
      <th>age</th>
      <th>fnlwgt</th>
      <th>educational-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3803.000000</td>
      <td>3803.000000</td>
      <td>3803.000000</td>
      <td>3803.000000</td>
      <td>3803.000000</td>
      <td>3803.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.075204</td>
      <td>190118.370497</td>
      <td>10.516697</td>
      <td>1878.522482</td>
      <td>115.553247</td>
      <td>41.729161</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.995897</td>
      <td>104600.743468</td>
      <td>2.633380</td>
      <td>9998.401297</td>
      <td>462.844230</td>
      <td>12.293716</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>20507.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>119199.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>178383.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>236515.000000</td>
      <td>13.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>981628.000000</td>
      <td>16.000000</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
ds[['age','educational-num','fnlwgt',"capital-gain","capital-loss","hours-per-week"]].corr()
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
      <th>age</th>
      <th>educational-num</th>
      <th>fnlwgt</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>0.098759</td>
      <td>-0.079115</td>
      <td>0.073901</td>
      <td>0.069046</td>
      <td>0.071616</td>
    </tr>
    <tr>
      <th>educational-num</th>
      <td>0.098759</td>
      <td>1.000000</td>
      <td>-0.020601</td>
      <td>0.137037</td>
      <td>0.088748</td>
      <td>0.193257</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>-0.079115</td>
      <td>-0.020601</td>
      <td>1.000000</td>
      <td>-0.002690</td>
      <td>-0.002966</td>
      <td>0.003846</td>
    </tr>
    <tr>
      <th>capital-gain</th>
      <td>0.073901</td>
      <td>0.137037</td>
      <td>-0.002690</td>
      <td>1.000000</td>
      <td>-0.046919</td>
      <td>0.084970</td>
    </tr>
    <tr>
      <th>capital-loss</th>
      <td>0.069046</td>
      <td>0.088748</td>
      <td>-0.002966</td>
      <td>-0.046919</td>
      <td>1.000000</td>
      <td>0.034786</td>
    </tr>
    <tr>
      <th>hours-per-week</th>
      <td>0.071616</td>
      <td>0.193257</td>
      <td>0.003846</td>
      <td>0.084970</td>
      <td>0.034786</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("parcentage-> work more than 50 hours per week:",ds[ds["hours-per-week"]>50]["hours-per-week"].count()/ds["hours-per-week"].count()*100)
print("parcentage-> work more than 70 hours per week:",ds[ds["hours-per-week"]>70]["hours-per-week"].count()/ds["hours-per-week"].count()*100)
print("parcentage-> work more than 80 hours per week: %f"%(ds[ds["hours-per-week"]>80]["hours-per-week"].count()/ds["hours-per-week"].count()*100))
```

    parcentage-> work more than 50 hours per week: 13.200105180120957
    parcentage-> work more than 70 hours per week: 1.8406521167499343
    parcentage-> work more than 80 hours per week: 0.631081
    

#### plots for distributions and relations: 


```python
ds.hist(color='darkred');
plt.subplots_adjust(wspace=.5, hspace=.5);
```


    
![png](output_19_0.png)
    



```python
sns.set_style('whitegrid')
sns.pairplot(ds[(ds.age >20)&(ds.age <80)],diag_kind="kde",plot_kws={'alpha': 0.2});
```


    
![png](output_20_0.png)
    


### Missing-Data:


```python
missing=pd.DataFrame({"missing":ds.isnull().sum()})
missing[missing['missing']>0].style.background_gradient(cmap="Reds",subset="missing")
```




<style type="text/css">
#T_f70ec_row0_col0 {
  background-color: #6d010e;
  color: #f1f1f1;
}
#T_f70ec_row1_col0 {
  background-color: #67000d;
  color: #f1f1f1;
}
#T_f70ec_row2_col0 {
  background-color: #fff5f0;
  color: #000000;
}
</style>
<table id="T_f70ec">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f70ec_level0_col0" class="col_heading level0 col0" >missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f70ec_level0_row0" class="row_heading level0 row0" >workclass</th>
      <td id="T_f70ec_row0_col0" class="data row0 col0" >191</td>
    </tr>
    <tr>
      <th id="T_f70ec_level0_row1" class="row_heading level0 row1" >occupation</th>
      <td id="T_f70ec_row1_col0" class="data row1 col0" >193</td>
    </tr>
    <tr>
      <th id="T_f70ec_level0_row2" class="row_heading level0 row2" >native-country</th>
      <td id="T_f70ec_row2_col0" class="data row2 col0" >62</td>
    </tr>
  </tbody>
</table>




### Outliers or Noisy Data:
- as we can see in the following plot,there is not any


```python
ds[['age','hours-per-week']].boxplot();
```


    
![png](output_24_0.png)
    


### 
---
> ## **Notices:**
> - **there is 3802 record and 15 columns**
> - ### ages are between 17-90
>    - most ages are between(30-49)
> - ### most of the recoreds has got post hightSchool education<br>(high-scoole educational number is -->9)
>    - the ppl who never got to be in school are only 4 ppl none of them gets >50K yearly
> - ### ppl work in an average of (41.7 hours/week)
>    - 75% of ppl work 48 hours or less
>    - only 24% work more than 49 hours
>    - less than 2% work more than 70 hours

> - ### **there are missing data in 3 columns `('workclass','occupation','native-country')`**
> - ## **Droping:**
>   - records: that has missing Data --> `occupation`
>   - columns: 
>       - `fnlwgt` + `capital-loss` +`capital-gain`--> doesn't effict the general "income"
>       - `education` --> is the same as educational-num

> - ## **Main columns:**
>   - Target column: **`income`**
>   - **`age | gender | race | educational-num | workclass | occupation | marital-status | relationship | hours-per-week | native-country`**
>   - `marital-status | relationship`   --> must chose one of them
>   - `workclass | occupation`          --> diciding which one is more important
>   - 
        

## **Data Cleaning:**
---
- dealing with missing Data since there is no Outliers

### Droping:

#### deleting empty records
- `occupation` is important and can't be filled manually + I noteced since the occupation is just the `workclass` but more spicific 
that means that if i droped the records that has missing `occupation` the `workclass` missing would also be droped automatically
- the `native-country` column also have missing values so it will be analised later to fill it with an apropriet values


```python
null_index=ds["occupation"].isnull()
ds.drop(ds.loc[null_index].index,inplace=True)
```


```python
missing=pd.DataFrame({"missing":ds.isnull().sum()})
missing[missing['missing']>0].style.background_gradient(cmap="Reds",subset="missing")
# by deleting the record that miss the occupation values we deleted the workclass that olso have missing values
```




<style type="text/css">
#T_754fb_row0_col0 {
  background-color: #fff5f0;
  color: #000000;
}
</style>
<table id="T_754fb">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_754fb_level0_col0" class="col_heading level0 col0" >missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_754fb_level0_row0" class="row_heading level0 row0" >native-country</th>
      <td id="T_754fb_row0_col0" class="data row0 col0" >60</td>
    </tr>
  </tbody>
</table>




#### deleting unimportant columns


```python
ds.drop(["fnlwgt","capital-gain","capital-loss","education"],axis=1,inplace=True)
ds.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>educational-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>Private</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>Self-emp-not-inc</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43</td>
      <td>Private</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>Private</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36</td>
      <td>Private</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>



### filling missing Data at `native-country` :



```python
Country=ds['native-country'].value_counts()
Country.sort_values(ascending=False).head(11).sort_values(ascending=True).plot(kind="barh")
plt.title("native-country count");
```


    
![png](output_34_0.png)
    


#### filling missing with "United-States":
- since most of the records are from there as we can see in the previous plot


```python
before = ds.loc[ds['native-country'].isnull()].head()
before
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
      <th>age</th>
      <th>workclass</th>
      <th>educational-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>146</th>
      <td>40</td>
      <td>Self-emp-inc</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>40</td>
      <td>NaN</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>244</th>
      <td>50</td>
      <td>Private</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>NaN</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>277</th>
      <td>54</td>
      <td>Self-emp-inc</td>
      <td>16</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>NaN</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>291</th>
      <td>46</td>
      <td>Self-emp-inc</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>42</td>
      <td>NaN</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>299</th>
      <td>42</td>
      <td>Private</td>
      <td>16</td>
      <td>Married-spouse-absent</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>60</td>
      <td>NaN</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
ds['native-country'].fillna("United-States",inplace=True)
missing=pd.DataFrame({"missing":ds.isnull().sum()})
missing[missing['missing']>0].style.background_gradient(cmap="Reds",subset="missing")
```




<style type="text/css">
</style>
<table id="T_04f74">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_04f74_level0_col0" class="col_heading level0 col0" >missing</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>




## **income and hours-per-week :**
---

> there is a positive relationship: **work more get more**


```python
axs=plt.figure(figsize=(20, 8)).subplot_mosaic("""AABBB""")
HperWeek=ds.groupby(['hours-per-week', 'income']).size().unstack().fillna(0)
HperWeek['rate']=HperWeek['>50K']/(HperWeek['>50K']+HperWeek['<=50K'])
mask=(HperWeek['rate']<1)&(HperWeek['rate']>0)
masked=HperWeek[mask]

axs["A"].scatter(x=HperWeek.index,y=HperWeek['rate'],color="black",edgecolors='white')# With outliers
axs["B"].scatter(x=masked.index, y=masked['rate'], color='red',edgecolors='black')# Without outliers
axs['A'].set_ylabel('rate')
axs['A'].set_xlabel('hours-per-week')
axs['A'].set_title('rate to the hours-per-week with Outliers')
axs['B'].set_ylabel('rate')
axs['B'].set_xlabel('hours-per-week')
axs['B'].set_title('leanier positive relationship - without outliers ')
plt.plot(np.arange(10,80),np.arange(15,85)/110,color='red');


```


    
![png](output_40_0.png)
    



```python
colors=['#67000d','#32DBF0']
axs=plt.figure(figsize=(15, 10)).subplot_mosaic("""AABB
                                   AABB
                                   AABB""")
pplWhoWorkMore=ds[ds['hours-per-week']>50]["income"].value_counts()
mask=(ds['hours-per-week']<50)&(ds['hours-per-week']>30)
avarageWorkH=ds[mask]["income"].value_counts()
axs["A"].pie(pplWhoWorkMore,labels=pplWhoWorkMore.index,colors=colors,shadow=True,autopct='%1.1f%%')
axs["A"].set_title("individuals who work more than 50h/week")
axs['A'].add_artist(plt.Circle((0, 0), 0.67, color='white'));
axs["B"].pie(avarageWorkH,labels=avarageWorkH.index,colors=['#32DBF0','black'],counterclock=False,shadow=True,autopct='%1.1f%%')
axs["B"].set_title("individuals who work (30-50)h");
axs['B'].add_artist(plt.Circle((0, 0), 0.65, color='white'));
```


    
![png](output_41_0.png)
    


## **income and age :**
---

- positive relationship :
        - there is a clear positive relationship as you can see the crean curves up 


```python
# plt.scatter(ds.age, ds.income, color="#272727", edgecolor='black');
axs=plt.figure(figsize=(20, 15)).subplot_mosaic("""AABB
                                                  CCDD""");
ageRelated=ds.groupby(['age', 'income']).size().unstack()
ageRelated.fillna(0,inplace=True)
ageRelated['rate']=ageRelated['>50K']/(ageRelated['>50K']+ageRelated['<=50K'])
axs['A'].plot(ageRelated.loc['17':'61'][['>50K',"<=50K"]])
axs['A'].legend(title='income',labels=['>50K','<=50K'])
axs["A"].set_title("row counts in the dataset")
axs['B'].plot(ageRelated.loc['17':'61']['rate'],color='lightgreen');
axs["B"].set_title("The percentag of getting >50k as we grow up")
axs['C'].plot(ageRelated.loc['30':'55'][['>50K','<=50K']])
axs['C'].legend(title='income',labels=['>50K','<=50K'])
axs["C"].set_title("Zomed-In counts at the middle age")
axs['D'].plot(ageRelated.loc['30':'55']['rate'],color='lightgreen');
axs["D"].set_title("The percentag of getting >50k for ages between 30 and 55");

```


    
![png](output_44_0.png)
    


## **gender :**
---


```python
GenderDs=ds.groupby(['gender','income',]).size().unstack()
```


```python
fig, axs = plt.subplots(2, 2, figsize=(10, 15))

axs[0, 1].pie(GenderDs.loc['Female',['>50K','<=50K']],labels=[">50K", "<=50K"], autopct='%1.1f%%', colors=['hotpink','skyblue'])
axs[0, 1].set_title("Femals income")
axs[0, 0].pie(GenderDs.loc['Male',['>50K','<=50K']],labels=[">50K", "<=50K"], autopct='%1.1f%%', colors=['hotpink','skyblue'])
axs[0, 0].set_title("Males income")
axs[1, 0].pie( ds['gender'].value_counts(),labels=["Male", "Femal"], autopct='%1.1f%%', colors=['skyblue','pink'])
axs[1, 0].set_title("Male-Female in dataSet")
axs[1, 1].remove()
axs[1, 0].set_position([0.3, 0.3, 0.4, 0.3])
```


    
![png](output_47_0.png)
    



```python
# [['gender','age',"educational-num","hours-per-week"]]
        # |
#         ^

GenderDs=ds[['gender','age',"educational-num","hours-per-week"]].groupby('gender').mean()
avarageWorkH=ds["hours-per-week"].mean()
GenderDs.columns.name="mean||"
GenderDs

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
      <th>mean||</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>37.525284</td>
      <td>10.428277</td>
      <td>37.316821</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>40.761833</td>
      <td>10.636123</td>
      <td>44.079137</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('-'*50)
print('women get less education than men at average')
print('-'*50)
print('average work hours for records in data set regardless of gender:',avarageWorkH)
print('delta averageWorkHours and the women',GenderDs.loc["Female"]["hours-per-week"]- avarageWorkH)
print('delta averageWorkHours and the men',GenderDs.loc["Male"]["hours-per-week"]- avarageWorkH)
print('-'*50)
print('women work less than average,while men work more')
```

    --------------------------------------------------
    women get less education than men at average
    --------------------------------------------------
    average work hours for records in data set regardless of gender: 42.26398891966759
    delta averageWorkHours and the women -4.947167454239313
    delta averageWorkHours and the men 1.815147770979891
    --------------------------------------------------
    women work less than average,while men work more
    

### **Top occupation based on gender :**


```python
FeOcc=ds[(ds['gender']=='Female')&(ds['income']=='>50K')]['occupation'].value_counts()
FeOcc
```




    Prof-specialty       83
    Adm-clerical         60
    Exec-managerial      60
    Sales                15
    Other-service        11
    Tech-support          8
    Machine-op-inspct     4
    Protective-serv       3
    Transport-moving      2
    Handlers-cleaners     1
    Craft-repair          1
    Name: occupation, dtype: int64




```python
MaOcc=ds[(ds['gender']=='Male')&(ds['income']=='>50K')]['occupation'].value_counts()
MaOcc
```




    Exec-managerial      390
    Prof-specialty       325
    Craft-repair         201
    Sales                184
    Adm-clerical          69
    Transport-moving      66
    Machine-op-inspct     57
    Tech-support          55
    Protective-serv       53
    Farming-fishing       27
    Other-service         19
    Handlers-cleaners     16
    Name: occupation, dtype: int64



### 
> - only 26% of the data-set is **Females**,while the rest are males
> - more than 55% of the males get more than 50k yearly
> - only 25.6% of Females get more than 50k yearly



> average work hours for records in data set regardless of gender: `42.26398891966759`<br>
> delta averageWorkHours and the women `-4.947167454239313`<br>
> delta averageWorkHours and the men `1.815147770979891`<br>
> women get less education than men at average<br>
> women work less than average,while men work more



## **race:**
---

> - **Asian-Pac-Islander :**
>   - get on average more eductaion than high school, get **"Assoc-voc"** on average
>       - has **50%** chance of getting more than 50K yearly
> - **white :**
>   - get on average more eductaion than high school, get **"some-colllage"** on average
>   - has **49%** chance of getting more than 50K yearly


```python
colors = ['#FF0181','#272727','#FAE100' , '#900DFF',"#32DBF0"]
ds['race'].value_counts().plot(kind='pie',colors = colors);
```


    
![png](output_56_0.png)
    



```python
RaceSta=ds.groupby(['race', 'income']).size().unstack().fillna(0)
RaceSta['rate']=RaceSta['>50K']/(RaceSta['>50K']+RaceSta['<=50K'])
RaceSta2=ds.groupby('race')[['age',"educational-num","hours-per-week"]].mean()
RaceSta=RaceSta.join(RaceSta2)
RaceSta
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>rate</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
    </tr>
    <tr>
      <th>race</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Amer-Indian-Eskimo</th>
      <td>20</td>
      <td>6</td>
      <td>0.230769</td>
      <td>34.884615</td>
      <td>9.115385</td>
      <td>40.307692</td>
    </tr>
    <tr>
      <th>Asian-Pac-Islander</th>
      <td>51</td>
      <td>53</td>
      <td>0.509615</td>
      <td>37.096154</td>
      <td>11.346154</td>
      <td>40.750000</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>210</td>
      <td>83</td>
      <td>0.283276</td>
      <td>37.839590</td>
      <td>9.750853</td>
      <td>40.221843</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>15</td>
      <td>4</td>
      <td>0.210526</td>
      <td>32.947368</td>
      <td>9.315789</td>
      <td>40.210526</td>
    </tr>
    <tr>
      <th>White</th>
      <td>1604</td>
      <td>1564</td>
      <td>0.493687</td>
      <td>40.257576</td>
      <td>10.651515</td>
      <td>42.530934</td>
    </tr>
  </tbody>
</table>
</div>




```python
RaceSta['rate'].sort_values().plot(kind='bar',color=colors)
plt.xticks(rotation=0);
```


    
![png](output_58_0.png)
    


## **education:**
---

> - The more education you have, give you a real chance up to 70% to get more than 50k yearly
> - most ppl in the data-set has got education up to high-school grade, only 30% of them gets >50k yearly
> - 80% of ppl who have a Doctorate degree gets >50k a year


```python
education={
     1:'Preschool',
     2:'1st-4th',
     3:'5th-6th',
     4:'7th-8th',
     5:'9th',
     6:'10th',
     7:'11th',
     8:'12th',
     9:'HS-grad',
     10:'Some-college',
     11:'Assoc-voc',
     12:'Assoc-acdm',
     13:'Bachelors',
     14:'Masters',
     15:'Prof-school',
     16:'Doctorate'}
```


```python
ds['educational-num']=ds['educational-num'].apply(lambda x:education[x])
ds['educational-num'].value_counts().sort_values().plot(kind='bar');
plt.title('education for the indiviuals in the dataset')
plt.xlabel("")
```




    Text(0.5, 0, '')




    
![png](output_62_1.png)
    



```python

edu=ds.groupby('educational-num').mean().join(ds.groupby(['educational-num','income']).size().unstack().fillna(0))
edu["rate"]=edu['>50K']/(edu['<=50K']+edu['>50K'])
edu.loc[list(education.values())]['rate'].plot.bar();
plt.title('the percentage of getting >50k yearly based on education level')
plt.xlabel("");
```

    C:\Users\Star\AppData\Local\Temp\ipykernel_904\957521756.py:1: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      edu=ds.groupby('educational-num').mean().join(ds.groupby(['educational-num','income']).size().unstack().fillna(0))
    


    
![png](output_63_1.png)
    



```python
def coding(x):
    dict={}
    for i,name in tuple(education.items()):
        dict[name]=i
    return dict[x]
ds['educational-num']=ds['educational-num'].apply(lambda x: coding(x))
```

## **work class -> occupation** ?
---

- ***occupation is more important*** than workclass if had to chose between them occupation is much detalid and has more classes than workclass classes
- but I decided to choose **both** when making the ML modul


```python
ds['occupation'].value_counts()
```




    Exec-managerial      625
    Prof-specialty       604
    Craft-repair         460
    Sales                416
    Adm-clerical         374
    Other-service        281
    Machine-op-inspct    201
    Transport-moving     181
    Handlers-cleaners    135
    Tech-support         125
    Farming-fishing      103
    Protective-serv       98
    Priv-house-serv        6
    Armed-Forces           1
    Name: occupation, dtype: int64




```python
ds['workclass'].value_counts()
```




    Private             2579
    Self-emp-not-inc     302
    Local-gov            260
    State-gov            178
    Self-emp-inc         161
    Federal-gov          129
    Without-pay            1
    Name: workclass, dtype: int64




```python
fig=plt.figure(figsize=(15,20))
plt.subplot(1,2,1)
ds['occupation'].value_counts().head(13).plot.pie(autopct='%1.1f%%',colors=list(colorsDict.values())[496:501])
plt.title('occupation');
plt.subplot(1,2,2)
ds['workclass'].value_counts().plot.pie(autopct='%1.1f%%',colors=list(colorsDict.values())[496:501])
plt.title('workclass');
```


    
![png](output_69_0.png)
    



```python
occ=ds.groupby(['occupation','income']).size().unstack().fillna(0).join(ds[['age','educational-num','hours-per-week','occupation']].groupby('occupation').mean())
occ['rate']=occ['>50K']/(occ['>50K']+occ['<=50K'])
workC=ds.groupby(['workclass','income']).size().unstack().fillna(0).join(ds[['age','educational-num','hours-per-week','workclass']].groupby('workclass').mean())
workC['rate']=workC['>50K']/(workC['>50K']+workC['<=50K'])
fig=plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
occ['rate'].sort_values().plot.bar()
plt.title('occupation')
plt.xlabel('')
plt.xticks(rotation=70)
plt.subplot(1,2,2)
workC['rate'].sort_values().plot.bar()
plt.title('workclass')
plt.xlabel('');
plt.xticks(rotation=30);
```


    
![png](output_70_0.png)
    


### **occupation**
---


```python
occ.columns.name='mean'
occ=occ.sort_values(by='rate',ascending=False)
occ
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
      <th>mean</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Exec-managerial</th>
      <td>175.0</td>
      <td>450.0</td>
      <td>43.051200</td>
      <td>11.806400</td>
      <td>45.996800</td>
      <td>0.720000</td>
    </tr>
    <tr>
      <th>Prof-specialty</th>
      <td>196.0</td>
      <td>408.0</td>
      <td>42.201987</td>
      <td>13.223510</td>
      <td>43.769868</td>
      <td>0.675497</td>
    </tr>
    <tr>
      <th>Protective-serv</th>
      <td>42.0</td>
      <td>56.0</td>
      <td>39.357143</td>
      <td>10.561224</td>
      <td>42.908163</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>Tech-support</th>
      <td>62.0</td>
      <td>63.0</td>
      <td>39.440000</td>
      <td>11.264000</td>
      <td>40.112000</td>
      <td>0.504000</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>217.0</td>
      <td>199.0</td>
      <td>39.692308</td>
      <td>10.622596</td>
      <td>42.548077</td>
      <td>0.478365</td>
    </tr>
    <tr>
      <th>Craft-repair</th>
      <td>258.0</td>
      <td>202.0</td>
      <td>40.415217</td>
      <td>9.163043</td>
      <td>43.073913</td>
      <td>0.439130</td>
    </tr>
    <tr>
      <th>Transport-moving</th>
      <td>113.0</td>
      <td>68.0</td>
      <td>39.204420</td>
      <td>8.955801</td>
      <td>44.624309</td>
      <td>0.375691</td>
    </tr>
    <tr>
      <th>Adm-clerical</th>
      <td>245.0</td>
      <td>129.0</td>
      <td>38.748663</td>
      <td>10.275401</td>
      <td>38.254011</td>
      <td>0.344920</td>
    </tr>
    <tr>
      <th>Machine-op-inspct</th>
      <td>140.0</td>
      <td>61.0</td>
      <td>37.487562</td>
      <td>8.641791</td>
      <td>41.069652</td>
      <td>0.303483</td>
    </tr>
    <tr>
      <th>Farming-fishing</th>
      <td>76.0</td>
      <td>27.0</td>
      <td>42.893204</td>
      <td>8.844660</td>
      <td>47.854369</td>
      <td>0.262136</td>
    </tr>
    <tr>
      <th>Handlers-cleaners</th>
      <td>118.0</td>
      <td>17.0</td>
      <td>32.807407</td>
      <td>8.466667</td>
      <td>37.244444</td>
      <td>0.125926</td>
    </tr>
    <tr>
      <th>Other-service</th>
      <td>251.0</td>
      <td>30.0</td>
      <td>34.085409</td>
      <td>8.725979</td>
      <td>35.220641</td>
      <td>0.106762</td>
    </tr>
    <tr>
      <th>Armed-Forces</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>23.000000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Priv-house-serv</th>
      <td>6.0</td>
      <td>0.0</td>
      <td>26.000000</td>
      <td>6.000000</td>
      <td>25.666667</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
occ['hours-per-week'].sort_values().plot.bar();plt.title('Job and average hours-per-week spent');
```


    
![png](output_73_0.png)
    


#### gender-occupation income:


```python
occF=ds[ds['gender']=='Female'].groupby(['occupation','income']).size().unstack().fillna(0).join(ds[ds['gender']=='Female'][['age','educational-num','hours-per-week','occupation']].groupby('occupation').mean())
occF['rate']=occF['>50K']/(occF['>50K'] + occF['<=50K'])
occF=occF.sort_values(by='rate',ascending=False)
occM=ds[ds['gender']=='Male'].groupby(['occupation','income']).size().unstack().fillna(0).join(ds[ds['gender']=='Male'][['age','educational-num','hours-per-week','occupation']].groupby('occupation').mean())
occM['rate']=occM['>50K']/(occM['>50K'] + occM['<=50K'])
occM=occM.sort_values(by='rate',ascending=False)
fig=plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
occM['rate'].sort_values().plot.bar(color='skyblue')
plt.title('Rate for Male')
plt.xlabel('')
plt.xticks(rotation=70)
plt.subplot(1,2,2)
occF['rate'].sort_values().plot.bar(color='hotpink')
plt.title('Rate for Female')
plt.xlabel('');
plt.xticks(rotation=70);

```


    
![png](output_75_0.png)
    



```python
occF
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Prof-specialty</th>
      <td>83.0</td>
      <td>83.0</td>
      <td>39.686747</td>
      <td>12.993976</td>
      <td>39.674699</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>Exec-managerial</th>
      <td>80.0</td>
      <td>60.0</td>
      <td>41.614286</td>
      <td>11.407143</td>
      <td>42.150000</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>Adm-clerical</th>
      <td>185.0</td>
      <td>60.0</td>
      <td>38.514286</td>
      <td>10.130612</td>
      <td>37.171429</td>
      <td>0.244898</td>
    </tr>
    <tr>
      <th>Protective-serv</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>34.076923</td>
      <td>10.307692</td>
      <td>39.923077</td>
      <td>0.230769</td>
    </tr>
    <tr>
      <th>Tech-support</th>
      <td>27.0</td>
      <td>8.0</td>
      <td>36.400000</td>
      <td>11.000000</td>
      <td>39.142857</td>
      <td>0.228571</td>
    </tr>
    <tr>
      <th>Transport-moving</th>
      <td>12.0</td>
      <td>2.0</td>
      <td>33.714286</td>
      <td>9.642857</td>
      <td>39.071429</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>101.0</td>
      <td>15.0</td>
      <td>33.163793</td>
      <td>9.913793</td>
      <td>34.612069</td>
      <td>0.129310</td>
    </tr>
    <tr>
      <th>Machine-op-inspct</th>
      <td>45.0</td>
      <td>4.0</td>
      <td>35.061224</td>
      <td>8.428571</td>
      <td>36.244898</td>
      <td>0.081633</td>
    </tr>
    <tr>
      <th>Other-service</th>
      <td>137.0</td>
      <td>11.0</td>
      <td>34.885135</td>
      <td>8.763514</td>
      <td>32.837838</td>
      <td>0.074324</td>
    </tr>
    <tr>
      <th>Handlers-cleaners</th>
      <td>15.0</td>
      <td>1.0</td>
      <td>39.125000</td>
      <td>8.062500</td>
      <td>31.312500</td>
      <td>0.062500</td>
    </tr>
    <tr>
      <th>Craft-repair</th>
      <td>18.0</td>
      <td>1.0</td>
      <td>36.157895</td>
      <td>9.105263</td>
      <td>39.947368</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>Farming-fishing</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>47.666667</td>
      <td>7.666667</td>
      <td>31.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Priv-house-serv</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>27.800000</td>
      <td>6.000000</td>
      <td>24.800000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
occM
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Exec-managerial</th>
      <td>95.0</td>
      <td>390.0</td>
      <td>43.465979</td>
      <td>11.921649</td>
      <td>47.107216</td>
      <td>0.804124</td>
    </tr>
    <tr>
      <th>Prof-specialty</th>
      <td>113.0</td>
      <td>325.0</td>
      <td>43.155251</td>
      <td>13.310502</td>
      <td>45.321918</td>
      <td>0.742009</td>
    </tr>
    <tr>
      <th>Protective-serv</th>
      <td>32.0</td>
      <td>53.0</td>
      <td>40.164706</td>
      <td>10.600000</td>
      <td>43.364706</td>
      <td>0.623529</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>116.0</td>
      <td>184.0</td>
      <td>42.216667</td>
      <td>10.896667</td>
      <td>45.616667</td>
      <td>0.613333</td>
    </tr>
    <tr>
      <th>Tech-support</th>
      <td>35.0</td>
      <td>55.0</td>
      <td>40.622222</td>
      <td>11.366667</td>
      <td>40.488889</td>
      <td>0.611111</td>
    </tr>
    <tr>
      <th>Adm-clerical</th>
      <td>60.0</td>
      <td>69.0</td>
      <td>39.193798</td>
      <td>10.550388</td>
      <td>40.310078</td>
      <td>0.534884</td>
    </tr>
    <tr>
      <th>Craft-repair</th>
      <td>240.0</td>
      <td>201.0</td>
      <td>40.598639</td>
      <td>9.165533</td>
      <td>43.208617</td>
      <td>0.455782</td>
    </tr>
    <tr>
      <th>Transport-moving</th>
      <td>101.0</td>
      <td>66.0</td>
      <td>39.664671</td>
      <td>8.898204</td>
      <td>45.089820</td>
      <td>0.395210</td>
    </tr>
    <tr>
      <th>Machine-op-inspct</th>
      <td>95.0</td>
      <td>57.0</td>
      <td>38.269737</td>
      <td>8.710526</td>
      <td>42.625000</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>Farming-fishing</th>
      <td>73.0</td>
      <td>27.0</td>
      <td>42.750000</td>
      <td>8.880000</td>
      <td>48.340000</td>
      <td>0.270000</td>
    </tr>
    <tr>
      <th>Other-service</th>
      <td>114.0</td>
      <td>19.0</td>
      <td>33.195489</td>
      <td>8.684211</td>
      <td>37.872180</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>Handlers-cleaners</th>
      <td>103.0</td>
      <td>16.0</td>
      <td>31.957983</td>
      <td>8.521008</td>
      <td>38.042017</td>
      <td>0.134454</td>
    </tr>
    <tr>
      <th>Armed-Forces</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>23.000000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Priv-house-serv</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>17.000000</td>
      <td>6.000000</td>
      <td>30.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



###
> ### **outComes:**
>   **note:***this info is not generlized and it's just based on the current data*
>- **Exec-managerial** jobs has the highest rate of getting more then 50k yearly which is **72%** :
>   - ppl who work in this occupation has the seconed highst education gaind (they went to collage or gaind Assoc-voc or Assoc-acdm)
>   - have the highest age average between all of the occupations (43years)
>   - have the second highest average work hours-per-week ( ≈ 46hours-per-week)
>- **Prof-specialty** the highst education between all of the occupation:
>   - most of them got bachlors degree and some of them even got to have masters and Doctorate
>   - they work(≈44 hours-per-week) which is higher than average
>- ### The highest Top5 occupations based on rate:
>   - gaind more education than the others 
>   - has more critical changes in the orgnization they work with
>   - 4 out of 5 are jobs at office
>- ### based on gender:
>   - women gain rate based on the occupation is lower than men in the same occupation
>       - according to the data : women work "much" less houre p week 
>- #### Top10 occupations for each gender:
>   #sorted bassed on the highest rate of getting more than 50k a year
>     | num | ForFemales| ForMales |
>     | :---: | :---: | :---|
>     |1 | Prof-specialty | Exec-managerial|
>     |2 | Exec-managerial | Prof-specialty|
>     |3 | Adm-clerical | Protective-serv|
>     |4 | Protective-serv | Sales|
>     |5 | Tech-support | Tech-support|
>     |6 | Transport-moving | Adm-clerical|
>     |7 | Sales | Craft-repair|
>     |8 | Machine-op-inspct | Transport-moving|
>     |9 | Other-servic | Machine-op-inspct|
>     |10 | Other-servic | Farming-fishing|


### **Work-Class**
---


```python
workC.columns.name='mean'
workC=workC.sort_values(by='rate',ascending=False)
workC
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
      <th>mean</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>workclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Self-emp-inc</th>
      <td>37.0</td>
      <td>124.0</td>
      <td>46.391304</td>
      <td>11.329193</td>
      <td>50.416149</td>
      <td>0.770186</td>
    </tr>
    <tr>
      <th>Federal-gov</th>
      <td>44.0</td>
      <td>85.0</td>
      <td>43.046512</td>
      <td>11.403101</td>
      <td>41.868217</td>
      <td>0.658915</td>
    </tr>
    <tr>
      <th>Local-gov</th>
      <td>111.0</td>
      <td>149.0</td>
      <td>42.619231</td>
      <td>11.446154</td>
      <td>41.461538</td>
      <td>0.573077</td>
    </tr>
    <tr>
      <th>Self-emp-not-inc</th>
      <td>139.0</td>
      <td>163.0</td>
      <td>45.284768</td>
      <td>10.817881</td>
      <td>44.980132</td>
      <td>0.539735</td>
    </tr>
    <tr>
      <th>State-gov</th>
      <td>89.0</td>
      <td>89.0</td>
      <td>42.747191</td>
      <td>11.983146</td>
      <td>42.612360</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>Private</th>
      <td>1479.0</td>
      <td>1100.0</td>
      <td>38.217914</td>
      <td>10.280729</td>
      <td>41.523846</td>
      <td>0.426522</td>
    </tr>
    <tr>
      <th>Without-pay</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>62.000000</td>
      <td>10.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
workC['hours-per-week'].sort_values().plot.bar();plt.title('average hours-per-week in each workclass');
```


    
![png](output_81_0.png)
    


#### gender-WorkClass income:


```python
workCF=ds[ds['gender']=='Female'].groupby(['workclass','income']).size().unstack().fillna(0).join(ds[ds['gender']=='Female'][['age','educational-num','hours-per-week','workclass']].groupby('workclass').mean())
workCF['rate']=workCF['>50K']/(workCF['>50K'] + workCF['<=50K'])
workCF=workCF.sort_values(by='rate',ascending=False)
workCM=ds[ds['gender']=='Male'].groupby(['workclass','income']).size().unstack().fillna(0).join(ds[ds['gender']=='Male'][['age','educational-num','hours-per-week','workclass']].groupby('workclass').mean())
workCM['rate']=workCM['>50K']/(workCM['>50K'] + workCM['<=50K'])
workCM=workCM.sort_values(by='rate',ascending=False)
fig=plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
workCM['rate'].sort_values().plot.bar(color='skyblue')
plt.title('Rate for Male')
plt.xlabel('')
plt.xticks(rotation=70)
plt.subplot(1,2,2)
workCF['rate'].sort_values().plot.bar(color='hotpink')
plt.title('Rate for Female')
plt.xlabel('');
plt.xticks(rotation=70);
```


    
![png](output_83_0.png)
    



```python
workCF
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>workclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Self-emp-inc</th>
      <td>4.0</td>
      <td>11.0</td>
      <td>51.000000</td>
      <td>11.666667</td>
      <td>45.266667</td>
      <td>0.733333</td>
    </tr>
    <tr>
      <th>Self-emp-not-inc</th>
      <td>24.0</td>
      <td>23.0</td>
      <td>44.659574</td>
      <td>11.255319</td>
      <td>36.872340</td>
      <td>0.489362</td>
    </tr>
    <tr>
      <th>Local-gov</th>
      <td>52.0</td>
      <td>33.0</td>
      <td>42.388235</td>
      <td>12.058824</td>
      <td>40.552941</td>
      <td>0.388235</td>
    </tr>
    <tr>
      <th>Federal-gov</th>
      <td>21.0</td>
      <td>9.0</td>
      <td>41.300000</td>
      <td>10.966667</td>
      <td>41.266667</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>State-gov</th>
      <td>43.0</td>
      <td>13.0</td>
      <td>42.446429</td>
      <td>11.250000</td>
      <td>39.446429</td>
      <td>0.232143</td>
    </tr>
    <tr>
      <th>Private</th>
      <td>576.0</td>
      <td>159.0</td>
      <td>35.669388</td>
      <td>10.077551</td>
      <td>36.514286</td>
      <td>0.216327</td>
    </tr>
    <tr>
      <th>Without-pay</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>62.000000</td>
      <td>10.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
workCM
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>workclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Self-emp-inc</th>
      <td>33</td>
      <td>113</td>
      <td>45.917808</td>
      <td>11.294521</td>
      <td>50.945205</td>
      <td>0.773973</td>
    </tr>
    <tr>
      <th>Federal-gov</th>
      <td>23</td>
      <td>76</td>
      <td>43.575758</td>
      <td>11.535354</td>
      <td>42.050505</td>
      <td>0.767677</td>
    </tr>
    <tr>
      <th>Local-gov</th>
      <td>59</td>
      <td>116</td>
      <td>42.731429</td>
      <td>11.148571</td>
      <td>41.902857</td>
      <td>0.662857</td>
    </tr>
    <tr>
      <th>State-gov</th>
      <td>46</td>
      <td>76</td>
      <td>42.885246</td>
      <td>12.319672</td>
      <td>44.065574</td>
      <td>0.622951</td>
    </tr>
    <tr>
      <th>Self-emp-not-inc</th>
      <td>115</td>
      <td>140</td>
      <td>45.400000</td>
      <td>10.737255</td>
      <td>46.474510</td>
      <td>0.549020</td>
    </tr>
    <tr>
      <th>Private</th>
      <td>903</td>
      <td>941</td>
      <td>39.233731</td>
      <td>10.361714</td>
      <td>43.520607</td>
      <td>0.510304</td>
    </tr>
  </tbody>
</table>
</div>



###
> ### **outComes:**
>   **note:***this info is not generlized and it's just based on the current data*
>- **Self-emp-inc** jobs has the highest rate of getting more then 50k yearly which is **77%** :
>   - have the highest average work hours-per-week ( ≈ 50.5 hours-per-week)
>- **Federal-gov** the second highst rate:
>   - most of them got Some collage or more
>   - they work(≈42 hours-per-week) 7hours a day for 6 day 
>- **Private**
>   - has the lowest rate of getting more than 50K yearly (42%)
>   - most records in this data Set work in private  
>- ### based on gender:
>   - **women** :<br>
>        gain rate based on the work-class is lower than men in the same work-class, but in **self-emp-inc** the rate goes up to **73%** which is OK compared to other workclasses
>   - **men** :<br>
>       - rate is above 50% foe all workClasses
>       - **Top 3 work-classes** based on rate :
>
>           | - Self-emp-inc |   77% |
>           | :---:|:---:|    
>           |**- Federal-gov**  |   **76%** |
>           |**- Local-gov**    |   **66%** |


## **relationship/marital-status**
---


```python
ds['marital-status'].value_counts()
```




    Married-civ-spouse       2115
    Never-married             895
    Divorced                  411
    Separated                  83
    Widowed                    67
    Married-spouse-absent      35
    Married-AF-spouse           4
    Name: marital-status, dtype: int64




```python
ds['relationship'].value_counts()
```




    Husband           1872
    Not-in-family      774
    Own-child          373
    Unmarried          300
    Wife               218
    Other-relative      73
    Name: relationship, dtype: int64




```python
maritalStatus= ds.groupby(['marital-status','income']).size().unstack().fillna(0)
maritalStatus['rate'] = maritalStatus['>50K']/(maritalStatus['>50K']+maritalStatus['<=50K'])
maritalStatus=ds[["age","educational-num","hours-per-week","marital-status"]].groupby('marital-status').mean().join(maritalStatus)
maritalStatus.sort_values(by='rate',ascending=False)

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
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>marital-status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Married-AF-spouse</th>
      <td>31.000000</td>
      <td>11.000000</td>
      <td>40.500000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Married-civ-spouse</th>
      <td>43.167376</td>
      <td>10.833570</td>
      <td>44.339480</td>
      <td>667.0</td>
      <td>1448.0</td>
      <td>0.684634</td>
    </tr>
    <tr>
      <th>Divorced</th>
      <td>43.340633</td>
      <td>10.406326</td>
      <td>42.311436</td>
      <td>304.0</td>
      <td>107.0</td>
      <td>0.260341</td>
    </tr>
    <tr>
      <th>Widowed</th>
      <td>56.373134</td>
      <td>9.597015</td>
      <td>34.940299</td>
      <td>51.0</td>
      <td>16.0</td>
      <td>0.238806</td>
    </tr>
    <tr>
      <th>Married-spouse-absent</th>
      <td>40.371429</td>
      <td>9.171429</td>
      <td>40.771429</td>
      <td>28.0</td>
      <td>7.0</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>Separated</th>
      <td>40.915663</td>
      <td>9.771084</td>
      <td>41.662651</td>
      <td>70.0</td>
      <td>13.0</td>
      <td>0.156627</td>
    </tr>
    <tr>
      <th>Never-married</th>
      <td>29.264804</td>
      <td>10.263687</td>
      <td>38.007821</td>
      <td>780.0</td>
      <td>115.0</td>
      <td>0.128492</td>
    </tr>
  </tbody>
</table>
</div>




```python
relation= ds.groupby(['relationship','income']).size().unstack().fillna(0)
relation['rate'] = relation['>50K']/(relation['>50K']+relation['<=50K'])
relation=ds[["age","educational-num","hours-per-week","relationship"]].groupby('relationship').mean().join(relation)
relation.sort_values(by='rate',ascending=False)
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
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>relationship</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Husband</th>
      <td>43.732372</td>
      <td>10.847756</td>
      <td>45.221688</td>
      <td>578</td>
      <td>1294</td>
      <td>0.691239</td>
    </tr>
    <tr>
      <th>Wife</th>
      <td>39.683486</td>
      <td>11.009174</td>
      <td>37.435780</td>
      <td>68</td>
      <td>150</td>
      <td>0.688073</td>
    </tr>
    <tr>
      <th>Not-in-family</th>
      <td>38.445736</td>
      <td>10.755814</td>
      <td>42.224806</td>
      <td>572</td>
      <td>202</td>
      <td>0.260982</td>
    </tr>
    <tr>
      <th>Unmarried</th>
      <td>40.156667</td>
      <td>10.043333</td>
      <td>40.146667</td>
      <td>259</td>
      <td>41</td>
      <td>0.136667</td>
    </tr>
    <tr>
      <th>Other-relative</th>
      <td>32.493151</td>
      <td>9.027397</td>
      <td>37.561644</td>
      <td>67</td>
      <td>6</td>
      <td>0.082192</td>
    </tr>
    <tr>
      <th>Own-child</th>
      <td>24.986595</td>
      <td>9.359249</td>
      <td>32.946381</td>
      <td>356</td>
      <td>17</td>
      <td>0.045576</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig=plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
maritalStatus.rate.sort_values().plot.bar(color=list(colorsDict.values())[496:501])
plt.title('Rate based on maritalStatus')
plt.subplot(2,2,2)
relation.rate.sort_values().plot.bar(color=list(colorsDict.values())[496:501])
plt.title('Rate based on relationship')
plt.subplot(2,2,3)
maritalStatus.rate.sort_values().plot.pie(autopct='%1.1f%%',colors=list(colorsDict.values())[496:501])
plt.ylabel('')
plt.subplot(2,2,4)
relation.rate.sort_values().plot.pie(autopct='%1.1f%%',colors=list(colorsDict.values())[496:501])
plt.ylabel('');
```


    
![png](output_92_0.png)
    


### **relationship:**
---


```python
relation=relation.sort_values(by='rate',ascending=False)
relation
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
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>relationship</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Husband</th>
      <td>43.732372</td>
      <td>10.847756</td>
      <td>45.221688</td>
      <td>578</td>
      <td>1294</td>
      <td>0.691239</td>
    </tr>
    <tr>
      <th>Wife</th>
      <td>39.683486</td>
      <td>11.009174</td>
      <td>37.435780</td>
      <td>68</td>
      <td>150</td>
      <td>0.688073</td>
    </tr>
    <tr>
      <th>Not-in-family</th>
      <td>38.445736</td>
      <td>10.755814</td>
      <td>42.224806</td>
      <td>572</td>
      <td>202</td>
      <td>0.260982</td>
    </tr>
    <tr>
      <th>Unmarried</th>
      <td>40.156667</td>
      <td>10.043333</td>
      <td>40.146667</td>
      <td>259</td>
      <td>41</td>
      <td>0.136667</td>
    </tr>
    <tr>
      <th>Other-relative</th>
      <td>32.493151</td>
      <td>9.027397</td>
      <td>37.561644</td>
      <td>67</td>
      <td>6</td>
      <td>0.082192</td>
    </tr>
    <tr>
      <th>Own-child</th>
      <td>24.986595</td>
      <td>9.359249</td>
      <td>32.946381</td>
      <td>356</td>
      <td>17</td>
      <td>0.045576</td>
    </tr>
  </tbody>
</table>
</div>




```python
relation['hours-per-week'].sort_values().plot.bar(color=list(colorsDict.values())[496:501])
plt.title('average hours-per-week for each relationship type');
```


    
![png](output_95_0.png)
    


#### **gender and relationship income :**
--- 


```python
relationF=ds[ds['gender']=='Female'].groupby(['relationship','income']).size().unstack().fillna(0).join(ds[ds['gender']=='Female'][['age','educational-num','hours-per-week','relationship']].groupby('relationship').mean())
relationF['rate']=relationF['>50K']/(relationF['>50K'] + relationF['<=50K'])
relationF=relationF.sort_values(by='rate',ascending=False)
relationM=ds[ds['gender']=='Male'].groupby(['relationship','income']).size().unstack().fillna(0).join(ds[ds['gender']=='Male'][['age','educational-num','hours-per-week','relationship']].groupby('relationship').mean())
relationM['rate']=relationM['>50K']/(relationM['>50K'] + relationM['<=50K'])
relationM=relationM.sort_values(by='rate',ascending=False)
fig=plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
relationM['rate'].sort_values().plot.bar(color='skyblue')
plt.title('Rate for Male')
plt.xlabel('')
plt.xticks(rotation=70)
plt.subplot(1,2,2)
relationF['rate'].sort_values().plot.bar(color='hotpink')
plt.title('Rate for Female')
plt.xlabel('');
plt.xticks(rotation=70);
```


    
![png](output_97_0.png)
    



```python
relationF
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>relationship</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Wife</th>
      <td>68</td>
      <td>150</td>
      <td>39.683486</td>
      <td>11.009174</td>
      <td>37.435780</td>
      <td>0.688073</td>
    </tr>
    <tr>
      <th>Not-in-family</th>
      <td>279</td>
      <td>69</td>
      <td>39.589080</td>
      <td>10.885057</td>
      <td>39.853448</td>
      <td>0.198276</td>
    </tr>
    <tr>
      <th>Other-relative</th>
      <td>28</td>
      <td>5</td>
      <td>36.727273</td>
      <td>9.848485</td>
      <td>35.272727</td>
      <td>0.151515</td>
    </tr>
    <tr>
      <th>Unmarried</th>
      <td>208</td>
      <td>21</td>
      <td>40.842795</td>
      <td>9.960699</td>
      <td>39.213974</td>
      <td>0.091703</td>
    </tr>
    <tr>
      <th>Own-child</th>
      <td>138</td>
      <td>3</td>
      <td>23.893617</td>
      <td>9.297872</td>
      <td>28.269504</td>
      <td>0.021277</td>
    </tr>
  </tbody>
</table>
</div>




```python
relationM
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>relationship</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Husband</th>
      <td>578</td>
      <td>1294</td>
      <td>43.732372</td>
      <td>10.847756</td>
      <td>45.221688</td>
      <td>0.691239</td>
    </tr>
    <tr>
      <th>Not-in-family</th>
      <td>293</td>
      <td>133</td>
      <td>37.511737</td>
      <td>10.650235</td>
      <td>44.161972</td>
      <td>0.312207</td>
    </tr>
    <tr>
      <th>Unmarried</th>
      <td>51</td>
      <td>20</td>
      <td>37.943662</td>
      <td>10.309859</td>
      <td>43.154930</td>
      <td>0.281690</td>
    </tr>
    <tr>
      <th>Own-child</th>
      <td>218</td>
      <td>14</td>
      <td>25.650862</td>
      <td>9.396552</td>
      <td>35.788793</td>
      <td>0.060345</td>
    </tr>
    <tr>
      <th>Other-relative</th>
      <td>39</td>
      <td>1</td>
      <td>29.000000</td>
      <td>8.350000</td>
      <td>39.450000</td>
      <td>0.025000</td>
    </tr>
  </tbody>
</table>
</div>



###
> ### **outComes:**
>   **note:***this info is not generlized and it's just based on the current data*
>- **"wife and husband"** 
>   - most of them got Some collage or more
>   - has the highest rate of getting >50K higher than **68%**
>   - ***"husband"*** : has the highest average work hours-per-week ( ≈ 45 hours-per-week)
>- **"Not-in-family","Unmarried","Other-relative",and "Own-child"** :
>   - most of them got Some collage or more
>   -  "own a child" has the lowest rate
>   - we can see a clear difference between married and non marrid indiviuals

>- ### based on gender:
>   - **women** :<br>
>        for marrid women ("wife") we can see a higher rate of gitting >50K yearly
>   - **men** :<br>
>        for marrid men ("Husband") we can see a higher rate of gitting >50K yearly
>


#### **gender and maritual-statuse income :**
--- 


```python

maritalStatusF=ds[ds['gender']=='Female'].groupby(['marital-status','income']).size().unstack().fillna(0).join(ds[ds['gender']=='Female'][['age','educational-num','hours-per-week','marital-status']].groupby('marital-status').mean())
maritalStatusF['rate']=maritalStatusF['>50K']/(maritalStatusF['>50K'] + maritalStatusF['<=50K'])
maritalStatusF=maritalStatusF.sort_values(by='rate',ascending=False)
maritalStatusM=ds[ds['gender']=='Male'].groupby(['marital-status','income']).size().unstack().fillna(0).join(ds[ds['gender']=='Male'][['age','educational-num','hours-per-week','marital-status']].groupby('marital-status').mean())
maritalStatusM['rate']=maritalStatusM['>50K']/(maritalStatusM['>50K'] + maritalStatusM['<=50K'])
maritalStatusM=maritalStatusM.sort_values(by='rate',ascending=False)
fig=plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
maritalStatusM['rate'].sort_values().plot.bar(color='skyblue')
plt.title('Rate for Male')
plt.xlabel('')
plt.xticks(rotation=70)
plt.subplot(1,2,2)
maritalStatusF['rate'].sort_values().plot.bar(color='hotpink')
plt.title('Rate for Female')
plt.xlabel('');
plt.xticks(rotation=70);
```


    
![png](output_102_0.png)
    



```python
maritalStatusF
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>marital-status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Married-AF-spouse</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>32.333333</td>
      <td>11.666667</td>
      <td>40.666667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Married-civ-spouse</th>
      <td>77.0</td>
      <td>151.0</td>
      <td>39.464912</td>
      <td>10.938596</td>
      <td>37.206140</td>
      <td>0.662281</td>
    </tr>
    <tr>
      <th>Widowed</th>
      <td>46.0</td>
      <td>11.0</td>
      <td>56.175439</td>
      <td>9.403509</td>
      <td>34.491228</td>
      <td>0.192982</td>
    </tr>
    <tr>
      <th>Divorced</th>
      <td>197.0</td>
      <td>39.0</td>
      <td>43.084746</td>
      <td>10.296610</td>
      <td>40.576271</td>
      <td>0.165254</td>
    </tr>
    <tr>
      <th>Married-spouse-absent</th>
      <td>14.0</td>
      <td>2.0</td>
      <td>38.687500</td>
      <td>8.812500</td>
      <td>35.500000</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>Never-married</th>
      <td>341.0</td>
      <td>40.0</td>
      <td>29.559055</td>
      <td>10.503937</td>
      <td>35.587927</td>
      <td>0.104987</td>
    </tr>
    <tr>
      <th>Separated</th>
      <td>46.0</td>
      <td>2.0</td>
      <td>42.000000</td>
      <td>9.729167</td>
      <td>39.291667</td>
      <td>0.041667</td>
    </tr>
  </tbody>
</table>
</div>




```python
maritalStatusM
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
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>age</th>
      <th>educational-num</th>
      <th>hours-per-week</th>
      <th>rate</th>
    </tr>
    <tr>
      <th>marital-status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Married-AF-spouse</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>27.000000</td>
      <td>9.000000</td>
      <td>40.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Married-civ-spouse</th>
      <td>590.0</td>
      <td>1297.0</td>
      <td>43.614732</td>
      <td>10.820880</td>
      <td>45.201378</td>
      <td>0.687334</td>
    </tr>
    <tr>
      <th>Widowed</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>57.500000</td>
      <td>10.700000</td>
      <td>37.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>Divorced</th>
      <td>107.0</td>
      <td>68.0</td>
      <td>43.685714</td>
      <td>10.554286</td>
      <td>44.651429</td>
      <td>0.388571</td>
    </tr>
    <tr>
      <th>Separated</th>
      <td>24.0</td>
      <td>11.0</td>
      <td>39.428571</td>
      <td>9.828571</td>
      <td>44.914286</td>
      <td>0.314286</td>
    </tr>
    <tr>
      <th>Married-spouse-absent</th>
      <td>14.0</td>
      <td>5.0</td>
      <td>41.789474</td>
      <td>9.473684</td>
      <td>45.210526</td>
      <td>0.263158</td>
    </tr>
    <tr>
      <th>Never-married</th>
      <td>439.0</td>
      <td>75.0</td>
      <td>29.046693</td>
      <td>10.085603</td>
      <td>39.801556</td>
      <td>0.145914</td>
    </tr>
  </tbody>
</table>
</div>



###
> ### **outComes:**
>   **note:***this info is not generlized and it's just based on the current data*
>- **"Married-AF-spouse/Married-civ-spouse"** 
>   - most of them got Some collage or more
>   - has the highest rate of getting >50K higher than **68%**
>   - ***"Married-AF-spouse" :***  
>        - has the highest rate of all of the data set making **100%**
>        - but there are only "4" people under this class which making it untrusted enough  
>- **"Widowed","Divorced","Separated","Married-spouse-absent",and "Never-married"** :
>   - most of them got high-school or more
>   -  *"Never-married"* has the lowest rate(12%)
>- we can see a clear difference between married and non married indiviuals
>- ### based on gender:
>   - **women** :<br>
>        - there are a really small differences but in general *Female income based on the maritual-status is as high as the Males*
>   - **men** :<br>
>        - for marrid men ("Widowed") has a 50% chance of gitting >50K yearly :
>           - but this is conflecting with the original data(male and females together)
>           - only 10 male indiviuals in the data_set are "Widowed" so we can't generlize that for all "Widowed" men


### actions to take :
> - since the 'relationship' and 'marital-status' are the same information but in different words 
> - when making the model we can reduce the information by :
>   - drop the column "marital-status"
>   - reform the 'relationship' column to present (if married) by :
>       - taking married indivuals ("wife"/"husband") as a class represented by the value "1"
>       - while others are represented by "0"



```python
ds.drop('marital-status',axis=1,inplace=True)
```


```python
def relaCod(x):
    if (x=="Wife")or(x=="Husband"):
        return(1)
    return(0)

ds['relationship']=ds['relationship'].apply(lambda y: relaCod(y))

```


```python
ds['relationship'].value_counts()
```




    1    2090
    0    1520
    Name: relationship, dtype: int64



# **Making the AI Model**
<div class="alert-block alert-info">
</div>


```python
ds
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
      <th>age</th>
      <th>workclass</th>
      <th>educational-num</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>Private</td>
      <td>13</td>
      <td>Sales</td>
      <td>1</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>Self-emp-not-inc</td>
      <td>13</td>
      <td>Exec-managerial</td>
      <td>1</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43</td>
      <td>Private</td>
      <td>13</td>
      <td>Prof-specialty</td>
      <td>1</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>Private</td>
      <td>13</td>
      <td>Craft-repair</td>
      <td>1</td>
      <td>White</td>
      <td>Male</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36</td>
      <td>Private</td>
      <td>9</td>
      <td>Sales</td>
      <td>1</td>
      <td>White</td>
      <td>Male</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3798</th>
      <td>32</td>
      <td>Private</td>
      <td>14</td>
      <td>Tech-support</td>
      <td>0</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>11</td>
      <td>Taiwan</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3799</th>
      <td>22</td>
      <td>Private</td>
      <td>10</td>
      <td>Protective-serv</td>
      <td>0</td>
      <td>White</td>
      <td>Male</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3800</th>
      <td>27</td>
      <td>Private</td>
      <td>12</td>
      <td>Tech-support</td>
      <td>1</td>
      <td>White</td>
      <td>Female</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3801</th>
      <td>58</td>
      <td>Private</td>
      <td>9</td>
      <td>Adm-clerical</td>
      <td>0</td>
      <td>White</td>
      <td>Female</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3802</th>
      <td>22</td>
      <td>Private</td>
      <td>9</td>
      <td>Adm-clerical</td>
      <td>0</td>
      <td>White</td>
      <td>Male</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
<p>3610 rows × 10 columns</p>
</div>



## **Transform Data type:**


```python
ds['workclass']=ds['workclass'].apply(lambda x: list(ds['workclass'].unique()).index(x)+1)
ds['occupation']=ds['occupation'].apply(lambda x: list(ds['occupation'].unique()).index(x)+1)
ds['race']=ds['race'].apply(lambda x: list(ds['race'].unique()).index(x)+1)
ds['native-country']=ds['native-country'].apply(lambda x: list(ds['native-country'].unique()).index(x)+1)
```


```python
def TransformData(x):
    if x=="Male" or x==">50K":
        return(1)
    if x=="Female" or x=="<=50K":
        return(0)
ds['income']=ds['income'].apply(lambda x :TransformData(x))
ds['gender']=ds['gender'].apply(lambda x :TransformData(x))
```


```python
ds.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>educational-num</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>50</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>2</td>
      <td>13</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>50</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43</td>
      <td>1</td>
      <td>13</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>50</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>1</td>
      <td>13</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>50</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## **correlation after transform data types**


```python
ds.corr()
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
      <th>age</th>
      <th>workclass</th>
      <th>educational-num</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>0.176192</td>
      <td>0.101284</td>
      <td>-0.178911</td>
      <td>0.321839</td>
      <td>-0.079054</td>
      <td>0.115204</td>
      <td>0.103384</td>
      <td>0.002506</td>
      <td>0.292770</td>
    </tr>
    <tr>
      <th>workclass</th>
      <td>0.176192</td>
      <td>1.000000</td>
      <td>0.169860</td>
      <td>-0.097003</td>
      <td>0.108145</td>
      <td>0.014746</td>
      <td>0.046105</td>
      <td>0.086165</td>
      <td>-0.059699</td>
      <td>0.163690</td>
    </tr>
    <tr>
      <th>educational-num</th>
      <td>0.101284</td>
      <td>0.169860</td>
      <td>1.000000</td>
      <td>-0.387210</td>
      <td>0.127258</td>
      <td>-0.098317</td>
      <td>0.035164</td>
      <td>0.195010</td>
      <td>-0.109463</td>
      <td>0.375456</td>
    </tr>
    <tr>
      <th>occupation</th>
      <td>-0.178911</td>
      <td>-0.097003</td>
      <td>-0.387210</td>
      <td>1.000000</td>
      <td>-0.211644</td>
      <td>0.137401</td>
      <td>-0.088254</td>
      <td>-0.196489</td>
      <td>0.088856</td>
      <td>-0.318307</td>
    </tr>
    <tr>
      <th>relationship</th>
      <td>0.321839</td>
      <td>0.108145</td>
      <td>0.127258</td>
      <td>-0.211644</td>
      <td>1.000000</td>
      <td>-0.122930</td>
      <td>0.434270</td>
      <td>0.213210</td>
      <td>-0.025184</td>
      <td>0.510145</td>
    </tr>
    <tr>
      <th>race</th>
      <td>-0.079054</td>
      <td>0.014746</td>
      <td>-0.098317</td>
      <td>0.137401</td>
      <td>-0.122930</td>
      <td>1.000000</td>
      <td>-0.113833</td>
      <td>-0.057932</td>
      <td>0.085042</td>
      <td>-0.121083</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>0.115204</td>
      <td>0.046105</td>
      <td>0.035164</td>
      <td>-0.088254</td>
      <td>0.434270</td>
      <td>-0.113833</td>
      <td>1.000000</td>
      <td>0.253948</td>
      <td>-0.031182</td>
      <td>0.264161</td>
    </tr>
    <tr>
      <th>hours-per-week</th>
      <td>0.103384</td>
      <td>0.086165</td>
      <td>0.195010</td>
      <td>-0.196489</td>
      <td>0.213210</td>
      <td>-0.057932</td>
      <td>0.253948</td>
      <td>1.000000</td>
      <td>-0.014362</td>
      <td>0.264204</td>
    </tr>
    <tr>
      <th>native-country</th>
      <td>0.002506</td>
      <td>-0.059699</td>
      <td>-0.109463</td>
      <td>0.088856</td>
      <td>-0.025184</td>
      <td>0.085042</td>
      <td>-0.031182</td>
      <td>-0.014362</td>
      <td>1.000000</td>
      <td>-0.063718</td>
    </tr>
    <tr>
      <th>income</th>
      <td>0.292770</td>
      <td>0.163690</td>
      <td>0.375456</td>
      <td>-0.318307</td>
      <td>0.510145</td>
      <td>-0.121083</td>
      <td>0.264161</td>
      <td>0.264204</td>
      <td>-0.063718</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
ds.describe()
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
      <th>age</th>
      <th>workclass</th>
      <th>educational-num</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3610.000000</td>
      <td>3610.000000</td>
      <td>3610.000000</td>
      <td>3610.000000</td>
      <td>3610.000000</td>
      <td>3610.000000</td>
      <td>3610.000000</td>
      <td>3610.000000</td>
      <td>3610.000000</td>
      <td>3610.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.893075</td>
      <td>1.757064</td>
      <td>10.580332</td>
      <td>4.908310</td>
      <td>0.578947</td>
      <td>1.235734</td>
      <td>0.731579</td>
      <td>42.263989</td>
      <td>2.168144</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.451276</td>
      <td>1.404772</td>
      <td>2.619661</td>
      <td>3.307156</td>
      <td>0.493796</td>
      <td>0.678693</td>
      <td>0.443199</td>
      <td>11.801834</td>
      <td>4.733079</td>
      <td>0.499376</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>40.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>48.750000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>7.000000</td>
      <td>16.000000</td>
      <td>14.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>37.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## **Random Sampling**
- this data set was devided : the first half was >50K the other was <=50K
,so we have to mix these tow togather to be ready for the next step


```python
sampler = np.random.permutation(3610)
sampler
```




    array([3199, 2538,   10, ..., 1485,  743, 3121])




```python
ds=ds.take(sampler)
ds
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
      <th>age</th>
      <th>workclass</th>
      <th>educational-num</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3362</th>
      <td>26</td>
      <td>1</td>
      <td>9</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>34</td>
      <td>1</td>
      <td>3</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>30</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3605</th>
      <td>41</td>
      <td>1</td>
      <td>10</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2803</th>
      <td>35</td>
      <td>1</td>
      <td>9</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>20</td>
      <td>1</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2618</th>
      <td>44</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>43</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1524</th>
      <td>58</td>
      <td>1</td>
      <td>12</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>50</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>765</th>
      <td>40</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>46</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3280</th>
      <td>60</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3610 rows × 10 columns</p>
</div>



## **Split the dataSet**


```python
from sklearn.model_selection import train_test_split
```


```python
x_train,x_test,y_train,y_test = train_test_split(ds,ds['income'],test_size=0.25,random_state=0)
```


```python
y_train
```




    2793    0
    1559    1
    2748    0
    1869    0
    2907    0
           ..
    3045    0
    2375    0
    39      1
    1575    1
    1612    1
    Name: income, Length: 2707, dtype: int64




```python
x_test
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
      <th>age</th>
      <th>workclass</th>
      <th>educational-num</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1588</th>
      <td>47</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2187</th>
      <td>19</td>
      <td>3</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2263</th>
      <td>34</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2534</th>
      <td>46</td>
      <td>1</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>48</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1705</th>
      <td>31</td>
      <td>5</td>
      <td>13</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>55</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3008</th>
      <td>32</td>
      <td>6</td>
      <td>11</td>
      <td>9</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2369</th>
      <td>29</td>
      <td>1</td>
      <td>13</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>63</td>
      <td>2</td>
      <td>4</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2790</th>
      <td>31</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>34</td>
      <td>1</td>
      <td>10</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>84</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>903 rows × 10 columns</p>
</div>



## **KNN**


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
```


```python
n=0
acc=0
for i in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train.drop(['income'],axis=1),y_train)
    y_pred=knn.predict(x_test.drop(['income'],axis=1))
    c=metrics.accuracy_score(y_test,y_pred)
    if c>acc:
        n=i
        acc=c
print(n,acc)
    

```

    31 0.7475083056478405
    


```python
knn=KNeighborsClassifier(n_neighbors=n)
knn.fit(x_train.drop(['income'],axis=1),y_train)
y_pred=knn.predict(x_test.drop(['income'],axis=1))
# ----------------------------------------
```

### **classification report:**


```python
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print('confusion Matrix: \n',metrics.confusion_matrix(y_test,y_pred))
print('classification Report: \n',metrics.classification_report(y_test,y_pred))
```

    Accuracy:  0.7475083056478405
    confusion Matrix: 
     [[328 133]
     [ 95 347]]
    classification Report: 
                   precision    recall  f1-score   support
    
               0       0.78      0.71      0.74       461
               1       0.72      0.79      0.75       442
    
        accuracy                           0.75       903
       macro avg       0.75      0.75      0.75       903
    weighted avg       0.75      0.75      0.75       903
    
    

## **NaiveBayas**


```python
from sklearn.naive_bayes import GaussianNB as gnb
```


```python
GNBmodel = gnb()
GNBmodel.fit(x_train.drop(['income'],axis=1),y_train)
predict=GNBmodel.predict(x_test.drop(['income'],axis=1))

```

### **classification report:**


```python
print("Accuracy: ",metrics.accuracy_score(y_test,predict))
print('confusion Matrix: \n',metrics.confusion_matrix(y_test,predict))
print('classification Report: \n',metrics.classification_report(y_test,predict))
```

    Accuracy:  0.7685492801771872
    confusion Matrix: 
     [[332 129]
     [ 80 362]]
    classification Report: 
                   precision    recall  f1-score   support
    
               0       0.81      0.72      0.76       461
               1       0.74      0.82      0.78       442
    
        accuracy                           0.77       903
       macro avg       0.77      0.77      0.77       903
    weighted avg       0.77      0.77      0.77       903
    
    

## **DecisionTreeClassifier**


```python
from sklearn.tree import DecisionTreeClassifier as DTC
```


```python
dtmodel=DTC(random_state=150)
dtmodel.fit(x_train.drop(['income'],axis=1),y_train)
predict=dtmodel.predict(x_test.drop(['income'],axis=1))
```


```python
print("Accuracy: ",metrics.accuracy_score(y_test,predict))
print('confusion Matrix: \n',metrics.confusion_matrix(y_test,predict))
print('classification Report: \n',metrics.classification_report(y_test,predict))
```

    Accuracy:  0.7264673311184939
    confusion Matrix: 
     [[327 134]
     [113 329]]
    classification Report: 
                   precision    recall  f1-score   support
    
               0       0.74      0.71      0.73       461
               1       0.71      0.74      0.73       442
    
        accuracy                           0.73       903
       macro avg       0.73      0.73      0.73       903
    weighted avg       0.73      0.73      0.73       903
    
    

## **trying them out:**


```python

dic={'age':[30],
 'workclass':[1],
 'educational-num':[13],
 'occupation':[5],
 'relationship':[0],
 'race':[1],
 'gender':[0],
 'hours-per-week':[35],
 'native-country':[1]}

# sample=np.array([40,1,13,5,0,1,1,40,1]).reshape(1, -1)
sample=pd.DataFrame(dic)
knn_pred=knn.predict(sample)
Naivepred=GNBmodel.predict(sample)
Dtreepred=dtmodel.predict(sample)
print("",knn_pred,"\n",Naivepred,"\n",Dtreepred,"\n")
```

     [0] 
     [0] 
     [0] 
    
    
