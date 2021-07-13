# A/B Test

**This notebook is used to simulate an A/B Test** based on an e-commerce website dataset provided by Udacity. The dataset is available [here](https://www.kaggle.com/zhangluyuan/ab-testing?select=ab_data.csv).

A/B test is "the simplest type of controlled experiment that compares two variants: A and B" [(KOHAVI; TANG; XU, 2020)](https://www.amazon.com.br/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264). **The goal of running an A/B test is to gather data to drive decision making**, rather than relying on intuition. 

A variant is an user experience being tested. In A/B tests, variants are usually called Control (the old version) and Treatment (the new version). More than one Treatment can be tested at once depending on the business problem you are trying to solve. These are called multivariant tests. Besides, it is also possible to test more than one variable at once, using Multivarible Tests (MVT) [(KOHAVI et al., 2009)](http://www.robotics.stanford.edu/~ronnyk/2009controlledExperimentsOnTheWebSurvey.pdf). In online A/B tests, "users are randomly split between variants in a persistent manner (a user receives the same variant in multiple visits)" (KOHAVI; TANG; XU, 2020).

The first step of an A/B Test is to decide what to test. This is done by figuring where there is most opportunity based on your company top-level goal (e.g., increase revenue from users who visit the company's homepage). This can be done by talking to experts, analysing your product and your customers or checking out the competition. Then, we need to translate the opportunity identified into a testable hypothesis (e.g., making the browse category menu more prominent will make the site easier to navigate and increase sales). Next, we select a target metric (e.g, convertion rate) and define the practical significance level (also called minimum detectable effect), i.e., the minimum change to the baseline rate that is useful to the business [(GEOGHEGAN, 2020)](https://robbiegeoghegan.medium.com/implementing-a-b-tests-in-python-514e9eb5b3a1).

We also need to examine whether any difference between the two variants is statistically significant. First, we can determine the confidence level that we will use in the A/B Test. It is usual to adopt a confidence level of 95%. "A 95% confidence interval is the range that covers the true difference 95% of the time" (KOHAVI; TANG; XU, 2020). Another common practice is to compute the p-value. "Given estimates from the Control and Treatment samples, we compute the p-value for the difference, which is the probability of observing such difference" (KOHAVI; TANG; XU, 2020). A p-value of 0.05 is commonly used in A/B Tests. Finally, it is necessary to define the statistical power, i.e., "the probability of detecting a meaningful difference between the variants when there really is one" (KOHAVI; TANG; XU, 2020). A common practice is to have a statistical power between 80 and 90%.

The next step, is to define the randomization unit. Kohavi, Tang and Xu (2020) recommend to use the users as the randomization unit for A/B Tests with online audiences. Then, we need to calculate the sample size of the experiment. This is done considering the baseline rate (an estimate of the metric being analyzed before making any changes), practical significance level, confidence level, and sensitivity (ability to detect statistically significant differences) (GEOGHEGAN, 2020). "The larger the sample size, the more precise our estimates, i.e. the smaller our confidence intervals, the higher the chance to detect a difference in the two groups, if present" [(FILLINICH, 2020)](https://medium.com/@RenatoFillinich/ab-testing-with-python-e5964dd66143). 

Finally, we can define how long it will take to run the A/B Test. For example, if our sample size is 10000 and the average traffic in our website is 1100 users per day, we would need to run the experiment for 10 days.

----------
We do not have an specific target with our dataset. So, we will create scenario for our notebook based on (FILLINICH, 2020):

    Let’s imagine we work on the product team at a medium-sized online e-commerce business. The UX designer worked really hard on a new version of the product page, and believe that it will lead to a higher conversion rate. The product manager told us that the current conversion rate is about 13% on average throughout the year, and that the team would be happy with an increase of 1%, meaning that the new design will be considered a success if it raises the conversion rate to 14%.
    
So, we decide to do an A/B Test. Our hypothesis is that the new version of the product page will improve the conversion rate. **We will adopt a confidence level of 95%**  and **80% power**. We use the website users as the radomization unit. According to what the PM told us, the **baseline conversion rate is 13%** and the **practical significance level is 1%**.

The notebook is divided as follows:
    
1. Sample size calculation
2. Data exploration
3. Interpreting the results
4. Conclusion

-------

## 1. Sample size calculation


```python
import statsmodels.stats.api as sms
import math

baseline_rate = .13
practical_sig_lvl = 0.01
power = 0.8
confidence_lvl=0.95

effect_size = sms.proportion_effectsize(baseline_rate, baseline_rate+practical_sig_lvl) 
required_n = sms.NormalIndPower().solve_power(effect_size,power=power,alpha=(1-confidence_lvl),ratio=1)    
required_n = math.ceil(required_n)
required_n
```




    18326



Therefore, **our sample size is 18,326 users per variant**. Unfortunately, we don't know the average traffic to our site. So, we cannot calculate how long our experiment should run. But let's import the dataset and have a look at it.

## 2. Data exploration

Thus, we see that A/B Test lasted 22 days, starting at 02/01 and ending at 24/01. It only last for half the day on 02/01 and 24/01. **The average traffic on the website was 13,388 users per day. Therefore, we would only need to run our experiment for 3 days**. Now, we have two options. Choose three consectuive days in our dataset and consider that the experiment only lasted that long or change our parameters, for example increase the confidence level to 99%. We decided to choose three days and continue to work with the scenario we created. So, let's select the first three days in the data set and have a look at them. We will ignore the first day (02/01), because the experiment only lasted for half a day on this day.

### Getting the data


```python
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
```


```python
data_ab = pd.read_csv(r'/Users/leuzinger/Dropbox/Data Science/Awari/Teste A-B/Data set/udacity/ab_data.csv')
```


```python
data_ab.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 294478 entries, 0 to 294477
    Data columns (total 5 columns):
     #   Column        Non-Null Count   Dtype 
    ---  ------        --------------   ----- 
     0   user_id       294478 non-null  int64 
     1   timestamp     294478 non-null  object
     2   group         294478 non-null  object
     3   landing_page  294478 non-null  object
     4   converted     294478 non-null  int64 
    dtypes: int64(2), object(3)
    memory usage: 11.2+ MB



```python
data_ab.head()
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
      <th>user_id</th>
      <th>timestamp</th>
      <th>group</th>
      <th>landing_page</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>851104</td>
      <td>2017-01-21 22:11:48.556739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>804228</td>
      <td>2017-01-12 08:01:45.159739</td>
      <td>control</td>
      <td>old_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>661590</td>
      <td>2017-01-11 16:55:06.154213</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>853541</td>
      <td>2017-01-08 18:28:03.143765</td>
      <td>treatment</td>
      <td>new_page</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>864975</td>
      <td>2017-01-21 01:52:26.210827</td>
      <td>control</td>
      <td>old_page</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.to_datetime(data_ab['timestamp']).value_counts()
pd.to_datetime(data_ab['timestamp']).dt.day.value_counts().sort_index()
```




    2      5783
    3     13394
    4     13284
    5     13124
    6     13528
    7     13381
    8     13564
    9     13439
    10    13523
    11    13553
    12    13322
    13    13238
    14    13329
    15    13449
    16    13327
    17    13322
    18    13285
    19    13293
    20    13393
    21    13475
    22    13423
    23    13511
    24     7538
    Name: timestamp, dtype: int64




```python
data_ab['day'] = pd.to_datetime(data_ab['timestamp']).dt.day
data_ab['day'].loc[(data_ab["day"] > 2) & (data_ab["day"] <24)].value_counts().mean()
```




    13388.42857142857



----

### Data Preparation

First, we verify that there is some users in both groups that saw the webpage they were not supposed to see. **All users in the control group were supposed to see the old webpage and all users in the treatment group should have seen the new webpage**. Therefore, we will remove the users who saw the wrong webpage from our dataset. 

Then, we check for repeated users but there is none. Thus, our data set is ready to be used.


```python
data_ab3 = data_ab.loc[(data_ab["day"] > 2) & (data_ab["day"] <6)]
data_ab3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 39802 entries, 8 to 294473
    Data columns (total 6 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   user_id       39802 non-null  int64 
     1   timestamp     39802 non-null  object
     2   group         39802 non-null  object
     3   landing_page  39802 non-null  object
     4   converted     39802 non-null  int64 
     5   day           39802 non-null  int64 
    dtypes: int64(3), object(3)
    memory usage: 2.1+ MB



```python
data_ab3['landing_page'].value_counts()
```




    new_page    19946
    old_page    19856
    Name: landing_page, dtype: int64




```python
data_ab3['group'].value_counts()
```




    treatment    19925
    control      19877
    Name: group, dtype: int64




```python
pd.crosstab(data_ab3['group'], data_ab3['landing_page'])
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
      <th>landing_page</th>
      <th>new_page</th>
      <th>old_page</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>control</th>
      <td>282</td>
      <td>19595</td>
    </tr>
    <tr>
      <th>treatment</th>
      <td>19664</td>
      <td>261</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_ab3 = data_ab3.drop(data_ab3.loc[(data_ab3["group"] == 'control') & (data_ab["landing_page"] == 'new_page')].index)
data_ab3 = data_ab3.drop(data_ab3.loc[(data_ab3["group"] == 'treatment') & (data_ab["landing_page"] == 'old_page')].index)
pd.crosstab(data_ab3['group'], data_ab3['landing_page'])
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
      <th>landing_page</th>
      <th>new_page</th>
      <th>old_page</th>
    </tr>
    <tr>
      <th>group</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>control</th>
      <td>0</td>
      <td>19595</td>
    </tr>
    <tr>
      <th>treatment</th>
      <td>19664</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_ab3['user_id'].duplicated().sum()
```




    0



-----

## 3. Interpreting the results 

First, we calculate the conversion rate, standard deviation and standard error for each variant. **We find that the conversion rate of the old page is slightly better than the new one, but the standard error for the old page is higher than the new one**.

Then, we use the chi-square to test our hypothesis:

* H0: “the conversion rate is the same for the two versions”
* H1: “the new version of the product page improves the conversion rate”

We chose the chi-square test because we are comparing two categorical variables from the same population [(UCLA, 2021)](https://stats.idre.ucla.edu/stata/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-stata/).

**Given that the p-value is greater than 0.05, we cannot reject the Null hypothesis (H0) and have to assume that both versions have the same convertion rate**.

Additionally, if we look at the confidence interval for the Treatment group we notice that it does not includes our target value and not even our baseline conversion rate. This is further proof that our new design is not likely to be an improvement on our old design.


```python
conversion_rates = data_ab3.groupby('landing_page')['converted']

std_p = lambda x: np.std(x, ddof=0)              
se_p = lambda x: stats.sem(x, ddof=0)            

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rates.style.format('{:.5f}')
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
      <th>conversion_rate</th>
      <th>std_deviation</th>
      <th>std_error</th>
    </tr>
    <tr>
      <th>landing_page</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>new_page</th>
      <td>0.115134</td>
      <td>0.319184</td>
      <td>0.002276</td>
    </tr>
    <tr>
      <th>old_page</th>
      <td>0.119622</td>
      <td>0.324519</td>
      <td>0.002318</td>
    </tr>
  </tbody>
</table>
</div>




```python
#calculate p-value
con_results = data_ab3[data_ab3['group'] == 'control']['converted']
treat_results = data_ab3[data_ab3['group'] == 'treatment']['converted']
T = np.array([[con_results.sum(), con_results.size-con_results.sum()], 
              [treat_results.sum(), treat_results.size-treat_results.sum()]])
print(f"p-value: {scipy.stats.chi2_contingency(T,correction=False)[1]:.5f}")

#create 95% confidence interval for population mean weight
(lower_con, upper_con) = stats.norm.interval(alpha=confidence_lvl, loc=np.mean(con_results), scale=stats.sem(con_results))
(lower_treat, upper_treat) = stats.norm.interval(alpha=confidence_lvl, loc=np.mean(treat_results), scale=stats.sem(treat_results))
print(f"Confidence interval 95% for control group: ({lower_con:.5f}, {upper_con:.5f})")
print(f"Confidence interval 95% for treatment group: ({lower_treat:.5f}, {upper_treat:.5f})")
```

    p-value: 0.16715
    Confidence interval 95% for control group: (0.11508, 0.12417)
    Confidence interval 95% for treatment group: (0.11067, 0.11960)


## 4. Conclusion

In this notebook, we simulate an A/B test. In our scenario, the UX designer at a midsize online e-commerce company designed a new version of the product page believing that this would lead to a higher conversion rate. **However, after analyzing the data from the A/B Test, we found that the new version did not perform better than the old one**. On the contrary, it appears to perform a little worse, although we cannot conclude that. 
