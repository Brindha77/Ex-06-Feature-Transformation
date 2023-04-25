# Ex-06-Feature-Transformation

# AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
# STEP 1:
Read the given Data

#STEP 2:
Clean the Data Set using Data Cleaning Process

#STEP 3:
Apply Feature Transformation techniques to all the features of the data set

#STEP 4:
Print the transformed features

# PROGRAM:
Developed by: BRINDHA R
Register No. : 212222230023

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
# OUTPUT:
# DATASET
![f1](https://user-images.githubusercontent.com/118889143/233913832-cb5791ef-7d54-4e89-9eec-1969fa33f233.png)

# ISNULL
![f2](https://user-images.githubusercontent.com/118889143/233913919-a505a861-f314-455a-8e10-7b1e9f0481b6.png)

# INFO
![f3](https://user-images.githubusercontent.com/118889143/233914021-fba93103-a020-4ef7-8bf3-bc942e54bdb7.png)

# DESCRIBE:
![f4](https://user-images.githubusercontent.com/118889143/233914116-8010daef-6071-46ab-83c3-d48ad87f01ef.png)

# HIGHLY POSITIVE SKEW:
![f5](https://user-images.githubusercontent.com/118889143/233914189-cda56ab7-5daa-40ef-8731-e351f2c4d055.png)

![f6](https://user-images.githubusercontent.com/118889143/233914293-81512048-9f9f-42df-8728-23340d226225.png)

# HIGHLY NEGATIVE SKEW:
![f7](https://user-images.githubusercontent.com/118889143/234240716-441f3262-4637-4562-8138-a7889f22ed32.png)

# MODERATE POSITIVE SKEW:
![f8](https://user-images.githubusercontent.com/118889143/234240831-9c9c1129-c260-437c-9e91-69c8eaf33dd9.png)

# MODERATE NEGATIVE SKEW:
![f9](https://user-images.githubusercontent.com/118889143/234240881-11bbb9a5-e833-4326-93cc-c933bf5b3bca.png)

# LOG OF MODERATE POSITIVE SKEW:
![f10](https://user-images.githubusercontent.com/118889143/234240938-ace8674a-7c17-4225-9282-8a3ae529180a.png)

# LOG OF HIGHLY POSITIVE SKEW:
![f11](https://user-images.githubusercontent.com/118889143/234240987-97eac9fb-0ac4-44f5-9c27-868ae3baf5dc.png)

# RECIPROCAL OF HIGHLY POSITIVE SKEW:
![f12](https://user-images.githubusercontent.com/118889143/234241055-670155b3-42ad-40b1-bc63-8311821a35d8.png)

# SQUARE ROOT TRANSFORMATION:
![f13](https://user-images.githubusercontent.com/118889143/234241146-13dc8e40-b46f-409f-8fb2-ee934b7f138f.png)

# POWER TRANSFORMATION OF MODERATE NEGATIVE SKEW:
![f14](https://user-images.githubusercontent.com/118889143/234241180-7157af4c-b21b-46b0-8cac-0f276b49d48e.png)

# QUANTILE TRANSFORMATION:
![f15](https://user-images.githubusercontent.com/118889143/234241240-cca8614e-33d4-41b7-9ec1-99f0144a2e8f.png)

# RESULT:
Thus, Feature transformation is performed and executed successfully for the given dataset
