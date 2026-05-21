# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
```python
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
![311415580-27aae021-a93c-4d62-8700-79b1beaba84e](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/1fd0dccf-1586-4fc2-badd-b9016805b254)

```python
dt.info()
```
![311415718-23c9a813-e29e-458e-b872-c0e96d2b75d4](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/924badbe-1ca7-4b87-89dc-96bc887deeb6)

```python
dt.shape
```
![311415809-9b3f8940-f34a-4efe-857c-5c3e286a59f2](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/c8ef2943-0773-4d27-be5c-24f802472f95)

```python
dt.set_index("PassengerId",inplace=True)
dt.describe()
```
![311417476-603a6476-0b6e-4939-be1d-020a5411eb5a](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/012c818e-74d7-4859-9235-46667a0e53d0)

```python
dt.nunique()
```
![311417491-e5e1049e-0421-4ed2-9d7c-15e0a7f0317b](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/b033cbdf-bb3f-433e-904a-ff4f1746353c)

```python
dt["Survived"].value_counts()
```
![311415948-19899d18-deb2-459d-8d6f-9610872d098f](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/6b232f0a-8f01-4e82-a87e-80f593354fbd)

```python
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![311416106-453f521f-e016-450a-823e-853513864c8a](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/93ba258a-99ca-411f-8585-1ecd000824bb)

```python
sns.countplot(data=dt,x="Survived")
```
![311416143-b38c5a88-9b85-4e68-a1fd-f25fb35da7d5](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/970c939e-4274-4f9b-85b1-1ee8caa034d1)

```python
dt
```
![311416203-75851857-515c-45a4-9ed0-f97bc501ac02](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/9cd750e3-a55d-4645-9883-14f9301da957)

```python
dt.Pclass.unique()
```
![311416245-4868a17d-a4de-45c3-86e4-c769762d77ca](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/e1f08259-26de-4b73-baab-8cb6d06a1a92)

```python
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![311416315-822c7791-fb41-44bb-8c23-8b8d1a3d811c](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/1c7c2c15-d9da-4de7-a45a-e6a16fe3d8d7)

```python
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5, aspect=.7)
```
![311416381-fdadc299-3453-47ae-86cd-8bd9e77120a1](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/79be55eb-eeaf-45fa-abe7-6662f9cd031c)

```python
sns.catplot(x='Survived',hue="Gender",data=dt,kind="count")
```
![311416409-7fb380ba-e219-4d79-aa78-c9591ddcb8b6](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/5e3f61ce-91ed-4a27-8f37-acc8a3abf518)

```python
dt.boxplot(column="Age",by="Survived")
```
![311416457-c84bad98-5cd5-4faf-9d15-85d0645205d9](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/39d14b7d-7fc4-4d25-8eac-648f30336659)

```python
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![311416485-089a4887-89a2-47b9-9f11-ef75e3016301](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/1cc80454-623c-4b60-96d3-a0b149b15175)

```python
sns.jointplot(x="Age",y="Fare",data=dt)
```
![311416523-3b43d8e9-fb94-4e0f-ac9a-d31b70eaeb77](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/2460cad5-d90d-46e6-854e-d3f31c9e2854)

```python
import matplotlib.pyplot as plt
fig, ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=dt)
```
![311416701-a5473432-41ab-4722-872e-53037544fdeb](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/85ee603a-bc34-42bf-a205-28582503910b)

```python
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![311416730-7b72bb3b-6824-479c-8703-3d5378b2023e](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/095f2bec-d345-46a2-a1b0-6ae782d7b96d)

```python
#co-relation
import seaborn as sns
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![311416761-6b0e3e5a-85cc-489e-92dc-235451d57476](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/061af082-14ef-4dbd-a8ef-0a465a41f335)

```python
sns.pairplot(dt)
```
![311416795-eb83a5af-b99a-41fd-9c84-e0c828434964](https://github.com/KesavDeepak/EXNO2DS/assets/139336019/a17306b9-e877-42b9-b4a4-47fb7b72270a)

# RESULT:-
The code is successfully executed.
