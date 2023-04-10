import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import statsmodels.api as sm

#import dataset
df = pd.read_csv("data/train.csv")
df2 = pd.read_csv("data/test.csv")
#observing dataset
df.head(10)
df.info()
df.describe()
df['Neighborhood'].value_counts()
sns.distplot(df['SalePrice'])

#使用scatterplot觀察兩變數的關係
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=df)
sns.scatterplot(x='YearBuilt', y='SalePrice', data=df)
#使用boxplot觀察兩變數的關係
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
sns.boxplot(x='Neighborhood', y='SalePrice', data=df)
#觀察correlation
corr_matrix = df.corr()
print(df.corr())
#使用heatmap觀察所有變數的關係
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, square=True)
#使用pairplot觀察四變數的關係
sns.pairplot(df, vars=["SalePrice", "OverallQual", "GrLivArea", "GarageCars"])
sns.pairplot(df)

#prepare to train model!
#檢查missing values
missingV = df.isnull().sum()
print("Total missing values in the dataset: ", missingV.sum())
print("\nColumns with missing values:\n", missingV[missingV > 0])

#確定missing values在哪個變數最多，然後丟掉他們
missing = missingV[missingV > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
#根據bar圖看到這五項卻露太多了，drop他們
df = df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
#Electrical只有一項缺漏
df = df.dropna(subset=['Electrical'])
#其餘的缺漏值fillna
#考慮變數的資料型態及分佈狀況
#若obj，則填入眾數mode；int，則填入中位數median
df.loc[:, 'LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df.loc[:, 'GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])
df.loc[:, 'GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
df.loc[:, 'GarageFinish'] = df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df.loc[:, 'GarageQual'] = df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df.loc[:, 'GarageCond'] = df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.loc[:, 'BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df.loc[:, 'BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df.loc[:, 'BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df.loc[:, 'BsmtFinType1'] = df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df.loc[:, 'BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df.loc[:, 'MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df.loc[:, 'MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
df.info()
id_nums = df.loc[:, 'Id']
print(id_nums)
#變數都數值化
#把要轉換的列提取出来並型成列表傳入函數中
cols = ['MSZoning', 'LotFrontage', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'SaleCondition']
df[cols] = df[cols].apply(lambda x: pd.factorize(x)[0])
df.info()

#prepare to train model!
#所有變數
var_names = df.columns.tolist()
print(var_names)
# 找出相關性大於 0.8 的變數
corr_matrix = df.corr()
high_corr_vars = []
for col in corr_matrix.columns:
    high_corr_cols = corr_matrix[corr_matrix[col] > 0.8].index.tolist()
    for high_corr_col in high_corr_cols:
        if high_corr_col != col and high_corr_col not in high_corr_vars:
            high_corr_vars.append(high_corr_col)
print(high_corr_vars)

#這些大於0.8：['1stFlrSF', 'TotalBsmtSF', 'TotRmsAbvGrd', 'GrLivArea', 'GarageArea', 'GarageCars']
X = df[['1stFlrSF', 'TotalBsmtSF', 'TotRmsAbvGrd', 'GrLivArea', 'GarageArea', 'GarageCars']]
#提出價格
y = df['SalePrice']
#重新命名變數就可以保留原有數據庫

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=54)

#using linear regression model & train
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#use model
predictions = reg.predict(X_test)
predictions
y_test

#evaluate model
from sklearn.metrics import r2_score
r2_score(y_test, predictions)

mpl.pyplot.scatter(y_test, predictions, color='blue', alpha=0.1)

import joblib
joblib.dump(reg, 'house-20230410.pkl', compress=3)