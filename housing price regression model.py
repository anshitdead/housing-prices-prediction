# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('train.csv')
dataset2 = pd.read_csv('test.csv')
#dataset.isnull().sum().sum()
#preparing the dataset
dataset1 = dataset1.fillna(dataset1.mean())
dataset1.drop(['Id', 'PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu'], axis=1, inplace=True)
dataset1 = dataset1.dropna(subset=['GarageType','GarageFinish','GarageQual','GarageCond'], how='all')
dataset1 = dataset1.dropna(subset=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], how='all')
dataset1 = dataset1.dropna(subset=['MasVnrType', 'Electrical', 'BsmtExposure', 'BsmtFinType2'], how='any')
dataset1 = pd.get_dummies(dataset1, columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'], drop_first=True)
y1 = dataset1.iloc[:,36].values
y1 = pd.DataFrame(y1)
dataset1.drop(['SalePrice'], axis=1, inplace=True)
X1 = dataset1.iloc[:, :].values
X1 = pd.DataFrame(X1)


#preparing the test dataset
dataset2 = dataset2.fillna(dataset2.mean())
dataset2.drop(['Id', 'PoolQC', 'Fence', 'MiscFeature', 'Alley', 'FireplaceQu'], axis=1, inplace=True)
dataset2['MSZoning'].fillna(value='RL', inplace=True)
dataset2['Utilities'].fillna(value='AllPub', inplace=True)
dataset2['Exterior1st'].fillna(value='VinylSd', inplace=True)
dataset2['Exterior2nd'].fillna(value='VinylSd', inplace=True)
dataset2['MasVnrType'].fillna(value='None', inplace=True)
dataset2['BsmtQual'].fillna(value='TA', inplace=True)
dataset2['BsmtCond'].fillna(value='TA', inplace=True)
dataset2['BsmtExposure'].fillna(value='No', inplace=True)
dataset2['BsmtFinType1'].fillna(value='GLQ', inplace=True)
dataset2['BsmtFinType2'].fillna(value='Unf', inplace=True)
dataset2['Heating'].fillna(value='GasA', inplace=True)
dataset2['HeatingQC'].fillna(value='Ex', inplace=True)
dataset2['CentralAir'].fillna(value='Y', inplace=True)
dataset2['Electrical'].fillna(value='SBrkr', inplace=True)
dataset2['KitchenQual'].fillna(value='TA', inplace=True)
dataset2['Functional'].fillna(value='Typ', inplace=True)
dataset2['GarageType'].fillna(value='Attchd', inplace=True)
dataset2['GarageFinish'].fillna(value='Unf', inplace=True)
dataset2['GarageQual'].fillna(value='TA', inplace=True)
dataset2['GarageCond'].fillna(value='TA', inplace=True)
dataset2['SaleType'].fillna(value='WD', inplace=True)
#dataset2.isnull().sum().sum()
dataset2 = pd.get_dummies(dataset2, columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'], drop_first=True)


tmp= np.zeros((1459,1), dtype=int)
dataset2.insert(47, "Utilities_NoSeWa", tmp)
dataset2.insert(90, "Condition2_RRAe", tmp)
dataset2.insert(91, "Condition2_RRAn", tmp)
dataset2.insert(92, "Condition2_RRNn", tmp)
dataset2.insert(99, "HouseStyle_2.5Fin", tmp)
dataset2.insert(109, "RoofMatl_CompShg", tmp)
dataset2.insert(110, "RoofMatl_Membran", tmp)
dataset2.insert(111, "RoofMatl_Metal", tmp)
dataset2.insert(112, "RoofMatl_Roll", tmp)
dataset2.insert(122, "Exterior1st_ImStucc", tmp)
dataset2.insert(204, "GarageQual_Fa", tmp)
X2 = dataset2.iloc[:, :].values
X2 = pd.DataFrame(X2)



#df.insert(1,'square_raw',df['raw']**3)




#Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy='mean', axis = 0)
#imputer = imputer.fit(dataset.values[:, 0:889])
#dataset.values[:, 0:889] = imputer.transform(dataset.values[:, 0:889])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X2 = sc_X.transform(X2)

#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X1, y1)

# Predicting the Test set results
y_pred = regressor.predict(X2)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()