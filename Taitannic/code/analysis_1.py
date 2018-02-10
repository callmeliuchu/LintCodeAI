import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
combine = [train_data,test_data]


obj_columns = train_data.columns[train_data.dtypes == 'object']
# print(obj_columns)
# print(train_data.info())
# print(test_data.info())
# print(train_data.head(10))
columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

#    Pclass  Survived
# 0       1  0.629630
# 1       2  0.472826
# 2       3  0.242363
# class_result = train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(class_result)


# Sex
#       Sex  Survived
# 0  female  0.742038
# 1    male  0.188908
# sex_result = train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(sex_result)


# SibSp
#    SibSp  Survived
# 1      1  0.535885
# 2      2  0.464286
# 0      0  0.345395
# 3      3  0.250000
# 4      4  0.166667
# 5      5  0.000000
# 6      8  0.000000
# sibsp_result = train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(sibsp_result)



# Parch
#    Parch  Survived
# 3      3  0.600000
# 1      1  0.550847
# 2      2  0.500000
# 0      0  0.343658
# 5      5  0.200000
# 4      4  0.000000
# 6      6  0.000000
# parch_result = train_data[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(parch_result)
#    FamilySize  Survived
# 3           4  0.724138
# 2           3  0.578431
# 1           2  0.552795
# 6           7  0.333333
# 0           1  0.303538
# 4           5  0.200000
# 5           6  0.136364
# 7           8  0.000000
# 8          11  0.000000
for dataSet in combine:
	dataSet['FamilySize'] = dataSet['SibSp'] + dataSet['Parch'] + 1
# family_result = train_data[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(family_result)
# print(train_data)

for dataSet in combine:
	dataSet['IsAlone'] = 0
	dataSet.loc[dataSet['FamilySize']==1,'IsAlone'] = 1

# g = sns.FacetGrid(train_data,col='Survived')
# g.map(plt.hist,'Age',bins=20)
# plt.show()

drop_columns = ['Ticket','Cabin']
train = train_data.drop(drop_columns,axis=1)
test = test_data.drop(drop_columns,axis=1)
combine = [train,test]


for dataSet in combine:
	dataSet['Sex'] = dataSet['Sex'].map({'female':1,'male':0}).astype(int)



guess_ages = np.zeros((2,3))
for dataSet in combine:
	for i in range(2):
		for j in range(3):
			guess = dataSet['Age'][(dataSet['Sex'] == i) & (dataSet['Pclass'] == j+1)].dropna().median()
			guess_ages[i][j] = guess
	for i in range(2):
		for j in range(3):
			dataSet.loc[dataSet['Age'].isnull() & (dataSet['Sex'] == i) & (dataSet['Pclass'] == j+1),'Age']= guess_ages[i][j]

	dataSet['Age'] = dataSet['Age'].astype(int)


train['AgeBand'] = pd.cut(train['Age'],5)
#        AgeBand  Survived
# 0  (-0.08, 16]  0.550000
# 3     (48, 64]  0.434783
# 2     (32, 48]  0.412037
# 1     (16, 32]  0.337374
# 4     (64, 80]  0.090909
# ageband_res = train[['AgeBand','Survived']].groupby('AgeBand',as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(ageband_res)

for dataSet in combine:
	dataSet.loc[dataSet['Age']<=16,'Age'] = 0
	dataSet.loc[(dataSet['Age']>16) & (dataSet['Age']<=32),'Age'] = 1
	dataSet.loc[(dataSet['Age']>32) & (dataSet['Age']<=48),'Age'] = 2
	dataSet.loc[(dataSet['Age']>48) & (dataSet['Age']<=64),'Age'] = 3
	dataSet.loc[(dataSet['Age']>64),'Age'] = 4

train = train.drop(['AgeBand'],axis=1)
combine = [train,test]


freq_embark = train['Embarked'].dropna().mode()[0]


for dataSet in combine:
	dataSet['Embarked'] = dataSet['Embarked'].fillna(freq_embark)

embarked_res = train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(embarked_res)


for dataSet in combine:
	dataSet['Embarked'] = dataSet['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)


test['Fare'].fillna(test['Fare'].dropna().median(),inplace=True)

for dataSet in combine:
	dataSet['Age*Pclass'] = dataSet['Age']*dataSet['Pclass']
#          FareBand  Survived
# 3   (31, 512.329]  0.581081
# 2    (14.454, 31]  0.454955
# 1  (7.91, 14.454]  0.303571
# 0       [0, 7.91]  0.197309
train['FareBand'] = pd.qcut(train['Fare'],4)
fareband_res = train[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(fareband_res)


