import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
# train = train.drop(['PassengerId'],axis=1)
# test = test.drop(['PassengerId'],axis=1)
combine = [train_data,test_data]

# we have 12 features
# ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
#  'Ticket' 'Fare' 'Cabin' 'Embarked']
# print(train_data.columns.values)
# print(len(train_data.columns.values))


#see none value
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
# Sex            891 non-null object
# Age            714 non-null float64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Ticket         891 non-null object
# Fare           891 non-null float64
# Cabin          204 non-null object
# Embarked       889 non-null object
# dtypes: float64(2), int64(5), o
# print(train_data.info())


# dataset description
# print(train_data)
# print(train_data.describe())

# print(train_data)
# passenger types
#    Pclass  Survived
# 0       1  0.629630
# 1       2  0.472826
# 2       3  0.242363
# Survived_result = train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(Survived_result)

# sex Survived
#       Sex  Survived
# 0  female  0.742038
# 1    male  0.188908
# Survived_sex_result = train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(Survived_sex_result)


# family survived
# 1      1  0.535885
# 2      2  0.464286
# 0      0  0.345395
# 3      3  0.250000
# 4      4  0.166667
# 5      5  0.000000
# 6      8  0.000000
# Survived_family_result = train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(Survived_family_result)

# family size Survived

for dataSet in combine:
	dataSet['FamilySize'] = dataSet['SibSp'] + dataSet['Parch'] + 1
# print(train_data.columns)
# print(train_data)
# 3           3  0.724138
# 2           2  0.578431
# 1           1  0.552795
# 6           6  0.333333
# 0           0  0.303538
# 4           4  0.200000
# 5           5  0.136364
# 7           7  0.000000
# 8          10  0.000000
# family_size_survived = train_data[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(family_size_survived)



# isAlone
for dataSet in combine:
	dataSet['IsAlone'] = 0
	dataSet.loc[dataSet['FamilySize'] == 1,'IsAlone'] = 1
# 0        0  0.505650
# 1        1  0.303538
# isAlone_res = train_data[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(isAlone_res)
# print(train_data)

# delete the useless features
# for dataSet in combine:
# 	dataSet = dataSet.drop(['Ticket','Cabin'],axis=1)

train = train_data.drop(['Ticket','Cabin'],axis=1)
test = test_data.drop(['Ticket','Cabin'],axis=1)
combine = [train,test]
# print(train_data)
# print(train.dtypes)

for dataSet in combine:
	dataSet['Salutation'] = dataSet['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)


#   Salutation  Survived
# 0     Master  0.575000
# 1       Miss  0.702703
# 2         Mr  0.156673
# 3        Mrs  0.793651
# 4       Rare  0.347826
for dataset in combine:
    dataset['Salutation'] = dataset['Salutation'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Salutation'] = dataset['Salutation'].replace('Mlle', 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Ms', 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Mme', 'Mrs')
# Salutation_result = train[['Salutation','Survived']].groupby(['Salutation'],as_index=False).mean()
# print(Salutation_result)
# so = train.Salutation
# for d in so:
# 	print(d)

Salutation_mapping = {'Mr':1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}
for dataSet in combine:
	dataSet['Salutation'] = dataSet['Salutation'].map(Salutation_mapping)
	dataSet['Salutation'] = dataSet['Salutation'].fillna(0)
# print(train['Salutation'])
# print(train.head())
# print(train.dtypes)

train = train.drop(['Name','PassengerId'],axis=1)
test = test.drop(['Name'],axis=1)
combine = [train,test]
# print(test)


for dataSet in combine:
	dataSet['Sex'] = dataSet['Sex'].map({'female':0,'male':1}).astype(int)
# print(train)
# print(train.dtypes)
guess_ages = np.zeros((2,3))
# print(guess_ages)
for dataSet in combine:
	for i in range(0,2):
		for j in range(0,3):
			guess = dataSet[(dataSet['Sex'] == i) & (dataSet['Pclass'] == j+1)]['Age'].dropna()
			# print(guess.median())
			age_guess = guess.median()
			guess_ages[i,j] = int(age_guess/0.5 + 0.5)*0.5

	for i in range(0,2):
		for j in range(0,3):
			dataSet.loc[(dataSet['Age'].isnull()) & (dataSet['Sex'] == i) & (dataSet['Pclass'] == j+1),'Age'] = guess_ages[i,j]

	dataSet['Age'] = dataSet['Age'].astype(int)

# band = pd.cut(train['Age'],5)
train['AgeBand'] = pd.cut(train['Age'],5)
AgeBand_result = train[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)


for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

train = train.drop(['AgeBand'],axis=1)
combine = [train,test]

# 2  (26, 512.329]  0.559322
# 1    (8.662, 26]  0.402778
# 0     [0, 8.662]  0.198052
# train['Fare'] = pd.qcut(train['Fare'],3)
# Fare_result = train[['Fare','Survived']].groupby(['Fare'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(Fare_result)
test['Fare'].fillna(test['Fare'].dropna().median(),inplace=True)
# print(train['Fare'].isnull().sum())


# train['FareBand'] = pd.qcut(train['Fare'],3)
# print(train)


for dataSet in combine:
	dataSet['Age*Class'] = dataSet.Age*dataSet.Pclass
# s = train.loc[:,['Age*Class','Age','Pclass']].head(10)
# print(s)
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#   Embarked  Survived
# 0        C  0.553571
# 1        Q  0.389610
# 2        S  0.336957
# res = train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(res)

# train['Embarked'].fillna(train['Embarked'].dropna().mode(),inplace=False)
# print(train['Embarked'].dropna().mode())
# print(train['Embarked'].isnull().sum())

freq_port = train['Embarked'].dropna()[0]
for dataSet in combine:
	dataSet['Embarked'] = dataSet['Embarked'].dropna().mode()[0]
# print(train['Embarked'].isnull().sum())

for dataSet in combine:
	dataSet['Embarked'] = dataSet['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

# print(train)
# test = test.drop(['PassengerId'],axis=1)
# print(test)

# print(train)

X_train = train.drop(['Survived'],axis=1)
Y_train = train['Survived']
X_test = test.drop('PassengerId',axis=1).copy()

# logreg = LogisticRegression()
# logreg.fit(X_train,Y_train)
# Y_pred = logreg.predict(X_test)
# print(Y_pred)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,Y_train)
Y_pred = decision_tree.predict(X_test)


submission = pd.DataFrame({
	"PassengerId":test["PassengerId"],
	"Survived":Y_pred
	})
print(submission)
submission.to_csv("../tmp/submission_decision.csv")
