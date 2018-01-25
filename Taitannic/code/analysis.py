import pandas as pd


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

print(train_data)
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