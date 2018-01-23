import pandas as pd
import csv

def isEmpty(obj):
	return obj == None or obj == ''

train_csv = csv.reader(open('../data/train.csv',encoding='utf8'))
train_list = list(train_csv)

test_csv = csv.reader(open('../data/test.csv',encoding='utf8'))
test_list = list(test_csv)


train_data = pd.DataFrame(train_list[1:],columns=train_list[0])
test_data = pd.DataFrame(test_list[1:],columns=test_list[0])

train_data = train_data.drop(['Ticket','Cabin','PassengerId'],axis=1)
test_data =   test_data.drop(['Ticket','Cabin'],axis=1)

combine = [train_data,test_data]

# FamilySize
for dataSet in combine:
	dataSet['SibSp'] = dataSet['SibSp'].fillna(dataSet['SibSp'].dropna().mode()[0]).astype(int)
	dataSet['Parch'] = dataSet['Parch'].fillna(dataSet['Parch'].dropna().mode()[0]).astype(int)
	dataSet['FamilySize'] = dataSet['SibSp'] + dataSet['Parch']

# sex
sex_mapping = {'female':0,'male':1}
for dataSet in combine:
	dataSet['Sex'] = dataSet['Sex'].fillna(dataSet['Sex'].mode()[0])
	dataSet['Sex'] = dataSet['Sex'].map(sex_mapping).astype(int)

# Embarked
for dataSet in combine:
	freq_port = dataSet['Embarked'].mode()[0]
	dataSet['Embarked'] = dataSet['Embarked'].fillna(freq_port)





