import pandas as pd


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train = train.drop(['PassengerId'],axis=1)
test = test.drop(['PassengerId'],axis=1)
combine = [train,test]

print(train.columns)
print(train.dtypes)
for dataSet in combine:
	print(dataSet)

