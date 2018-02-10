import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# def c2num_mapping():
# 	c_str = 'abcdefghijklmnopqrstuvwxyz'
# 	dic = {}
# 	for c in c_str:
# 		for i in range(len(c_str)):
# 			dic[c_str[i]] = i
# 	return dic


# mapping = c2num_mapping()
# train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
# columns = train.columns[train.dtypes=='object']
# for column in columns:
# 	train[column] = train[column].map(mapping)
# 	test[column] = test[column].map(mapping)



# # print(train)
# X_train = train.drop(['Score','Id'],axis=1)
# Y_train = train['Score']
# X_test = test.drop('Id',axis=1).copy()
# # svc = SVC(decision_function_shape='ovo')
# # svc.fit(X_train,Y_train)
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
# submission = pd.DataFrame({
# 	"Id":test["Id"],
# 	"Score":Y_pred
# 	})

# submission.to_csv('../tmp/svc_submission_7.csv')

print("ssss"+"dddb")