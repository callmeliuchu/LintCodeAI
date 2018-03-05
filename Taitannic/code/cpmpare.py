import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# print(train_data)

# 合并数据集
combine = [train_data, test_data]

# we have 12 features
# print(train_data.columns.values)
# print(len(train_data.columns.values))
# print(train_data.head())

# 查看训练数据集信息,训练集数据中,特征Age, Cabin, Embarked有缺少值
# Age, Cabin and Embarked are the features that have missing values in the Train dataset
# print(train_data.info())

# 查看测试数据集信息,测试集数据中,特征Age, Fare, Cabin有缺少值
# Age, Fare and Cabin are the features that have missing values in the Test dataset
# print(test_data.info())

# 训练数据集特征描述
# Survived is a categorical feature (0 or 1)
# Pclass (Passenger class) have values 1,2 or 3
# Minimum age of the passengers is 4 months and maximum age is 80
# 52% of passengers have their siblings on board
# 38% of passengers had their parents on board
# Fare varies hugely, leaping to a maximum of $ 512.32

# print(train_data.describe())

# 乘客类型 passengers class
#    Pclass  Survived
# 0       1  0.629630
# 1       2  0.472826
# 2       3  0.242363
# Survived_result = train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(Survived_result)

# 性别
#       Sex  Survived
# 0  female  0.742038
# 1    male  0.188908
# Sex_result = train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(Sex_result)

# 兄弟姐妹(这个地方可以包括配偶)
#    SibSp  Survived
# 1      1  0.535885
# 2      2  0.464286
# 0      0  0.345395
# 3      3  0.250000
# 4      4  0.166667
# 5      5  0.000000
# 6      8  0.000000
# SibSp_result = train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(SibSp_result)

# 家庭
#    Parch  Survived
# 3      3  0.600000
# 1      1  0.550847
# 2      2  0.500000
# 0      0  0.343658
# 5      5  0.200000
# 4      4  0.000000
# 6      6  0.000000
# Parch_result = train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(Parch_result)

# 家庭成员人数
#    FamilySize  Survived
# 0           1  0.303538
# 1           2  0.552795
# 2           3  0.578431
# 3           4  0.724138
# 4           5  0.200000
# 5           6  0.136364
# 6           7  0.333333
# 7           8  0.000000
# 8          11  0.000000
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1  # 这里 +1 指的是孤身一人
# print(train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

# 是否孤身一人
#    IsAlone  Survived
# 0        0  0.505650
# 1        1  0.303538
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# print (train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# 年龄存活比例图
# g = sns.FacetGrid(train_data, col='Survi
#ved')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

# In the below chart, we are visualizing 4 attributes : Fare, Embarked, Survived and sex
# grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.8, ci=None)
# grid.add_legend()
# plt.show()

# colormap = plt.cm.viridis
# plt.figure(figsize=(12, 12))
# plt.title('Feature correlations', y=1.05, size=15)
# sns.heatmap(train_data.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()

# 重构数据集 - 去除Ticket,Cabin特征
train = train_data.drop(['Ticket', 'Cabin'], axis=1)
test = test_data.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]

for dataset in combine:
    dataset['Salutation'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# print(pd.crosstab(train['Salutation'], train['Sex']))

# 乘客敬语(尊称) 存活比例
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
Salutation_result = train[['Salutation', 'Survived']].groupby(['Salutation'], as_index=False).mean()
# print(Salutation_result)

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Salutation'] = dataset['Salutation'].map(Salutation_mapping)
    dataset['Salutation'] = dataset['Salutation'].fillna(0)
# print(train.head())

# 继续重构数据集 - 删除不需要的特征
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
# print(train.head())

# 将性别属性转化为int类型
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
# print(train.head())

grid = sns.FacetGrid(train, row='Pclass', col='Sex')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# plt.show()

# 创建一个空的 2 X 3 矩阵
guess_ages = np.zeros((2, 3))
# print(guess_ages)

# 猜测乘客的年龄
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()

            age_guess = guess.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            # 判断数据集中年龄为空的记录并将猜测的年龄赋值
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)
# print(train.head())

# 将年龄划分为5个年龄段,计算各段的存活率
#          AgeBand  Survived
# 0  (-0.08, 16.0]  0.550000
# 1   (16.0, 32.0]  0.337374
# 2   (32.0, 48.0]  0.412037
# 3   (48.0, 64.0]  0.434783
# 4   (64.0, 80.0]  0.090909
train['AgeBand'] = pd.cut(train['Age'], 5)
AgeBand_result = train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# print(AgeBand_result)

# 重构数据,将各个年龄段用0~4代表
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
# print(train.head())
train = train.drop(['AgeBand'], axis=1)
combine = [train, test]
# print(train.head())

freq_port = train.Embarked.dropna().mode()[0]

# 计算从各个登船口登船的存活率
#   Embarked  Survived
# 0        C  0.553571
# 1        Q  0.389610
# 2        S  0.339009
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
Embarked_result = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# print(Embarked_result)

# 重构数据集
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
# print(train.head())

# print(test.head())
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
# print(test.head())

# 将票价切分为3段
train['FareBand'] = pd.qcut(train['Fare'], 3)
# 计算每段的存活率
#           FareBand  Survived
# 0  (-0.001, 8.662]  0.198052
# 1    (8.662, 26.0]  0.402778
# 2  (26.0, 512.329]  0.559322
FareBand_result = train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
# print(FareBand_result)

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

# 重构数据
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]
# print(train.head(10))

colormap = plt.cm.viridis
plt.figure(figsize=(12, 12))
plt.title('Feature correlations', y=1.05, size=15)
sns.heatmap(train.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()

# 现在,我们尝试各种算法测试

# 1 逻辑回归 Logistic Regression
X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']
# print(X_train)
# print(Y_train)
X_test = test.drop("PassengerId", axis=1).copy()
# print(X_test.head())
# print(X_train.shape, Y_train.shape, X_test.shape)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

# 2 支持向量机 Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)

# 3 K-近邻 KNN or k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)

# 4 朴素贝叶斯分类 Naive Bayes classifier
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)

# 5 感知器 Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)

# 6 线性补偿 Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)

# 7 决策树 Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)

# 8 随机森林 Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

# 9 随机梯度下降法 Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)

# 评价模型,各个算法评分
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models)
# print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
print(submission.to_string())